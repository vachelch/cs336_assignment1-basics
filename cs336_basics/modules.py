import math
import torch
from torch import Tensor
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from einops import einsum, rearrange
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device = device, dtype = dtype)
        )
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, 0.0, std, -3 * std, 3 * std)

    def forward(
            self, x: Float[Tensor, " ... in_features"]
    ) -> Float[Tensor, " ... out_features"]:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

class Embedding(nn.Module):
    def __init__(
        self, 
        num_embeddings: int, 
        embedding_dim: int, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device = device, dtype = dtype)
        )
        std = 1.0
        nn.init.trunc_normal_(self.weight, 0.0, std, -3 * std, 3 * std)

    def forward(
            self, token_ids: Int[Tensor, " ..."]
    ) -> Float[Tensor, " ... embedding_dim"]:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        eps: float = 1e-5, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(d_model, device = device, dtype = dtype)
        )

    def forward(self, x: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
        in_type = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt((x**2).mean(dim = -1, keepdim = True) + self.eps)
        res = x / rms * self.weight
        return res.to(in_type)

def silu(x: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        d_ff: int | None = None,
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()

        if d_ff is None:
            d_ff = int(d_model * 8 / 3)
            d_ff = (d_ff + 63) // 64 * 64

        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        return self.w2(silu(self.w1(x)) * self.w3(x))

class ROPE(nn.Module):
    def __init__(
        self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()

        assert d_k % 2 == 0
        # angle = m * 10000 ^ (-2i / d), [seq_len, d_k/2]
        pos = torch.arange(max_seq_len, device = device, dtype = torch.float32) # (max_seq_len,)

        inv_freq = theta ** (-1.0 * torch.arange(0, d_k, 2, device = device, dtype = torch.float32)  / d_k) # (d_k/2,)
        # (seq_len,) -> (seq_len, 1), (d_k/2,) -> (1, d_k/2)
        angle = einsum(pos, inv_freq, "seq_len, half -> seq_len half") # (max_seq_len, d_k/2)

        self.register_buffer("cos_cached", torch.cos(angle), persistent = False)
        self.register_buffer("sin_cached", torch.sin(angle), persistent = False)

    def forward(
        self,
        x: Float[Tensor, " ... seq_len d_k"],
        token_positions: Int[Tensor, " ... seq_len"]
    ) -> Float[Tensor, " ... seq_len d_k"]:
        # R = [[cos, -sin], [sin, cos]]
        # X: [a1, b1, ...], X @ R.T = [a1 * cos - b1 * sin, a1 * sin + b1 * cos]
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        res = torch.empty_like(x)
        res[..., ::2] = x_even * cos - x_odd * sin
        res[..., 1::2] = x_even * sin + x_odd * cos
        return res

def softmax(x: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    in_type = x.dtype
    x = x.to(torch.float32)
    x_max = x.max(dim = dim, keepdim = True).values
    x_exp = (x - x_max).exp()
    res = x_exp / x_exp.sum(dim = dim, keepdim = True)
    return res.to(in_type)

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.shape[-1]
    atten_score = Q @ K.transpose(-2,-1) / math.sqrt(d_k)
    if mask is not None:
        atten_score = atten_score.masked_fill(~mask, float("-inf"))
    return softmax(atten_score, dim = -1) @ V

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self, 
        d_model: int,
        num_heads: int,
        rope: ROPE | None = None,
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()

        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.q_proj = Linear(d_model, d_model, device, dtype)
        self.k_proj = Linear(d_model, d_model, device, dtype)
        self.v_proj = Linear(d_model, d_model, device, dtype)
        self.output_proj = Linear(d_model, d_model, device, dtype)
        self.rope = rope

    def forward(
        self, 
        x: Float[Tensor, " ... seq_len d_model"],
        token_positions: Int[Tensor, " ... seq_len"] | None = None,
        mask: Bool[Tensor, " ... queries keys"] | None = None
    ) -> Float[Tensor, " ... seq_len d_model"]:
        seq_len = x.shape[-2]
        q = rearrange(self.q_proj(x), "... seq_len (num_heads h) -> ... num_heads seq_len h", num_heads = self.num_heads)
        k = rearrange(self.k_proj(x), "... seq_len (num_heads h) -> ... num_heads seq_len h", num_heads = self.num_heads)
        v = rearrange(self.v_proj(x), "... seq_len (num_heads h) -> ... num_heads seq_len h", num_heads = self.num_heads)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device = x.device)
            token_positions = rearrange(token_positions, "... seq_len -> ... 1 seq_len")
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        if mask is None:
            mask = torch.tril(
                torch.ones(seq_len, seq_len, device = x.device, dtype = torch.bool))
        if True:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask = mask)
        else:
            out = scaled_dot_product_attention(q, k, v, mask)
        out = rearrange(out, "... num_heads seq_len h_size -> ... seq_len (num_heads h_size)")
        return self.output_proj(out)

class TransformerBlock(nn.Module):
    def __init__(
        self, 
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: ROPE,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope, device, dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)
        self.ln1 = RMSNorm(d_model, device = device, dtype = dtype)
        self.ln2 = RMSNorm(d_model, device = device, dtype = dtype)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def forward(
        self, 
        x: Float[Tensor, " ... seq_len d_model"], 
        token_positions: Int[Tensor, " ... seq_len"] | None = None,
        mask: Bool[Tensor, " ... queries keys"] | None = None
    ) -> Float[Tensor, " ... seq_len d_model"]:
        # keep residual stream in fp32, x is fp32, branches run in param dtype
        x_att = x + self.attn(self.ln1(x.to(self.dtype)), token_positions, mask)
        x_out = x_att + self.ffn(self.ln2(x_att.to(self.dtype)))
        return x_out

class TransformerLM(nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        context_length: int, 
        num_layers: int,
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        rope_theta: float = 10000.0, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None,
        eos_id: int | None = None
    ):
        super().__init__()

        self.eos_id = eos_id
        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
        rope = ROPE(rope_theta, d_model // num_heads, context_length, device)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, rope, device, dtype) for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device = device, dtype = dtype)
        self.lm_head = Linear(d_model, vocab_size, device, dtype)

    def forward(
        self, 
        x: Int[Tensor, " ... seq_len"]
    ) -> Float[Tensor, " ... seq_len vocab_size"]:
        seq_len = x.shape[-1]
        assert seq_len <= self.context_length

        # document aware attention
        mask = torch.tril(
            torch.ones(seq_len, seq_len, device = x.device, dtype = torch.bool)
        )
        token_positions = torch.arange(seq_len, device = x.device)
        if self.eos_id is not None:
            doc_ids, token_positions = get_doc_ids_and_positions(x, self.eos_id)
            same_doc = doc_ids[..., :, None] == doc_ids[..., None, :]
            mask = (mask & same_doc)[..., None, :, :]

        # keep residual stream in fp32
        x = self.token_embeddings(x)
        in_dtype = x.dtype
        x = x.to(torch.float32)
        for layer in self.layers:
            x = layer(x, token_positions, mask)
        x = self.lm_head(self.ln_final(x.to(in_dtype)))
        return x

def get_doc_ids_and_positions(
    x: Int[Tensor, " ... seq_len"], eos_id
) -> tuple[Int[Tensor, " ... seq_len"], Int[Tensor, " ... seq_len"]]:
    # EOS belongs to the doc it ends
    is_eos = (x == eos_id).long()
    doc_ids = is_eos.cumsum(dim = -1) - is_eos

    # restart positions at 0 for each doc: position = index - doc start index
    index = torch.arange(x.shape[-1], device = x.device).expand_as(x)
    is_start = torch.ones_like(x, dtype = torch.bool)
    is_start[..., 1:] = is_eos[..., :-1]
    start_index = torch.where(is_start, index, 0)
    return doc_ids, index - torch.cummax(start_index, dim = -1).values


    # is_eos = (x == eos_id).long()
    # doc_ids = torch.cumsum(is_eos, dim = -1) - is_eos

    # # restart positions at 0 for each doc: position = index - doc start index
    # idx = torch.arange(x.shape[-1], device = x.device).expand_as(x)
    # is_start = torch.ones_like(x, dtype = torch.bool)
    # is_start[..., 1:] = doc_ids[..., 1:] != doc_ids[..., :-1]
    # start = torch.where(is_start, idx, 0).cummax(dim = -1).values
    # return doc_ids, idx - start



def print_stats(
    model_name = "GPT-2 XL",
    context_length: int = 1024,
    vocab_size: int = 50257,
    d_model: int = 1600,
    d_ff: int | None = None,
    num_layers: int = 48
):
    if d_ff is None:
        d_ff = (int(d_model * 8 / 3) + 63) // 64 * 64

    print("\n" + "---------" * 10)
    print(f"{model_name} stats, T = {context_length}:")
    # param num & memory
    param_size = 2 * vocab_size * d_model + num_layers * (4 * d_model**2 + 3 * d_model * d_ff + 2 * d_model) + d_model
    print(f"(a) ~{param_size / 10**9 :.2f}B trainable parameters,", 
            f"~{4 * param_size / 10**9 :.2f}GB memory in FP32\n")

    # FLOPS
    qkv_proj = 3 * context_length * d_model**2
    attn = 2 * context_length**2 * d_model
    out_proj = context_length * d_model**2
    ffn = 3 * context_length * d_model * d_ff
    lm_head = context_length * d_model * vocab_size
    total_flops = 2* (num_layers * (qkv_proj + attn + out_proj + ffn) + lm_head)
    print(f"(b) ~{total_flops / 10**12 :.2f}TFLOPS\n")

    # FLOPS percentage
    components = ["atten_qkv_proj", "attn_scoring", "attn_out_proj", "ffn", "lm_head"]
    flops = [2 * num_layers * qkv_proj, 2 * num_layers * attn, 2 * num_layers * out_proj, 2 * num_layers * ffn, 2 * lm_head]
    percentages = sorted([(v / total_flops, v, name) for name, v in zip(components, flops)], reverse=True)
    print("(c) FLOPS for every model component")
    print("ratio\tflops\tcomponent")
    for ratio, v, name in percentages:
        print(f"{ratio * 100 :.2f}%\t{v/10**12 :.2f}TB\t{name}")
    

if __name__ == "__main__":
    # (d) As model size grows, the FFN and attention prokection (qkv&out) (∝ L·d²) take a larger share of FLOPs, while the LM head (∝ d·V) and attention scoring (∝ L·T²·d) take a smaller share; the LM head drops from 27% to under 5%.
    # (e) Total FLOPs rise ~38× (3.5 → 134 TFLOPs). Attention scores (QKᵀ, A·V) scale with T² and grow from 9% to 62% of the total, becoming dominant, while the FFN falls from 58% to 24%.
    
    print_stats("GPT-2 small", d_model = 768, num_layers = 12)
    print_stats("GPT-2 medium", d_model = 1024, num_layers = 24)
    print_stats("GPT-2 large", d_model = 1280, num_layers = 36)
    print_stats("GPT-2 XL", d_model = 1600, num_layers = 48)
    print_stats("GPT-2 XL", context_length = 16384, d_model = 1600, num_layers = 48)
    
