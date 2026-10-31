import torch
from torch import Tensor
from cs336_basics.train import Config
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.modules import TransformerLM, softmax

from jaxtyping import Float, Int


def sample_next_token(
    logits: Float[Tensor, " ... seq_len vocab_size"], 
    temperature: float, 
    top_p: float
) -> Int[Tensor, " ... 1"]:
    # shape: [batch, vocab_size]
    logits = logits[:, -1, :]
    if temperature <= 0:
        _, indices = torch.max(logits, dim = -1, keepdim = True)
        return indices

    # apply temperature
    scaled = logits.to(torch.float32) / temperature
    probs = softmax(scaled, dim = -1)

    # top_p
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cum_probs = sorted_probs.cumsum(dim = -1)
    kept = (cum_probs - sorted_probs) < top_p
    sorted_probs[~kept] = 0
    sorted_probs /= sorted_probs.sum(dim = -1, keepdim = True)
    
    # sample from probs
    sampled_indices = torch.multinomial(sorted_probs, num_samples = 1) # [B, 1]
    next_token = sorted_indices.gather(-1, sampled_indices)
    return next_token

@torch.no_grad()
def generate(
    prompt: str,
    tokenizer,
    model,
    context_length: int,
    stop_token: str = "<|endoftext|>",
    temperature: float = 1.0,
    top_p: float = 0.9,
    max_new_tokens: int = 256
) -> str:
    assert 0 < top_p <= 1.0
    assert len(prompt) > 0

    stop_id = tokenizer.encode(stop_token)[0]

    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(
        input_ids, 
        dtype = torch.long, 
        device = next(model.parameters()).device
    ).unsqueeze(0)

    for _ in range(max_new_tokens):
        logits = model(input_ids[:, -context_length:])
        nxt_token = sample_next_token(logits, temperature, top_p)
        if nxt_token.item() == stop_id: 
            break

        input_ids = torch.concat([input_ids, nxt_token], dim = 1)
    output = tokenizer.decode(input_ids[0].tolist())
    return output

if __name__ == "__main__":
    cfg = Config()

    vocab_filepath = "data/vocab/TinyStoriesV2-vocab.pkl"
    merges_filepath = "data/vocab/TinyStoriesV2-merges.pkl"
    tokenizer = Tokenizer.from_files(
        vocab_filepath = vocab_filepath,
        merges_filepath = merges_filepath,
        special_tokens = ["<|endoftext|>"],
    )

    model = TransformerLM(
        cfg.model.vocab_size,
        cfg.model.context_length,
        cfg.model.num_layers,
        cfg.model.d_model,
        cfg.model.num_heads,
        cfg.model.d_ff,
        cfg.model.rope_theta,
        cfg.model.device,
        cfg.model.dtype,
        cfg.model.eos_id,
    )
    ckpt_path = "data/checkpoints/ckpt_step_50000.pt"
    ckpt = torch.load(ckpt_path, map_location = cfg.model.device)
    model.load_state_dict(ckpt["model"])

    prompt = "Once upon a time, in a big forest"
    print("prompt:", prompt)
    print("generate: ")
    print("----------" * 10)
    output = generate(
        prompt, tokenizer, model, 
        context_length = cfg.model.context_length, temperature = 0.7, top_p = 0.8
    )