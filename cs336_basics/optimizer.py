import math
import torch
from torch import Tensor
import torch.nn as nn
from jaxtyping import Float, Int
from typing import Optional, Callable, Iterable
from einops import rearrange


# - log (exp(xi) / sum_i(exp(xi)))
# -(xi - v_max ) + log sum(exp(xi - v_max))
def cross_entropy(
        logits: Float[Tensor, " ... seq_len vocab_size"], 
        targets: Int[Tensor, " ... seq_len"]
    ) -> Float[Tensor, ""]:
    shifted = logits - logits.max(dim = -1, keepdim = True).values
    targets = rearrange(targets, "... -> ... 1")
    target_shifted = shifted.gather(-1, targets).squeeze(-1)
    log_sum_exp = torch.log(torch.exp(shifted).sum(dim = -1))
    return (-target_shifted + log_sum_exp).mean()

class AdamW(torch.optim.Optimizer):
    def __init__(
        self, params, lr = 1e-3, betas = (0.9, 0.999), weight_decay = 1e-2, eps = 1e-8
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        if eps < 0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        
        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "eps": eps,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.data

                # get state
                state = self.state[p]
                m = state.get("m", torch.zeros_like(g))
                v = state.get("v", torch.zeros_like(g))
                t = state.get("t", 1)

                # update params
                m = beta1*m + (1-beta1) * g
                v = beta2*v + (1-beta2) * g ** 2
                a = 1 / (1 - beta1 ** t)
                b = 1 / (1 - beta2 ** t)
                adjust = a / math.sqrt(b)
                p.data -= lr * weight_decay * p.data
                p.data -= lr * adjust * m / (torch.sqrt(v) + eps)

                # update state
                state["m"] = m
                state["v"] = v
                state["t"] = t + 1
        return loss

def get_lr_cosine_schedule(
    t: int,
    lr_max: float,
    lr_min: float,
    warmup_iters: int,
    cosine_cycle_iters: int
) -> float:
    if t < warmup_iters:
        return t / warmup_iters * lr_max
    if t >= cosine_cycle_iters:
        return lr_min
    angle = (t - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi
    return lr_min + 0.5 * (1 + math.cos(angle)) * (lr_max - lr_min)

def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter], 
    max_l2_norm: float,
    eps: float = 1e-6
) -> None:
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    total_norm = torch.stack([(g.data ** 2).sum() for g in grads]).sum()
    total_norm = torch.sqrt(total_norm).item()
    if total_norm <= max_l2_norm: 
        return
    scale = max_l2_norm / (total_norm + eps)
    for g in grads:
        g.data *= scale

def learning_rate_tuning(lr: float):
    print("----------"*10)
    print(f"lr = {lr}:\n")
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = torch.optim.SGD([weights], lr=lr)
    for t in range(10):
        opt.zero_grad()
        loss = (weights**2).mean()
        print(loss.cpu().item())
        loss.backward()
        opt.step()

def adam_accounting(
        bs: int = 3, V: int = 50257, T: int = 1024, L: int = 48, d: int = 1600, h: int = 25
    ):
    # peak memory, counting using float32
    ## parameters
    """
    P = 2Vd + L(12d^2 + 2d) + d
        * input emb, out head: 2Vd
        * 2 RMSNorm: 2d
        * qkv: 3d^2
        * out_proj: d^2
        * swiglu: 8d^2
        * final RMSNorm: d
    """
    num_params = 2 * V * d + L * (12 * d**2 + 2 * d) + d

    ## parameters = P, gradients = P, optimizer_state = 2*P
    ## activatinons
    """
    A = L * (56/3 BTd + 2BhT^2) + BTd + 2BTV
        * 2 RMSNorm: 2BTd
        * QKV proj: 3BTd
        * Q@K.T: BhT^2
        * softmax: BhT^2
        * attn @ V: BTd
        * out proj: BTd
        * swiglu: BT*d_ff * 4 + BTd = 35/3*BTd
        * final RMSNorm, logits, log-softamx: BTd + 2BTV
    """
    num_activations = L * (56/3 * bs * T * d + 2 * bs * h * T ** 2) + bs * T * d + 2 * bs * T * V
    print(f"peak memory in float32: {4 * (num_params * 4 + num_activations) / 1e9 :.2f}GB")
    print(f"  - params, gradients, optimizer_states takes: {4*4*num_params / 1e9 :.2f}GB")
    print(f"  - activations takes: {4*num_activations / 1e9 :.2f}GB, with batch_size = {bs}")

    # FLOPS
    """
    flops = forward + backward + optimizer ~= 3 * forward = 3(L*(24BTd^2 + 4BT^2d) + 2BTdV)
    * forward:
        * qkv_proj: 6BTd^2
        * attn = 4BT^2d
        * out_proj = 2BTd^2
        * ffn = 16BTd^2
        * lm_head = 2BTdV
    * backward = 2 * forward
    * optimizer = 13 * num_params
    """
    flops = 3 * (L * (24 * T * d**2 + 4 * T**2 * d) + 2 * T * d * V)
    print(f"Take {flops / 1e12 :.2f}TFLOPs for one step of AdamW (batch_size = 1)")

    # time counting for peak 495 TFLOP/s, 50% MFU, batch_size = 1024, 400k step
    seconds = 4e5 * flops * 1024 / (495e12 * 0.5)
    print(f"{seconds :.0f} seconds, {seconds / 24 / 3600 :.1f} days to train GPT-2 XL, "
          "for 400k step and batch size = 1024, "
          "on a single H100 (495 TFLOP/s for float32, MFU 50%)")


if __name__ == "__main__":
    # loss = mean(w**2) -> g = 1/100 * 2*w
    # w_ = w - lr * g = w * (1 - 0.02 * lr), loss_ = (w_**2).mean()
    # when |1 - 0.02 * lr| > 1, loss_ get bigger and bigger
    # learning_rate_tuning(1000)

    adam_accounting()

