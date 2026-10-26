import os, time
import typing
import torch
from torch import Tensor
from jaxtyping import Float, Int
import numpy.typing as npt
import numpy as np
from cs336_basics.modules import TransformerLM
from cs336_basics.optimizer import AdamW, cross_entropy, get_lr_cosine_schedule, gradient_clipping
from cs336_basics.util_device_bench import device_bench
from cs336_basics.optimizer import adam_accounting

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[Int[Tensor, "bs context_len"], Int[Tensor, "bs context_len"]]:
    assert len(dataset) > context_length

    starts = np.random.randint(0, len(dataset)-context_length, size = batch_size)
    data = np.stack([dataset[i: i + context_length+1] for i in starts])

    inputs = torch.tensor(data[:, :-1], dtype = torch.int64, device = device)
    targets = torch.tensor(data[:, 1:], dtype = torch.int64, device = device)

    return inputs, targets

def save_checkpoint(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    iteration: int, 
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(ckpt, out)

def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
):
    ckpt = torch.load(src, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["iteration"]

def split_decay_parameters(name_params):
    no_decay = []
    decay = []
    for name, p in name_params:
        if any(n in name for n in ("token_embeddings", "ln_final", "ln1", "ln2")):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": no_decay, "weight_decay": 0}, {"params": decay}]

def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class ModelConfig:
    vocab_size: int = 10000
    context_length: int = 256
    num_layers: int = 4
    d_model: int = 512
    num_heads: int = 16
    d_ff: int = 1344
    rope_theta: float = 10000.0
    device: str = default_device()
    dtype: torch.dtype = torch.float16
    eos_id: int | None = 0  # <|endoftext|> is id 0 in the TinyStories vocab

class OptimConfig:
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.01
    lr_max: float = 3e-4
    lr_min: float = 3e-5
    max_l2_norm: float = 1.0

class DataConfig:
    train_path: str = "data/tokenized/TinyStoriesV2_train_ids.bin"
    eval_path: str = "data/tokenized/TinyStoriesV2_valid_ids.bin"
    data_type: str = "uint16"
    batch_size: int = 32
    # 32 * 256 * 40000 ~= 327M tokens
    max_iter: int = 40000

class ArgsConfig:
    resume: bool = False
    ckpt_path: str = "checkpoints/tinystories.pt"
    log_interval: int = 5

class Config:
    model: ModelConfig = ModelConfig()
    optim: OptimConfig = OptimConfig()
    data: DataConfig = DataConfig()
    args: ArgsConfig = ArgsConfig()

def train():
    cfg = Config()

    # device benchmark
    bench_dict = device_bench()

    # current setting devcie peak flops/s
    peak_FLOPS = bench_dict[(cfg.model.device, str(cfg.model.dtype))]
    print(f"current setting, device: {cfg.model.device}, dtype: {str(cfg.model.dtype)}, peak_FLOPS: {peak_FLOPS/1e12 :.2f}TFLOP/s")

    # params, flops, memory
    FLOPs_step = adam_accounting(
        cfg.data.batch_size,
        cfg.model.vocab_size,
        cfg.model.context_length,
        cfg.model.num_layers,
        cfg.model.d_model,
        cfg.model.num_heads
    )

    train_data = np.memmap(cfg.data.train_path, np.dtype(cfg.data.data_type), mode = 'r')
    eval_data = np.memmap(cfg.data.eval_path, np.dtype(cfg.data.data_type), mode = 'r')

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
    param_groups = split_decay_parameters(model.named_parameters())
    optimizer = AdamW(
        param_groups,
        cfg.optim.lr, 
        cfg.optim.betas, 
        cfg.optim.weight_decay,
    )
    start_iter = 0
    if cfg.args.resume and os.path.exists(cfg.args.ckpt_path):
        start_iter = load_checkpoint(cfg.args.ckpt_path, model, optimizer)
        print(f"Resume from {cfg.args.ckpt_path} at iteration {start_iter}")

    t0 = time.time()
    tokens_per_step = cfg.data.batch_size * cfg.model.context_length
    for it in range(start_iter, cfg.data.max_iter):
        inputs, targets = get_batch(train_data, cfg.data.batch_size, cfg.model.context_length, cfg.model.device)
        lr = get_lr_cosine_schedule(
            it, cfg.optim.lr_max, cfg.optim.lr_min, cfg.data.max_iter // 10, cfg.data.max_iter
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        
        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), cfg.optim.max_l2_norm)
        optimizer.step()

        if it % cfg.args.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            steps = cfg.args.log_interval if it != start_iter else 1

            print(FLOPs_step, steps, dt, peak_FLOPS)
            real_FLOPS = FLOPs_step * steps / dt
            real_MFU = round(real_FLOPS / peak_FLOPS, 4)

            print({
                "train/loss": loss.item(),
                "lr": lr,
                "real_MFU": real_MFU,
                "step_per_sec": round(steps / max(dt, 1e-9), 2),
                "tokens_per_sec": int(tokens_per_step * steps / max(dt, 1e-9)),
                "tokens_seen": (it + 1) * tokens_per_step,
            }, it)


if __name__ == "__main__":
    train()
