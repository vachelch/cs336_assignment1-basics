import os, time, math
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
from dataclasses import dataclass, asdict
import torch.nn.functional as F
import argparse

def parse_train_args():
    p = argparse.ArgumentParser(description="Train a Transformer LM")
    # data
    p.add_argument("--train_path", type=str, default = "data/tokenized/TinyStoriesV2_train_ids.bin")
    p.add_argument("--eval_path", type=str, default = "data/tokenized/TinyStoriesV2_valid_ids.bin")
    p.add_argument("--batch_size", type=int, default = 32)

    # model
    # optimizer

    # args
    p.add_argument("--resume", action="store_true")
    p.add_argument("--ckpt_path", type=str, default = "data/checkpoints/ckpt_step_40.pt")
    p.add_argument("--ckpt_dir", type=str, default = "data/checkpoints/")
    p.add_argument("--log_interval", type=int, default = 100)
    p.add_argument("--save_ckpt_interval", type=int, default = 1000)
    p.add_argument("--eval_iters", type=int, default = 40)
    p.add_argument("--eval_interval", type=int, default = 500)
    p.add_argument("--enable_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default = "cs336")
    p.add_argument("--wandb_run_name", type=str, default = "tiny_stories_test1")

    args, _ = p.parse_known_args()
    return args

args = parse_train_args()

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

@torch.no_grad()
def evaluate(model, eval_data, cfg):
    total_loss = []
    for _ in range(cfg.args.eval_iters):
        inputs, targets = get_batch(eval_data, cfg.data.batch_size, cfg.model.context_length, cfg.model.device)
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        total_loss.append(loss)
    return (sum(total_loss) / len(total_loss)).item()


def split_decay_parameters(name_params):
    no_decay = []
    decay = []
    for name, p in name_params:
        if not p.requires_grad:
            continue
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

@dataclass
class ModelConfig:
    vocab_size: int = 10000
    context_length: int = 256
    num_layers: int = 4
    d_model: int = 512
    num_heads: int = 4
    d_ff: int = 1344
    rope_theta: float = 10000.0
    device: str = default_device()
    dtype: torch.dtype = torch.bfloat16
    eos_id: int | None = 0  # <|endoftext|> is id 0 in the TinyStories vocab

@dataclass
class OptimConfig:
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.01
    lr_max: float = 3e-4
    lr_min: float = 3e-5
    max_l2_norm: float = 1.0

@dataclass
class DataConfig:
    train_path: str = args.train_path
    eval_path: str = args.eval_path
    data_type: str = "uint16"
    batch_size: int = args.batch_size
    # 32 * 256 * 40000 ~= 327M tokens
    max_iter: int = 40000
    warmup_iter: int = 40000 // 10

@dataclass
class ArgsConfig:
    resume: bool = args.resume
    ckpt_path: str = args.ckpt_path
    ckpt_dir: str = args.ckpt_dir
    log_interval: int = args.log_interval
    save_ckpt_interval: int = args.save_ckpt_interval
    eval_iters: int = args.eval_iters
    eval_interval: int = args.eval_interval
    enable_wandb: bool = args.enable_wandb
    wandb_project: str = args.wandb_project
    wandb_run_name: str = args.wandb_run_name

class Config:
    def __init__(self):
        self.model: ModelConfig = ModelConfig()
        self.optim: OptimConfig = OptimConfig()
        self.data: DataConfig = DataConfig()
        self.args: ArgsConfig = ArgsConfig()

def train():
    cfg = Config()
    os.makedirs(cfg.args.ckpt_dir, exist_ok = True)
    if cfg.args.enable_wandb:
        import wandb
        config = {k: asdict(v) for k, v in vars(cfg).items()}
        config["model"]["dtype"] = str(cfg.model.dtype)  # torch.dtype is not JSON-serializable
        wandb.init(
            project = cfg.args.wandb_project,
            name = cfg.args.wandb_run_name,
            config = config,
        )

    # device benchmark
    bench_dict = device_bench()

    # current setting device peak flops/s
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
    compiled_model = torch.compile(model)

    param_groups = split_decay_parameters(model.named_parameters())
    if True:
        optimizer = torch.optim.AdamW(
            param_groups,
            lr = cfg.optim.lr_min,
            betas = cfg.optim.betas,
            weight_decay = cfg.optim.weight_decay,
        )
    else:
        optimizer = AdamW(
            param_groups,
            cfg.optim.lr_min,
            cfg.optim.betas,
            cfg.optim.weight_decay,
        )

    start_iter = 0
    if cfg.args.resume:
        start_iter = load_checkpoint(cfg.args.ckpt_path, model, optimizer)
        print(f"Resume from {cfg.args.ckpt_path} at iteration {start_iter}")

    t0 = time.time()
    tokens_per_step = cfg.data.batch_size * cfg.model.context_length
    for it in range(start_iter, cfg.data.max_iter):
        inputs, targets = get_batch(train_data, cfg.data.batch_size, cfg.model.context_length, cfg.model.device)
        lr = get_lr_cosine_schedule(
            it, cfg.optim.lr_max, cfg.optim.lr_min, cfg.data.warmup_iter, cfg.data.max_iter
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        logits = compiled_model(inputs)
        if True:
            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                targets.view(-1)
            )
        else:
            loss = cross_entropy(logits, targets)
        
        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), cfg.optim.max_l2_norm)
        optimizer.step()

        if it % cfg.args.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            steps = cfg.args.log_interval if it != start_iter else 1

            real_FLOPS = FLOPs_step * steps / dt
            real_MFU = round(real_FLOPS / peak_FLOPS, 4)

            print({
                "step": it,
                "train/loss": round(loss.item(), 4),
                "lr": float(f"{lr:.3g}"),
                "real_MFU": real_MFU,
                "step_per_sec": round(steps / max(dt, 1e-9), 2),
                "tokens_per_sec": int(tokens_per_step * steps / max(dt, 1e-9)),
                "tokens_seen": (it + 1) * tokens_per_step,
            })

            if cfg.args.enable_wandb:
                wandb.log({"train/loss": loss.item(), "lr": lr}, step=it)

        if it > 0 and it % cfg.args.eval_interval == 0:
            val_loss = evaluate(model, eval_data, cfg)
            print(f"  >> eval_loss {val_loss:.4f} | eval_ppl {math.exp(val_loss):.2f}")
            if cfg.args.enable_wandb:
                wandb.log({"eval/loss": val_loss, "eval/ppl": math.exp(val_loss)}, step=it)

        if (it+1) % cfg.args.save_ckpt_interval == 0:
            ckpt_path = os.path.join(cfg.args.ckpt_dir, f"ckpt_step_{it+1}.pt")
            save_checkpoint(model, optimizer, it+1, ckpt_path)
            print(f"Saved ckpt to {ckpt_path}")

    ckpt_path = os.path.join(cfg.args.ckpt_dir, f"ckpt_step_{cfg.data.max_iter}.pt")
    save_checkpoint(model, optimizer, cfg.data.max_iter, ckpt_path)
    print(f"Saved ckpt to {ckpt_path}")



if __name__ == "__main__":
    train()
