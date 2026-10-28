import torch
import time
from collections import defaultdict

def is_available(device):
    if device == "mps":
        return torch.backends.mps.is_available()
    if device == "cuda":
        return torch.cuda.is_available()
    if device == "cpu":
        return True
    return False

def sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()

def bench(
    devices = ("cpu", "mps", "cuda"), 
    dtypes = (torch.float32, torch.float16, torch.bfloat16),
    sizes = (1024, 2048, 4096),
    iters = 20
):
    bench_res = defaultdict(float)
    for device in devices:
        if not is_available(device):
            print(f"{device} is not avalable, skip")
            continue

        for dtype in dtypes:
            for size in sizes:
                a = torch.rand((size, size), dtype = dtype, device = device)
                b = torch.rand((size, size), dtype = dtype, device = device)

                for _ in range(3):
                    a @ b
                sync(device)

                t0 = time.perf_counter()
                for _ in range(iters):
                    a @ b

                sync(device)
                seconds = time.perf_counter() - t0
                flops = 2 * size ** 3
                flops_s = flops * iters / seconds
                print(f"{device:4} {str(dtype):15} size:{size:5}, {flops/1e12:.3f}TFLOPS/iter, {seconds :.2f}s, {flops_s/1e12 :.2f} TFLOP/s")

                bench_res[(device, str(dtype))] = max(bench_res[(device, str(dtype))], flops_s)
    return bench_res        

def device_bench():
    bench_cpu = bench(devices = ("cpu",), dtypes = (torch.float32,))
    bench_gpu = bench(devices = ("mps", "cuda"), dtypes = (torch.float32, torch.float16, torch.bfloat16))

    bench_gpu.update(bench_cpu)
    return bench_gpu



if __name__ == "__main__":
    bench_res = device_bench()
    print(bench_res)