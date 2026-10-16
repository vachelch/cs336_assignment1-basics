# run 
# uv run python tools/monitor_usage.py .venv/bin/python -m cs336_basics.train_bpe

import sys, time, subprocess, resource, psutil

cmd = sys.argv[1:]
start = time.perf_counter()
proc = subprocess.Popen(cmd)
root = psutil.Process(proc.pid)

peak_rss = peak_uss = 0
peak_cpu = 0.0
cpu_samples = []
cache = {}        # pid -> Process, so cpu_percent is measured against the previous sample
uss_ok = True     # USS may be unavailable on macOS without elevated privileges
errors = {}       # exception type -> count, so failures are not silent

def note(e):
    errors[type(e).__name__] = errors.get(type(e).__name__, 0) + 1

while proc.poll() is None:
    rss = uss = 0
    cpu = 0.0
    try:
        procs = [root] + root.children(recursive=True)
    except psutil.NoSuchProcess:
        break
    for p in procs:
        p = cache.setdefault(p.pid, p)
        # CPU: first call per process only initializes and returns 0
        try:
            cpu += p.cpu_percent(None)
        except psutil.Error as e:
            note(e)
        # RSS: needs no special privileges
        try:
            rss += p.memory_info().rss
        except psutil.Error as e:
            note(e)
        # USS: may raise AccessDenied on macOS; stop trying after the first failure
        if uss_ok:
            try:
                uss += p.memory_full_info().uss
            except psutil.AccessDenied:
                uss_ok = False
            except psutil.Error as e:
                note(e)
    peak_rss = max(peak_rss, rss)
    if uss_ok:
        peak_uss = max(peak_uss, uss)
    peak_cpu = max(peak_cpu, cpu)
    cpu_samples.append(cpu)
    time.sleep(0.2)

proc.wait()
wall = time.perf_counter() - start

# Exact total CPU time of all reaped descendants (including workers)
ru = resource.getrusage(resource.RUSAGE_CHILDREN)
cpu_time = ru.ru_utime + ru.ru_stime
ncores = psutil.cpu_count()

print(f"Wall time:          {wall:.2f} s")
print(f"Total CPU time:     {cpu_time:.2f} s (user {ru.ru_utime:.2f} + sys {ru.ru_stime:.2f})")
print(f"Average CPU:        {cpu_time / wall * 100:.0f}%  (~{cpu_time / wall:.1f} of {ncores} cores)")
print(f"Peak CPU:           {peak_cpu:.0f}%  (sampled)")
print(f"Samples taken:      {len(cpu_samples)}")
if cpu_samples:
    print(f"Sampled avg CPU:    {sum(cpu_samples) / len(cpu_samples):.0f}%  (for reference only)")
print(f"Peak total RSS:     {peak_rss / 1024**3:.2f} GB")
if uss_ok:
    print(f"Peak total USS:     {peak_uss / 1024**3:.2f} GB")
else:
    print("Peak total USS:     unavailable (AccessDenied, try running with sudo)")
if errors:
    print(f"Sampling errors:    {errors}")