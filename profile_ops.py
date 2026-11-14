"""Kernel-level profile of one LM training step (where does time/launches go?).
Usage: python profile_ops.py <transformer|lstm|cfc> <optimized|naive> [seq_len=256] [batch=16]
Run on a GPU notebook. Use the numbers to split 'architecture' vs 'kernel maturity'.
"""
import sys
import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
from models import build_lm

arch, impl = sys.argv[1], sys.argv[2]
L = int(sys.argv[3]) if len(sys.argv) > 3 else 256
B = int(sys.argv[4]) if len(sys.argv) > 4 else 16
V = 8192

m = build_lm(arch, impl=impl).cuda()
ids = torch.randint(0, V, (B, L + 1), device="cuda")
opt = torch.optim.AdamW(m.parameters(), 1e-3)


def step():
    loss = F.cross_entropy(m(ids[:, :-1]).reshape(-1, V), ids[:, 1:].reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()


for _ in range(3):  # warm-up
    step()
torch.cuda.synchronize()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
    for _ in range(3):
        step()
    torch.cuda.synchronize()

ev = p.key_averages()
try:
    print(ev.table(sort_by="self_cuda_time_total", row_limit=10))
except Exception:
    print(ev.table(sort_by="self_device_time_total", row_limit=10))
n_kernels = sum(e.count for e in ev if e.device_type == torch.autograd.DeviceType.CUDA)
print(f"{arch}/{impl} L={L} B={B}: CUDA kernel launches over 3 steps = {n_kernels}")