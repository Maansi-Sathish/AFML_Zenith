"""bench_lm.py -- energy per training step / prefill / decode for the causal LMs.

Uses measure_updated.run_measured (idle-subtracted NVML energy). Random token ids and
random weights are fine: energy depends on shapes and kernels, not values.

Phases
  train   : one step = forward + CE loss + backward + AdamW   (n_items = B*T tokens)
  prefill : one forward pass over (B,T), no grad              (n_items = B*T tokens)
  decode  : n_new tokens generated one at a time via model.step() at context length T
            (n_items = B*n_new tokens). Transformer carries a KV cache that grows with T;
            LSTM/CfC carry a fixed-size state.

Examples (GPU):
  python bench_lm.py --arch lstm --impl optimized --phases train --seq_lens 256 --batch_sizes 32
  python bench_lm.py --arch lstm --impl naive     --phases train --seq_lens 256 --batch_sizes 32
  python bench_lm.py --arch transformer --phases decode --seq_lens 128,512,2048 --batch_sizes 1,4
Smoke test on a laptop (no GPU/NVML needed):
  python bench_lm.py --arch cfc --phases train,prefill,decode --seq_lens 32 --batch_sizes 2 --n_new 4 --dry_run

Caveat for the paper: transformer decode needs the hand-written KV-cache layers
(impl="naive" in models.py), so it is recorded with impl="naive" even if --impl optimized.
"""
import argparse
import os
import subprocess
import time

import torch
import torch.nn.functional as F

from models import build_lm, count_params, set_seed

V = 8192
MAX_LEN = 4096  # positional-encoding table size in models.py


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def autocast(device, use_amp):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp)


def make_train_fn(model, B, T, device, use_amp):
    x = torch.randint(0, V, (B, T + 1), device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    model.train()

    def fn():
        with autocast(device, use_amp):
            logits = model(x[:, :-1])
        loss = F.cross_entropy(logits.float().reshape(-1, V), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    return fn


def make_prefill_fn(model, B, T, device, use_amp):
    x = torch.randint(0, V, (B, T), device=device)
    model.eval()

    def fn():
        with torch.no_grad(), autocast(device, use_amp):
            model(x)
    return fn


def make_state(model, B, L, device, dtype):
    """Fake a context of length L (values don't matter for energy, shapes do)."""
    if model.arch != "transformer":
        return None  # LSTM/CfC: fresh fixed-size state
    caches = []
    for layer in model.body.layers:
        sa = layer.self_attn
        k = torch.randn(B, sa.h, L, sa.dh, device=device, dtype=dtype)
        caches.append((k, torch.randn_like(k)))
    return (L, caches)


def make_decode_fn(model, B, L, n_new, device, use_amp):
    model.eval()
    state0 = make_state(model, B, L, device, torch.float16 if use_amp else torch.float32)
    tok0 = torch.randint(0, V, (B, 1), device=device)

    def fn():
        st, tok = state0, tok0
        with torch.no_grad(), autocast(device, use_amp):
            for _ in range(n_new):
                logits, st = model.step(tok, st)
                tok = logits[:, -1].argmax(-1, keepdim=True)
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["transformer", "lstm", "cfc"], required=True)
    ap.add_argument("--impl", choices=["optimized", "naive"], default="optimized")
    ap.add_argument("--phases", default="train")
    ap.add_argument("--seq_lens", default="256")
    ap.add_argument("--batch_sizes", default="32")
    ap.add_argument("--n_new", type=int, default=32, help="tokens generated per decode call")
    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out_csv", default="results/bench_lm.csv")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--idle_seconds", type=float, default=15)
    ap.add_argument("--warmup_s", type=float, default=5)
    ap.add_argument("--min_duration", type=float, default=20)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--compile", action="store_true", help="torch.compile the model (best-effort)")
    ap.add_argument("--dry_run", action="store_true", help="CPU smoke test, no measurement")
    args = ap.parse_args()

    dry = args.dry_run
    device = torch.device("cpu" if dry else f"cuda:{args.gpu}")
    if not dry:
        assert torch.cuda.is_available(), "needs a GPU (use --dry_run for a CPU smoke test)"
        from measure_updated import run_measured, measure_idle
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    use_amp = (args.precision == "fp16") and not dry
    set_seed(args.seed)
    commit = git_commit()
    idle_w = None if dry else measure_idle(seconds=args.idle_seconds, gpu=args.gpu)
    if idle_w is not None:
        print(f"idle power: {idle_w:.1f} W")

    models = {}

    def get_model(impl):
        if impl not in models:
            m = build_lm(args.arch, impl=impl).to(device)
            if args.compile:
                try:
                    m = torch.compile(m)  # CfC loops over time in Python: compile can be very slow
                except Exception as e:
                    print(f"torch.compile failed ({e}); using eager")
            models[impl] = m
        return models[impl]

    for phase in args.phases.split(","):
        impl = "naive" if (phase == "decode" and args.arch == "transformer") else args.impl
        for B in [int(b) for b in args.batch_sizes.split(",")]:
            for T in [int(t) for t in args.seq_lens.split(",")]:
                if phase == "decode" and args.arch == "transformer" and T + args.n_new > MAX_LEN:
                    print(f"skip decode T={T}: needs T + n_new <= {MAX_LEN}")
                    continue
                if phase != "decode" and T > MAX_LEN:
                    print(f"skip T={T}: above {MAX_LEN}")
                    continue
                model = get_model(impl)
                n_params = count_params(model)
                try:
                    if phase == "train":
                        fn, n_items = make_train_fn(model, B, T, device, use_amp), B * T
                    elif phase == "prefill":
                        fn, n_items = make_prefill_fn(model, B, T, device, use_amp), B * T
                    elif phase == "decode":
                        fn, n_items = make_decode_fn(model, B, T, args.n_new, device, use_amp), B * args.n_new
                    else:
                        raise ValueError(f"unknown phase '{phase}'")
                    tag = f"{args.arch}/{impl}/{phase} B={B} T={T}"
                    if dry:
                        t0 = time.perf_counter()
                        fn(); fn()
                        print(f"[dry] {tag}: ok, {(time.perf_counter() - t0) / 2:.3f}s per call, params={n_params:,}")
                        continue
                    print(f"measuring {tag} ...")
                    meta = dict(model=args.arch, dataset="synthetic-ids", impl=impl, phase=phase,
                                seed=args.seed, params=n_params, batch_size=B, seq_len=T, commit=commit)
                    rows = run_measured(fn, n_items, meta, out_csv=args.out_csv, idle_w=idle_w,
                                        warmup_s=args.warmup_s, min_duration=args.min_duration,
                                        trials=args.trials, gpu=args.gpu)
                    mean_j = sum(r["j_per_item_net"] for r in rows) / len(rows)
                    mean_t = sum(r["items_per_s"] for r in rows) / len(rows)
                    print(f"  -> {mean_j:.3e} J/token (net), {mean_t:,.0f} tokens/s")
                except torch.cuda.OutOfMemoryError:
                    print(f"OOM for {args.arch}/{impl}/{phase} B={B} T={T}; skipped")
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()