"""bench_lean.py -- short, resumable energy benchmark for Person 3 (no dataset needed).

Runs bench_lm.py one config at a time and SKIPS configs already in the CSVs, so after a
crash just run it again. Also (optionally) saves profiler summaries.

  python bench_lean.py                 # train (naive vs optimized) + decode, ~20-25 min on a T4
  python bench_lean.py --profile       # also run profile_ops.py for each variant (~1 min each)
  python bench_lean.py --only train    # or: --only decode / --only profile
Env: TRIALS (default 3), MINDUR seconds per trial (default 15).
"""
import argparse
import csv
import os
import subprocess
import sys

RES = os.environ.get("RES", "results")
TRIALS = int(os.environ.get("TRIALS", 3))
MINDUR = float(os.environ.get("MINDUR", 15))

TRAIN = [  # (arch, impl) at batch 32, seq 256 -> naive vs optimized comparison
    ("lstm", "optimized"), ("lstm", "naive"),
    ("transformer", "optimized"), ("transformer", "naive"),
    ("cfc", "optimized"),
]
DECODE_LENS = [128, 512, 2048]
DECODE_ARCH = [("lstm", "optimized"), ("cfc", "optimized"), ("transformer", "naive")]  # transformer KV decode = naive layers


def done(csv_path, arch, impl, phase, B, T):
    """True if the CSV already holds >= TRIALS rows for this config."""
    if not os.path.exists(csv_path):
        return False
    n = 0
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if (r["model"], r["impl"], r["phase"], int(r["batch_size"]), int(r["seq_len"])) == (arch, impl, phase, B, T):
                n += 1
    return n >= TRIALS


def run(cmd, timeout=1500):
    print("$", " ".join(cmd), flush=True)
    try:
        r = subprocess.run(cmd, timeout=timeout)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        print("  !! timed out, skipping", flush=True)
        return False


def bench(arch, impl, phase, B, T, csv_path):
    if done(csv_path, arch, impl, phase, B, T):
        print(f"skip (done): {arch}/{impl}/{phase} B={B} T={T}", flush=True)
        return
    ok = run([sys.executable, "bench_lm.py", "--arch", arch, "--impl", impl, "--phases", phase,
              "--seq_lens", str(T), "--batch_sizes", str(B), "--trials", str(TRIALS),
              "--min_duration", str(MINDUR), "--idle_seconds", "8", "--out_csv", csv_path])
    if not ok:
        print(f"  !! failed: {arch}/{impl}/{phase} B={B} T={T} (rerun to retry)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["train", "decode", "profile"])
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    tr, dc = f"{RES}/bench_train.csv", f"{RES}/bench_decode.csv"

    if args.only in (None, "train"):
        for arch, impl in TRAIN:
            bench(arch, impl, "train", 32, 256, tr)
    if args.only in (None, "decode"):
        for arch, impl in DECODE_ARCH:
            for T in DECODE_LENS:
                bench(arch, impl, "decode", 1, T, dc)
    if args.only == "profile" or args.profile:
        for arch, impl in TRAIN:
            out = f"{RES}/profile_{arch}_{impl}.txt"
            if os.path.exists(out):
                print("skip (done):", out); continue
            with open(out, "w") as f:
                print("$ profile", arch, impl, flush=True)
                subprocess.run([sys.executable, "profile_ops.py", arch, impl, "256", "16"],
                               stdout=f, stderr=subprocess.STDOUT)
    print("\nDONE. Files in", RES, ":", sorted(os.listdir(RES)))


if __name__ == "__main__":
    main()