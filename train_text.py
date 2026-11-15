#7
import argparse, csv, importlib.util, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Model builder
# ----------------------------------------------------------------------------
def build_model(arch, vocab, d_model, n_layer, n_head):
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(here, "models.py")):
        import sys, importlib
        sys.path.insert(0, here)
        ext = importlib.import_module("models")
        if hasattr(ext, "build_lm"):
            print("using build_lm from models.py with ITS default sizes "
                  "(--d_model / --n_layer / --n_head are ignored)")
            return ext.build_lm(arch)

    raise RuntimeError("models.py with build_lm() was not found.")


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_bin(path):
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(arr, B, T, rng, device):
    ix = rng.integers(0, len(arr) - T - 1, size=B)
    x = np.stack([arr[i:i + T] for i in ix]).astype(np.int64)
    y = np.stack([arr[i + 1:i + 1 + T] for i in ix]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


@torch.no_grad()
def evaluate(model, arr, T, B, device, use_amp, max_tokens):
    model.eval()
    n_win = min(len(arr) - 1, max_tokens) // T
    total_loss, total_tok = 0.0, 0

    for s in range(0, n_win, B):
        ids = range(s, min(s + B, n_win))
        x = np.stack([arr[i*T:(i+1)*T] for i in ids]).astype(np.int64)
        y = np.stack([arr[i*T+1:(i+1)*T+1] for i in ids]).astype(np.int64)
        x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(x)

        loss = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            y.reshape(-1), reduction="sum"
        )
        total_loss += loss.item()
        total_tok += y.numel()

    model.train()
    avg = total_loss / total_tok
    return avg, math.exp(avg)


# ----------------------------------------------------------------------------
# Checkpoint saving
# ----------------------------------------------------------------------------
def save_checkpoint(path, model, opt, scaler, step, best_ppl,
                    train_time, rng, np_rng):

    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": opt.state_dict(),
        "scaler_state": scaler.state_dict(),
        "step": step,
        "best_ppl": best_ppl,
        "train_time": train_time,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng": np_rng.bit_generator.state,
        "batch_rng": rng.bit_generator.state,
    }, path)


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    np_rng = np.random.default_rng(args.seed + 1)

    os.makedirs(args.out_dir, exist_ok=True)

    meta = json.load(open(os.path.join(args.data_dir, "meta.json")))
    vocab = meta["vocab_size"]

    train_arr = load_bin(os.path.join(args.data_dir, "train.bin"))
    val_arr = load_bin(os.path.join(args.data_dir, "val.bin"))
    test_arr = load_bin(os.path.join(args.data_dir, "test.bin"))

    model = build_model(
        args.arch, vocab, args.d_model, args.n_layer, args.n_head
    ).to(device)

    n_params = count_params(model)

    with torch.no_grad():
        probe = model(torch.zeros(1, 8, dtype=torch.long, device=device))

    assert probe.shape == (1, 8, vocab), (
        f"model output {tuple(probe.shape)} but tokenizer vocab is {vocab}"
    )

    print(
        f"arch={args.arch} params={n_params/1e6:.2f}M "
        f"device={device} precision={args.precision}"
    )

    use_amp = args.precision == "fp16" and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )

    tok_per_step = args.batch * args.seq_len
    total_steps = max(1, args.max_tokens // tok_per_step)
    eval_every = max(1, args.eval_every_tokens // tok_per_step)
    warmup = max(1, min(200, total_steps // 10))

    def lr_at(step):
        if step <= warmup:
            return args.lr * step / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    log_path = os.path.join(args.out_dir, "log.csv")
    ckpt_path = os.path.join(args.out_dir, "ckpt_last.pt")
    best_path = os.path.join(args.out_dir, "ckpt_best.pt")

    start_step = 0
    best_ppl = float("inf")
    train_time = 0.0

    # Resume only when --resume is supplied
    if args.resume and os.path.exists(ckpt_path):
        print("Resuming from checkpoint...")

        checkpoint = torch.load(
            ckpt_path, map_location=device, weights_only=False
        )

        model.load_state_dict(checkpoint["model_state"])
        opt.load_state_dict(checkpoint["optimizer_state"])

        if checkpoint.get("scaler_state"):
            scaler.load_state_dict(checkpoint["scaler_state"])

        start_step = checkpoint["step"]
        best_ppl = checkpoint["best_ppl"]
        train_time = checkpoint["train_time"]

        rng.bit_generator.state = checkpoint["batch_rng"]
        np_rng.bit_generator.state = checkpoint["numpy_rng"]

        torch.set_rng_state(checkpoint["torch_rng"].cpu())

        # Correct CUDA RNG restoration
        if device == "cuda" and checkpoint["cuda_rng"] is not None:
            cuda_states = checkpoint["cuda_rng"]

            if isinstance(cuda_states, torch.Tensor):
                cuda_states = [cuda_states]

            torch.cuda.set_rng_state_all([
                state.cpu().to(torch.uint8) for state in cuda_states
            ])

        print(f"Resumed at step {start_step}/{total_steps}")

    elif args.resume:
        print("No checkpoint found. Starting a fresh run.")

    # Preserve logs when resuming
    log_exists = os.path.exists(log_path) and os.path.getsize(log_path) > 0
    write_header = not (args.resume and log_exists)

    logf = open(
        log_path,
        "a" if args.resume else "w",
        newline=""
    )
    writer = csv.writer(logf)

    if write_header:
        writer.writerow([
            "step", "tokens_seen", "train_loss", "val_loss",
            "val_ppl", "train_time_s", "peak_mem_mb"
        ])

    model.train()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t_last = time.time()
    run_loss, run_n = 0.0, 0

    for step in range(start_step + 1, total_steps + 1):

        for g in opt.param_groups:
            g["lr"] = lr_at(step)

        x, y = get_batch(train_arr, args.batch, args.seq_len, rng, device)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp
        ):
            logits = model(x)

        loss = F.cross_entropy(
            logits.float().reshape(-1, vocab),
            y.reshape(-1)
        )

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        if step % 10 == 0:
            run_loss += loss.item()
            run_n += 1

        if step % eval_every == 0 or step == total_steps:

            if device == "cuda":
                torch.cuda.synchronize()

            train_time += time.time() - t_last

            v_loss, v_ppl = evaluate(
                model, val_arr, args.seq_len, 64,
                device, use_amp, args.eval_tokens
            )

            peak = (
                torch.cuda.max_memory_allocated() / 2**20
                if device == "cuda" else 0.0
            )

            tr = run_loss / max(1, run_n)

            writer.writerow([
                step,
                step * tok_per_step,
                f"{tr:.4f}",
                f"{v_loss:.4f}",
                f"{v_ppl:.3f}",
                f"{train_time:.1f}",
                f"{peak:.0f}"
            ])

            logf.flush()

            print(
                f"step {step}/{total_steps} "
                f"tokens {step*tok_per_step/1e6:.1f}M "
                f"train {tr:.3f} val_ppl {v_ppl:.2f} "
                f"time {train_time:.0f}s"
            )

            if v_ppl < best_ppl:
                best_ppl = v_ppl
                torch.save(model.state_dict(), best_path)

            save_checkpoint(
                ckpt_path, model, opt, scaler, step,
                best_ppl, train_time, rng, np_rng
            )

            run_loss, run_n = 0.0, 0
            t_last = time.time()

    logf.close()

    t_loss, t_ppl = evaluate(
        model, test_arr, args.seq_len, 64,
        device, use_amp, args.eval_tokens
    )

    result = dict(
        arch=args.arch,
        params=n_params,
        precision=args.precision,
        seq_len=args.seq_len,
        batch=args.batch,
        tokens_trained=total_steps * tok_per_step,
        train_time_s=train_time,
        best_val_ppl=best_ppl,
        test_ppl=t_ppl,
        seed=args.seed,
        d_model=args.d_model,
        n_layer=args.n_layer,
        n_head=args.n_head,
        lr=args.lr
    )

    json.dump(
        result,
        open(os.path.join(args.out_dir, "result.json"), "w"),
        indent=2
    )

    print("RESULT", result)
    return result


# ----------------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------------
def get_args(argv=None):
    ap = argparse.ArgumentParser()

    ap.add_argument("--arch", choices=["transformer", "lstm", "cfc"], required=True)
    ap.add_argument("--data_dir", default="/kaggle/working/data")
    ap.add_argument("--out_dir", default="/kaggle/working/runs/run")

    ap.add_argument("--d_model", type=int, default=320)
    ap.add_argument("--n_layer", type=int, default=6)
    ap.add_argument("--n_head", type=int, default=5)

    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)

    ap.add_argument("--max_tokens", type=int, default=50_000_000)
    ap.add_argument("--eval_every_tokens", type=int, default=2_000_000)
    ap.add_argument("--eval_tokens", type=int, default=1_000_000)

    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--resume", action="store_true")

    return ap.parse_args(argv)


if __name__ == "__main__":
    train(get_args())
