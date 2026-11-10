"""Train, validate, and measure GPU energy for the HAR models."""
import argparse
import csv
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import pynvml
import torch
import torch.nn as nn

import measure_updated as measure
from models import build_model, count_params


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = (
    ROOT
    / "human+activity+recognition+using+smartphones"
    / "UCI HAR Dataset"
    / "UCI HAR Dataset"
)
try:
    COMMIT = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
except (subprocess.CalledProcessError, OSError):
    COMMIT = "unknown"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_har(path=DEFAULT_DATA_PATH, n_val_subjects=5, split_seed=123, device="cuda:0"):
    """Load the UCI HAR directory or a compatible NPZ archive.

    Directory input uses the nine inertial-signal channels and the official
    train/test split. NPZ input must contain Xtr, ytr, subj_tr, Xte, and yte.
    """
    path = Path(path)
    if not path.is_file():
        if not path.is_dir():
            raise FileNotFoundError(
                f"HAR data not found: {path}. Provide --data pointing to the "
                "UCI HAR Dataset directory or a compatible NPZ file."
            )

    if path.is_dir():
        signal_names = (
            "body_acc_x",
            "body_acc_y",
            "body_acc_z",
            "body_gyro_x",
            "body_gyro_y",
            "body_gyro_z",
            "total_acc_x",
            "total_acc_y",
            "total_acc_z",
        )

        def read_split(split):
            split_dir = path / split
            labels_path = split_dir / f"y_{split}.txt"
            if not labels_path.is_file():
                raise FileNotFoundError(f"Required HAR label file not found: {labels_path}")
            labels = np.loadtxt(labels_path, dtype=np.int64).reshape(-1)
            subjects = None
            if split == "train":
                subjects_path = split_dir / "subject_train.txt"
                if not subjects_path.is_file():
                    raise FileNotFoundError(
                        f"Required HAR subject file not found: {subjects_path}"
                    )
                subjects = np.loadtxt(subjects_path, dtype=np.int64).reshape(-1)

            signal_dir = split_dir / "Inertial Signals"
            channels = []
            for name in signal_names:
                signal_path = signal_dir / f"{name}_{split}.txt"
                if not signal_path.is_file():
                    raise FileNotFoundError(
                        f"Required HAR signal file not found: {signal_path}"
                    )
                channel = np.loadtxt(signal_path, dtype=np.float32, ndmin=2)
                if channel.shape[0] != len(labels):
                    raise ValueError(
                        f"{signal_path} has {channel.shape[0]} samples; "
                        f"expected {len(labels)}."
                    )
                channels.append(channel)

            if any(channel.shape[1] != channels[0].shape[1] for channel in channels):
                raise ValueError(f"Signal files in {signal_dir} have different timesteps.")
            return np.stack(channels, axis=-1), labels, subjects

        X_train, y_train, subjects = read_split("train")
        X_test, y_test, _ = read_split("test")
        if subjects is None:
            raise ValueError("The UCI HAR training split must include subject IDs.")
    else:
        with np.load(path) as archive:
            required = {"Xtr", "ytr", "subj_tr", "Xte", "yte"}
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(
                    f"{path} is missing NPZ arrays: {', '.join(sorted(missing))}"
                )
            X_train = archive["Xtr"].astype(np.float32, copy=False)
            y_train = archive["ytr"].reshape(-1)
            subjects = archive["subj_tr"].reshape(-1)
            X_test = archive["Xte"].astype(np.float32, copy=False)
            y_test = archive["yte"].reshape(-1)

    if X_train.ndim != 3 or X_test.ndim != 3:
        raise ValueError("Xtr and Xte must have shape (examples, timesteps, features).")
    if X_train.shape[1:] != X_test.shape[1:]:
        raise ValueError("Xtr and Xte must have the same timestep and feature dimensions.")
    if len(X_train) != len(y_train) or len(X_train) != len(subjects):
        raise ValueError("Xtr, ytr, and subj_tr must contain the same number of examples.")
    if len(X_test) != len(y_test) or not len(X_train) or not len(X_test):
        raise ValueError("Training and test arrays must be non-empty and have matching labels.")
    if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
        raise ValueError("HAR features contain NaN or infinite values.")

    labels = np.concatenate((y_train, y_test))
    if np.isin(labels, np.arange(1, 7)).all():
        y_train = y_train - 1
        y_test = y_test - 1
    elif not np.isin(labels, np.arange(0, 6)).all():
        raise ValueError("HAR labels must use class IDs 0-5 or 1-6.")

    unique_subjects = np.unique(subjects)
    if not 1 <= n_val_subjects < len(unique_subjects):
        raise ValueError(
            f"n_val_subjects must be between 1 and {len(unique_subjects) - 1} "
            f"for this dataset; got {n_val_subjects}."
        )
    val_subjects = np.random.RandomState(split_seed).choice(
        unique_subjects, n_val_subjects, replace=False
    )
    is_val = np.isin(subjects, val_subjects)
    Xtr, ytr = X_train[~is_val], y_train[~is_val]
    Xva, yva = X_train[is_val], y_train[is_val]
    if not len(Xtr) or not len(Xva):
        raise ValueError("Subject split produced an empty training or validation set.")

    # Fit normalization on training subjects only to avoid validation leakage.
    mean = Xtr.mean(axis=(0, 1), keepdims=True)
    std = Xtr.std(axis=(0, 1), keepdims=True) + 1e-8

    def features(array):
        return torch.as_tensor((array - mean) / std, dtype=torch.float32, device=device)

    def targets(array):
        return torch.as_tensor(array, dtype=torch.long, device=device)

    return features(Xtr), targets(ytr), features(Xva), targets(yva), features(X_test), targets(y_test)


@torch.no_grad()
def accuracy(model, X, y, batch_size=1024):
    model.eval()
    correct = 0
    for start in range(0, len(X), batch_size):
        correct += (model(X[start:start + batch_size]).argmax(1) == y[start:start + batch_size]).sum().item()
    return correct / len(y)


def _append_rows(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    has_data = path.exists() and path.stat().st_size > 0
    if has_data:
        with path.open("r", newline="", encoding="utf-8") as file:
            existing_fields = next(csv.reader(file), [])
        if existing_fields != fieldnames:
            raise ValueError(
                f"CSV schema mismatch in {path}; use a new --tag or move the old results file."
            )
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not has_data:
            writer.writeheader()
        writer.writerows(rows)


def train_har(
    name,
    data,
    epochs=100,
    batch_size=64,
    lr=1e-3,
    seed=0,
    tag="final",
    output_dir="results",
    gpu=0,
    idle_w=None,
    final_test=True,
    **model_kw,
):
    """Train a model, validate each epoch, and measure gross/net training energy."""
    if epochs < 1 or batch_size < 1 or lr <= 0:
        raise ValueError("epochs and batch_size must be positive, and lr must be positive.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Select a GPU runtime (for example, Kaggle T4).")

    set_seed(seed)
    Xtr, ytr, Xva, yva, Xte, yte = data
    device = torch.device(f"cuda:{gpu}")
    model = build_model(name, **model_kw).to(device)
    n_params = count_params(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
    if idle_w is None:
        idle_w = measure.measure_idle(gpu=gpu)

    torch.cuda.reset_peak_memory_stats(device)
    cumulative_energy = 0.0
    cumulative_time = 0.0
    rows = []

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(Xtr), device=device)
        torch.cuda.synchronize(device)
        sampler = measure.PowerSampler(gpu)
        sampler.start()
        start = time.perf_counter()
        try:
            for offset in range(0, len(Xtr), batch_size):
                indices = permutation[offset:offset + batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(Xtr[indices]), ytr[indices])
                loss.backward()
                optimizer.step()
            torch.cuda.synchronize(device)
            end = time.perf_counter()
        except BaseException:
            sampler.abort()
            raise

        measured = sampler.stop(start, end)
        duration = end - start
        energy_gross = measured["e_integral"]
        energy_net = energy_gross - float(idle_w) * duration
        cumulative_energy += energy_gross
        cumulative_time += duration
        val_accuracy = accuracy(model, Xva, yva)
        try:
            temperature = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            )
        except pynvml.NVMLError:
            temperature = None

        rows.append(
            dict(
                model=name,
                dataset="har",
                impl="default",
                phase="train",
                seed=seed,
                lr=lr,
                params=n_params,
                epoch=epoch,
                epoch_time_s=duration,
                train_time_s=cumulative_time,
                train_energy_j_gross=cumulative_energy,
                epoch_energy_j_gross=energy_gross,
                epoch_energy_j_net=energy_net,
                epoch_j_per_train_sample_gross=energy_gross / len(Xtr),
                val_acc=val_accuracy,
                test_acc=None,
                temp_c=temperature,
                peak_mem_mb=torch.cuda.max_memory_allocated(device) / 2**20,
                commit=COMMIT,
            )
        )
        print(
            f"{name} epoch {epoch}/{epochs}: val_acc={val_accuracy:.4f}, "
            f"epoch_energy={energy_gross:.2f} J"
        )

    if final_test:
        rows[-1]["test_acc"] = accuracy(model, Xte, yte)
        print(f"{name} final test accuracy: {rows[-1]['test_acc']:.4f}")
    _append_rows(Path(output_dir) / f"har_train_{tag}.csv", rows)
    return rows, model


def infer_har(
    name,
    model,
    Xte,
    batch_sizes=(1, 16, 64, 256),
    seed=0,
    tag="final",
    output_dir="results",
    gpu=0,
    idle_w=None,
    **measurement_kw,
):
    """Measure inference energy and throughput for each requested batch size."""
    model = model.to(f"cuda:{gpu}").eval()
    n_params = count_params(model)
    output_csv = Path(output_dir) / f"har_infer_{tag}.csv"
    for requested_batch_size in batch_sizes:
        batch_size = min(requested_batch_size, len(Xte))
        xb = Xte[:batch_size].contiguous()

        def run_batch():
            with torch.inference_mode():
                model(xb)
            torch.cuda.synchronize(gpu)

        measure.run_measured(
            run_batch,
            n_items=batch_size,
            meta=dict(
                model=name,
                dataset="har",
                impl="default",
                phase="infer",
                seed=seed,
                params=n_params,
                batch_size=batch_size,
                seq_len=Xte.shape[1],
                commit=COMMIT,
            ),
            out_csv=str(output_csv),
            idle_w=idle_w,
            gpu=gpu,
            **measurement_kw,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train/validate HAR models and collect T4 GPU energy readings."
    )
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA_PATH),
        help="UCI HAR Dataset directory or NPZ dataset path.",
    )
    parser.add_argument("--models", default="lstm,transformer,cfc",
                        help="Comma-separated model names: lstm,transformer,cfc.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-subjects", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=123)
    parser.add_argument("--gpu", type=int, default=0, help="CUDA/NVML GPU index.")
    parser.add_argument("--tag", default="final", help="Results filename tag.")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--idle-seconds", type=float, default=15)
    parser.add_argument("--warmup-seconds", type=float, default=5)
    parser.add_argument("--measure-seconds", type=float, default=20)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--inference-batches", default="1,16,64,256")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. In Kaggle, enable Accelerator: GPU (T4).")
    if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        raise ValueError(f"GPU index {args.gpu} is unavailable.")
    names = [name.strip().lower() for name in args.models.split(",") if name.strip()]
    if not names or any(name not in {"lstm", "transformer", "cfc"} for name in names):
        raise ValueError("--models must contain one or more of lstm, transformer, cfc.")
    batch_sizes = tuple(int(value) for value in args.inference_batches.split(","))
    if any(size < 1 for size in batch_sizes):
        raise ValueError("--inference-batches must contain positive integers.")

    torch.cuda.set_device(args.gpu)
    device = f"cuda:{args.gpu}"
    data = load_har(args.data, args.val_subjects, args.split_seed, device)
    print(
        f"GPU: {torch.cuda.get_device_name(args.gpu)} | "
        f"train={len(data[1])} val={len(data[3])} test={len(data[5])}"
    )
    idle_w = measure.measure_idle(seconds=args.idle_seconds, gpu=args.gpu)
    print(f"Idle GPU power reference: {idle_w:.2f} W")

    for name in names:
        rows, model = train_har(
            name,
            data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            tag=args.tag,
            output_dir=args.output_dir,
            gpu=args.gpu,
            idle_w=idle_w,
        )
        if not args.skip_inference:
            infer_har(
                name,
                model,
                data[4],
                batch_sizes=batch_sizes,
                seed=args.seed,
                tag=args.tag,
                output_dir=args.output_dir,
                gpu=args.gpu,
                idle_w=idle_w,
                warmup_s=args.warmup_seconds,
                min_duration=args.measure_seconds,
                trials=args.trials,
            )
        del rows, model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
