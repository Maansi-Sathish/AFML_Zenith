"""Plot bench_lm.py CSVs. Usage: python plot_bench.py results/bench_train.csv results/bench_decode.csv
Writes PNGs next to the first CSV: energy_train.png, energy_decode.png (mean +- std over trials).
"""
import sys
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.concat([pd.read_csv(p) for p in sys.argv[1:]], ignore_index=True)
out = os.path.dirname(sys.argv[1]) or "."
df["label"] = df["model"] + " (" + df["impl"] + ")"

tr = df[df.phase == "train"]
if len(tr):
    g = tr.groupby("label")["j_per_item_net"].agg(["mean", "std"]).sort_values("mean")
    ax = g["mean"].plot.barh(xerr=g["std"].fillna(0), figsize=(7, 4))
    ax.set_xlabel("net energy per training token (J)")
    ax.set_title("Training energy per token")
    plt.tight_layout(); plt.savefig(f"{out}/energy_train.png", dpi=160); plt.close()
    print(g)

dc = df[df.phase == "decode"]
if len(dc):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for (label, B), s in dc.groupby(["label", "batch_size"]):
        g = s.groupby("seq_len")["j_per_item_net"].agg(["mean", "std"])
        axes[0].errorbar(g.index, g["mean"], yerr=g["std"].fillna(0), marker="o", label=f"{label} B={B}")
        t = s.groupby("seq_len")["items_per_s"].mean()
        axes[1].plot(t.index, t.values, marker="o", label=f"{label} B={B}")
    axes[0].set_xlabel("context length"); axes[0].set_ylabel("J per generated token"); axes[0].set_xscale("log", base=2)
    axes[1].set_xlabel("context length"); axes[1].set_ylabel("tokens / s"); axes[1].set_xscale("log", base=2)
    axes[0].legend(fontsize=7); axes[0].set_title("Decode energy vs context")
    plt.tight_layout(); plt.savefig(f"{out}/energy_decode.png", dpi=160); plt.close()