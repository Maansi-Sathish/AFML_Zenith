# Benchmarks: training and decode energy of LSTM, Transformer and CfC

Controlled, shape-only energy benchmarks of the three ~10M-parameter language models,
plus a kernel-level profile. These results feed the "naive vs optimized" (implementation
maturity) and "decode / KV-cache" parts of the study. **All numbers are single-seed and
preliminary** (see Caveats).

## Result Files

| File | What it is |
|---|---|
| `bench_train.csv` | One row per trial: energy and throughput of one training step (5 configs) |
| `bench_decode.csv` | One row per trial: energy and throughput of token-by-token decoding (9 configs) |
| `profile_<model>_<impl>.txt` | PyTorch-profiler summary of 3 training steps (top kernels, CPU vs CUDA time, kernel launches) |
| `energy_train.png`, `energy_decode.png` | Figures made by `plot_bench.py` from the two CSVs |

## Setup

- **Hardware:** one Kaggle Tesla T4 (70 W power cap). Decode runs: no clock throttling (`throttle_mask = 0`).
- **Models (`models.py`, `build_lm`):** vocab 8192, d_model 320, 6 layers, tied embeddings.
  Transformer 10,019,840 params; LSTM 10,028,992; CfC 9,946,560.
  `optimized` = fused PyTorch kernels; `naive` = same maths and params written by hand
  (LSTM: `LSTMCell` time loop; Transformer: manual attention). CfC has one implementation (`ncps`).
- **Workload:** random token ids and random weights. Energy depends on shapes and kernels, not values.
- **Training step:** batch 32, sequence 256, fp16 autocast, forward + loss + backward + AdamW.
- **Decode:** batch 1, 32 tokens generated per call, at context length 128 / 512 / 2048.
  The Transformer carries a KV cache (random values, correct shapes); LSTM and CfC carry a fixed-size state.
  Transformer decode uses the hand-written KV-cache layers, so it is recorded as `naive`.
- **Measurement (`measure_updated.py`):** NVML power integrated over time, minus idle power x duration
  (net); the raw integral is the gross energy. Also logged: NVML hardware energy counter and CodeCarbon.
  5 s warm-up, 3 trials of at least 15 s each. Idle power measured for 8 s before each config. Seed 1337, commit `d89f704`.
- **Profile (`profile_ops.py`):** 3 training steps, batch 16, sequence 256, **fp32** (no autocast).

## Results

### Training step (batch 32, sequence 256, fp16). Mean of 3 trials.

| Model | Impl | Net J/token | Tokens/s | vs optimized LSTM (energy) | Idle used (W) |
|---|---|---|---|---|---|
| LSTM | optimized | 5.4e-4 | 104,297 | 1.0x | 10.4 |
| LSTM | naive | 4.4e-3 | 11,611 | 8.1x | 11.9 |
| Transformer | optimized | 5.0e-4 | 101,822 | 0.9x | 15.2 |
| Transformer | naive | 6.2e-4 | 87,564 | 1.1x | 11.7 |
| CfC | single | 8.0e-3 | 3,387 | 14.7x | 12.0 |

### Decode (batch 1, 32 tokens per call). Mean ± std over 3 trials.

| Model | Context | Gross J/token | Net J/token | Tokens/s | Peak memory (MB) | Idle used (W) |
|---|---|---|---|---|---|---|
| LSTM | 128 | 0.0850 ± 0.0014 | 0.0541 | 448 | 128.8 | 13.8 |
| LSTM | 512 | 0.0856 ± 0.0010 | 0.0484 | 452 | 128.8 | 16.8 |
| LSTM | 2048 | 0.0868 ± 0.0002 | 0.0491 | 451 | 128.8 | 17.0 |
| CfC | 128 | 0.1048 ± 0.0015 | 0.0663 | 344 | 66.1 | 13.2 |
| CfC | 512 | 0.1028 ± 0.0011 | 0.0660 | 353 | 66.1 | 13.0 |
| CfC | 2048 | 0.1018 ± 0.0004 | 0.0435 | 359 | 66.1 | **21.0** |
| Transformer (naive, KV cache) | 128 | 0.1279 ± 0.0014 | 0.0817 | 279 | 75.9 | 12.9 |
| Transformer (naive, KV cache) | 512 | 0.1312 ± 0.0004 | 0.0829 | 276 | 84.3 | 13.3 |
| Transformer (naive, KV cache) | 2048 | 0.1334 ± 0.0016 | 0.0874 | 282 | 120.3 | 13.0 |

Use the **gross** column to compare decode runs (see Caveats on idle power).

### Profile of 3 training steps (batch 16, sequence 256, fp32)

| Model | Impl | CUDA kernel launches | Self CPU time | Self CUDA time |
|---|---|---|---|---|
| LSTM | optimized | 25,266 | 0.72 s | 1.16 s |
| LSTM | naive | 87,708 | 2.82 s | 1.46 s |
| Transformer | optimized | 1,503 | 0.37 s | 0.35 s |
| Transformer | naive | 1,395 | 0.36 s | 0.35 s |
| CfC | single | 290,346 | 8.62 s | 2.75 s |

## What the data show

1. **Implementation maturity dominates for recurrent models.** The naive LSTM does the same maths
   as the optimized one (outputs agree to about 1e-7) but uses 8.1x the energy per token and runs 9x slower.
   It makes 3.5x more kernel launches and spends 3.9x more CPU time.
2. **Most of CfC's training gap is launch overhead.** CfC uses 14.7x the energy of the optimized LSTM and
   16x that of the optimized Transformer, but only 1.8x the naive LSTM. It runs 290,346 kernel launches
   (193x the Transformer, 3.3x the naive LSTM), and its CPU time is 3.1x its GPU time, so the GPU waits on
   the Python loop. The remaining 1.8x versus the naive LSTM mixes the heavier CfC cell with `ncps`
   implementation details; it cannot be split further without a fused CfC kernel.
3. **The Transformer is insensitive to implementation.** Naive is only 1.24x the optimized energy, with about
   the same number of launches (1,395 vs 1,503), because both are dominated by large matrix multiplies.
4. **Decode energy is flat in context length up to 2048 at batch 1** (gross, 128 to 2048: LSTM +2.0%,
   CfC -2.8%, Transformer +4.2%). All three are launch-bound here, so no energy crossover is visible yet.
5. **The KV cache shows up in memory, not energy.** Transformer peak memory grows from 75.9 to 120.3 MB
   (+58%) as context grows; LSTM (128.8 MB) and CfC (66.1 MB) stay constant.

## Caveats

- **Single seed (1337) and 3 trials per config.** The ± values are trial-to-trial measurement noise, not
  training variance. No confidence intervals across seeds yet.
- **Idle power varied from 10.4 W to 21.0 W between runs** because it was measured for only 8 s each time.
  This biases net energy: CfC decode at context 2048 looks 34% cheaper in net energy (0.0435 vs 0.0663 J/token)
  only because its idle estimate was 21 W instead of 13 W. Gross energy is flat (0.1048 to 0.1018), so decode
  comparisons use gross. Training net values carry roughly +-10% from the same effect.
- **NVML is an estimate.** Expect at least about 5% error, and no wall meter was available to check it.
- **Profile is fp32 at batch 16; the training benchmark is fp16 at batch 32.** Use the profile for relative
  launch counts and CPU/GPU balance, not for absolute energy.
- **Random weights and a synthetic KV cache.** Fine for energy (shape-dependent), not for quality.
- **Transformer decode uses hand-written attention** (the only cached path implemented). A fused cached
  attention would likely be cheaper, so the Transformer decode numbers are an upper bound.
- **Scope:** decode at batch 1 only, contexts up to 2048, one GPU type, one model size (~10M).
- **Not a quality comparison.** Perplexity and accuracy come from the full training runs in `results/text/`.

## Reproduce

```
git clone https://github.com/Maansi-Sathish/AFML_Orange_Zenith && cd AFML_Orange_Zenith
pip install -q ncps codecarbon nvidia-ml-py
python bench_lean.py --profile        # about 25 min on a T4; resumable (skips finished configs)
python plot_bench.py results/bench_train.csv results/bench_decode.csv
```

For steadier net energy, set `IDLE=30` (idle seconds) and rerun on a fresh `results/` folder.
