# Project Report: Sequence Models for Text and Human-Activity Recognition

## Executive summary

This project compares three sequence-model families—LSTM, Transformer, and CfC—on two very different workloads:

1. **Next-token language modeling** using a filtered sample of LMSYS-Chat-1M.
2. **Six-class human-activity recognition (HAR)** using smartphone inertial-sensor sequences from the UCI HAR Dataset.

The study records predictive quality, training time and GPU energy, inference throughput and GPU energy per item, and peak GPU memory. The central result is that there is no single best model independent of task and operating point. On the recorded text run, CfC obtains the lowest perplexity but takes far longer and uses much more training energy; LSTM is the fastest and least energy-intensive to train. On HAR, Transformer and LSTM reach roughly 89–91% test accuracy, while CfC is substantially less accurate and slower in the saved runs. Inference batching generally improves throughput and lowers energy per item, but the effect depends on architecture and environment.

These are measurements of the saved runs, not a broad claim about all GPUs or model deployments. Training measurements are single-seed traces, and the laptop/Colab comparison notebooks explicitly treat inference repeats as descriptive rather than as evidence of statistical significance.

## 1. Aim and questions

The project asks how recurrent, attention-based, and continuous-time-inspired sequence models compare when trained and evaluated under reproducible protocols. It explores:

- How model quality changes during training.
- The time, energy, and GPU memory required to train each model.
- How inference throughput and energy per token/sample change with batch size.
- How recorded laptop and Colab results differ.
- Whether a model's quality advantage justifies its additional compute cost.

The two tasks are intentionally different. Text modeling predicts a vocabulary token at each sequence position; HAR classifies a complete sensor window into one activity. Their accuracy/quality measures and per-item units therefore should not be compared directly.

## 2. Data and preparation

### 2.1 LMSYS conversational text

The source dataset contains one million real-world conversations. The local preparation run used the first **20,000 rows** as a practical sample, not the full million. In [prep_data.py](../afml_text_10_nov/prep_data.py), conversations are filtered to English, non-flagged and non-redacted entries by default; empty turns and conversations outside the configured character-length range are rejected. Duplicate detection uses the normalized first message's content. A hash of that content assigns the conversation consistently to train, validation, or test, avoiding duplicate first-message content crossing splits.

The observed preparation statistics are:

| Item | Recorded value |
|---|---:|
| Rows examined | 20,000 |
| Kept conversations after filtering/deduplication | 8,775 |
| Tokenizer | Shared byte-level BPE |
| Vocabulary | 8,192 |
| Training tokens written | 3,410,365 |
| Validation tokens written | 31,815 |
| Test tokens written | 35,108 |

The tokenizer is fitted on training conversations only. Conversation role markers (`<|user|>`, `<|assistant|>`) and an end-of-conversation marker (`<|eos|>`) are included. Token IDs are stored as `uint16`. The requested 60-million training-token budget was not reached on this 20,000-row sample; the prepared training file contains about 3.41 million tokens.

### 2.2 Smartphone activity data

The HAR experiment uses the UCI dataset's inertial signal files. Each example is a **128-timestep sequence with 9 channels**: three body-acceleration axes, three gyroscope axes, and three total-acceleration axes. The six target classes are walking, walking upstairs, walking downstairs, sitting, standing, and laying.

The official split contains 7,352 training windows from 21 subjects and 2,947 held-out test windows. The code further selects five of the 21 training subjects for validation using split seed 123; in this run those subjects are 6, 7, 8, 14, and 28. This gives 5,733 training windows and 1,619 validation windows. The official test subjects remain separate. Per-channel normalization statistics are fitted on the training-subject subset only and then applied to validation and test data.

## 3. Models and training

### 3.1 Text language models

The text models share the same training data, sequence length, batch size, seed, and optimizer settings in the saved laptop run:

- Sequence length: **256 tokens**
- Batch size: **32**
- Optimization: AdamW, learning rate **0.001**, gradient clipping at 1.0
- Seed: **1337**
- Precision recorded in each result file: **FP32**
- Training: **244 optimizer steps**, or **1,998,848 token positions** (about 2.0 million)
- Validation: non-overlapping 256-token windows, using next-token cross-entropy; perplexity is `exp(mean loss)`

The architectures are implemented in [train_text.py](../afml_text_10_nov/train_text.py). Because the text project folder has no language-model `models.py` builder, this run uses the script's local implementations: a causal Transformer with sinusoidal positions and tied embeddings, a multilayer LSTM with tied embeddings, and an `ncps` CfC stack with residual updates and tied embeddings. Recorded parameter counts differ: LSTM **7.55M**, Transformer **10.02M**, CfC **4.11M**. Consequently, this is a comparison at shared data/training settings, not a parameter-matched benchmark.

### 3.2 HAR classifiers

The HAR models, defined in [models.py](../afml_project_9_nov/models.py), map each `(128, 9)` window to six class logits:

- **LSTM:** one recurrent layer, hidden width 144, dropout 0.1.
- **Transformer:** input projection to width 64, four attention heads, two encoder layers, feed-forward width 192, dropout 0.1, sinusoidal positional encoding, and mean pooling.
- **CfC:** 136 recurrent units, implemented with `ncps`, followed by dropout and a linear classifier.

The saved training protocol uses AdamW with learning rate **0.001**, batch size **64**, **100 epochs**, and seed **0**. Validation accuracy is recorded every epoch; the final held-out test accuracy is measured after training. The model sizes are similar but not identical: LSTM **90,150**, Transformer **84,486**, and CfC **89,686** parameters.

## 4. Inference and measurement methods

### 4.1 What “inference” means in these experiments

For **text**, the benchmark loads fixed token windows from `test.bin` and times a model forward pass that produces next-token logits. It reports token throughput and energy per input token. It does **not** repeatedly sample a next token, decode text, or measure end-to-end autoregressive generation with a KV cache or recurrent-state stepping. The separate sequence-length benchmark is also a forward-pass test on random token IDs, not generated text.

For **HAR**, inference is the forward pass of the classifier over held-out sensor windows, reporting classified samples per second and energy per sample.

Text inference requested batch sizes 1, 16, 64, and 256. The test file contains only 35,108 tokens, so at sequence length 256 the maximum number of full windows is 137. The code caps the last requested batch to the available data: the largest saved text batch is therefore **137**, not 256. HAR inference uses batches 1, 16, 64, and 256.

### 4.2 GPU energy and performance measurements

The shared measurement harness is [measure_updated.py](../afml_text_10_nov/measure_updated.py); the HAR project has a corresponding copy. It polls GPU power through NVML at approximately 50 ms intervals and integrates the power samples over the measured time window. The standard inference protocol is a 15-second idle reference, 5-second warm-up, at least 20 seconds per trial, and five repeated trials. Each inference record includes duration, iteration count, average/peak GPU power, integrated gross energy, idle-subtracted net energy, throughput, joules per item, peak GPU memory, temperature, clock, and optional NVML energy-counter and CodeCarbon values.

**Gross energy** is the integrated GPU power during the run. **Net energy** subtracts estimated idle power multiplied by run duration; it is baseline-dependent and is not the primary physical-energy comparison. NVML measurements reflect GPU-board energy, not complete system energy. Text training records four measurement intervals per model, each covering roughly 500,000 token positions; HAR training records one power-measured trace per epoch. Validation and inference are separate from the HAR training epoch power window.

## 5. Results

### 5.1 Text quality and training cost

Values below are from the saved local text model result files and training energy CSV. Training cost sums the four measured training intervals. Perplexity is lower when next-token predictions are better.

| Model | Parameters | Best validation perplexity | Test perplexity | Training time | Gross training energy | Net training energy |
|---|---:|---:|---:|---:|---:|---:|
| LSTM | 7.55M | 498.8 | 423.4 | 36.1 s | 1.94 kJ | 1.77 kJ |
| Transformer | 10.02M | 324.7 | 248.1 | 44.1 s | 2.12 kJ | 1.90 kJ |
| CfC | 4.11M | 277.1 | 212.0 | 680.5 s | 17.49 kJ | 15.19 kJ |

All three validation-loss curves decline over the measured 2.0 million training tokens. The Transformer improves substantially over the LSTM at a modest time/energy increase. CfC reaches the best recorded validation and test perplexity despite having the fewest parameters, but its training trace is about 15 times as long as the Transformer's and uses about 8 times its gross GPU energy. This is a quality-versus-cost trade-off, not evidence that CfC is universally preferable; the short data sample and one seed limit the conclusion.

The text comparison notebook also summarizes laptop/Colab training measurements. The four recorded intervals sum to approximately:

| Environment label | Model | Training time | Gross training energy |
|---|---|---:|---:|
| Laptop | LSTM | 36.1 s | 1.94 kJ |
| Laptop | Transformer | 44.1 s | 2.12 kJ |
| Laptop | CfC | 680.5 s | 17.49 kJ |
| Colab | LSTM | 35.8 s | 2.41 kJ |
| Colab | Transformer | 42.8 s | 2.88 kJ |
| Colab | CfC | 568.3 s | 23.88 kJ |

These are single training traces per model/environment, not independent trial averages. The text energy CSVs do not include cross-environment validation or test quality, so the comparison supports cost/time observations only.

### 5.2 HAR quality and training cost

Each environment below has one 100-epoch seed-0 training trace. The validation column is the final-epoch validation accuracy; test accuracy is the held-out test result after training.

| Environment label | Model | Final validation accuracy | Test accuracy | Training time | Gross training energy |
|---|---|---:|---:|---:|---:|
| Laptop | LSTM | 85.2% | 89.9% | 109.9 s | 5.49 kJ |
| Laptop | Transformer | 87.9% | 90.0% | 186.1 s | 6.43 kJ |
| Laptop | CfC | 67.5% | 64.1% | 6,117.4 s | 70.32 kJ |
| Colab | LSTM | 85.4% | 89.0% | 113.6 s | 7.47 kJ |
| Colab | Transformer | 88.2% | 90.5% | 77.8 s | 5.11 kJ |
| Colab | CfC | 72.3% | 67.6% | 1,611.9 s | 53.89 kJ |

The validation-accuracy curves show the Transformer starting at a high level and improving more gradually; the LSTM starts lower and improves substantially during training. In the saved runs, both finish near 90% test accuracy. CfC ends notably lower and takes much longer in both environments. The Transformer is the best quality/cost balance in the Colab trace; the LSTM is a little faster and slightly more energy-efficient on the laptop trace, with nearly the same test accuracy as the Transformer. Since these are one-seed traces, small differences should not be treated as statistically reliable.

### 5.3 Inference throughput and gross energy per item

The following are medians recomputed from the five inference trials in each laptop file; energy is **gross** joules per item. For text, an item is a token; for HAR, it is one classified sensor window.

#### Text (laptop-labelled runs)

| Model | Batch 1 tokens/s | Batch 1 J/token | Batch 137 tokens/s | Batch 137 J/token |
|---|---:|---:|---:|---:|
| LSTM | 3,224 | 0.003453 | 262,755 | 0.000173 |
| Transformer | 25,841 | 0.001023 | 149,504 | 0.000289 |
| CfC | 110 | 0.073526 | 13,781 | 0.002137 |

#### HAR (laptop-labelled and Colab-labelled runs)

| Model | Laptop B=1 samples/s; J/sample | Laptop B=256 samples/s; J/sample | Colab B=1 samples/s; J/sample | Colab B=256 samples/s; J/sample |
|---|---:|---:|---:|---:|
| LSTM | 745; 0.038452 | 16,598; 0.002703 | 979; 0.056654 | 7,492; 0.008810 |
| Transformer | 352; 0.016243 | 27,595; 0.001773 | 1,112; 0.033083 | 28,517; 0.002299 |
| CfC | 5; 1.092086 | 1,269; 0.004807 | 21; 1.739698 | 5,493; 0.007189 |

The batch-size curves generally rise in throughput and fall in energy per item as work is batched, because fixed overhead is amortized across more inputs. This is particularly pronounced for CfC, which is slowest at batch 1 but improves substantially at larger batches. LSTM leads the largest-batch text throughput in the laptop-labelled run; Transformer is much faster at text batch 1 and has the highest HAR throughput at batch 256 in both labelled environments. These rankings vary with task, batch size, and environment.

The curves are not perfectly monotonic in every measurement. For example, the Colab-labelled HAR LSTM has slightly lower median throughput at batch 256 than batch 16; the text Transformer throughput also largely plateaus at large batches. Saturation, memory pressure, device scheduling, clock/thermal state, and measurement variation can all affect these points. The existing plots' error bars are descriptive trial variability, not significance tests.

### 5.4 Peak memory and other measured signals

The local text training logs report peak allocated GPU memory of about **1.51 GB** for LSTM, **2.14 GB** for Transformer, and **1.58 GB** for CfC. This is per training run's recorded peak, and should not be directly equated with inference memory. Inference CSVs report a separate peak for each batch/trial; memory generally increases with batch size. The harness also records GPU temperature, peak power, clocks, and throttling flags, but these should be interpreted as diagnostic context rather than model-quality metrics.

## 6. Interpretation of the comparison graphs

The analysis notebooks ([LMSYS comparison](./LMSYS_Colab_vs_Laptop_Analysis.ipynb) and [HAR comparison](./HAR_Colab_vs_Laptop_Analysis.ipynb)) generate summary tables and figures from the saved CSVs:

1. **Training time and energy by model/environment:** These emphasize the very high CfC training cost, especially on HAR, and the relatively small cost gap between LSTM and Transformer on text. Model quality must be considered alongside these plots; cost alone does not select the best model.
2. **Text cumulative energy against tokens seen:** The traces rise as training progresses. Their slopes reflect energy per additional training interval; CfC's curve is much steeper in the saved laptop run.
3. **HAR validation accuracy against epoch:** LSTM improves from a weaker starting point, Transformer remains strongest or close to strongest, and CfC plateaus at lower validation accuracy in these runs.
4. **Inference throughput against batch size:** Throughput generally increases with batching, but gains level off for some model/environment combinations. CfC remains the slowest model at corresponding batch sizes in both tasks.
5. **Inference gross energy per token/sample against batch size:** Energy per item usually falls at larger batch sizes. The lowest value is not always attached to the highest throughput, so the right operating point depends on both latency/throughput and energy requirements.
6. **Cross-environment comparisons:** They show different run-level outcomes, not an isolated causal effect of “Colab versus laptop.” Hardware identifiers and fully controlled machine conditions are not present in the summarized CSVs.

## 7. Limitations and reproducibility cautions

- **Small text subset:** The local prepared corpus is a filtered 20,000-row sample yielding 3.41M train tokens, not the full LMSYS-Chat-1M corpus.
- **Single training seed:** Text quality results use seed 1337; HAR training records show seed 0. A 100-epoch curve is still one run, not 100 independent experiments.
- **Not parameter-matched:** Text model sizes range from 4.11M to 10.02M parameters. HAR sizes are closer but not exactly equal.
- **Text inference is not generation:** It measures fixed-window next-token forward passes; it does not measure decode latency, generated-token quality, KV caching, or end-to-end response latency.
- **Different dataset/model tasks:** Perplexity, activity accuracy, token throughput, and sample throughput measure different things and must not be collapsed into one ranking.
- **Environment attribution:** “Laptop” and “Colab” are folder/notebook labels. The exported records do not consistently identify exact GPU model, driver, runtime, or host state; some `commit` fields are `unknown`.
- **Trial interpretation:** HAR and laptop text inference have five repeated trials per configuration. Colab text CSVs contain ten records per configuration; the LMSYS notebook separates two measurement cohorts using idle-power profiles rather than claiming a particular device identity. Training intervals are not independent repeats.
- **Energy scope:** NVML gross energy covers the GPU measurement window, not the whole computer or facility. Idle-subtracted net energy depends on the chosen idle reference. CodeCarbon fields are supplementary estimates.
- **Comparison conditions:** Batch size, input lengths, model architecture, numerical precision, and hardware/runtime all affect the result. The saved text training runs use FP32, whereas measurement scripts also support other precision settings.
- **Dataset terms:** The raw LMSYS data and token files are subject to the dataset license; this report summarizes measurements and does not redistribute those files. The UCI HAR dataset's citation and use terms should be followed for publication.

## 8. Overall conclusion

The experiments demonstrate that model selection is a quality–cost decision. For the saved text run, CfC offers the best perplexity but at a steep training and inference-speed cost; LSTM is the economical training choice, and Transformer offers a middle ground with strong quality and competitive batched inference. For HAR, Transformer and LSTM are the practical leaders in the recorded runs, with similar held-out accuracy and far lower training cost than CfC. Batching is an effective way to improve throughput and reduce per-item energy, but the optimal batch depends on the architecture and environment.

These conclusions are useful as a description of the current experiment and a basis for a better-controlled follow-up. Stronger claims would require multiple training seeds, exact hardware/software capture, matched parameter or compute budgets, full-corpus text evaluation, and true autoregressive text-generation benchmarks.

## Source files and recorded evidence

- Text preprocessing and training: [prep_data.py](../afml_text_10_nov/prep_data.py), [train_text.py](../afml_text_10_nov/train_text.py), [sequence-length benchmark](../afml_text_10_nov/bench_seqlen.py)
- Text measurements/results: [measurement harness](../afml_text_10_nov/measure_updated.py), [training energy CSV](../afml_text_10_nov/results/text_train_energy.csv), [inference energy CSV](../afml_text_10_nov/results/text_infer_energy.csv), [prepared-data metadata](../afml_text_10_nov/prepared_data/meta.json)
- HAR training and models: [HAR.py](../afml_project_9_nov/HAR.py), [models.py](../afml_project_9_nov/models.py)
- HAR laptop/Colab comparison data: `laptop_har/`, `colab_har/`, `laptop_lmsys/`, and `colab_lmsys/` alongside this report.
