# Does the Efficiency Edge Travel?
### A Cross-Domain, Hardware-Measured Evaluation of CfC Against LSTM and Transformer Models

## Overview

This project investigates whether the computational efficiency of **Closed-form Continuous-time (CfC) neural networks** translates across different data domains and sequence-modelling tasks.

We compare three sequence-modelling architectures:

- **CfC (Closed-form Continuous-time)**
- **LSTM (Long Short-Term Memory)**
- **Transformer**

The models are evaluated on two different datasets:

1. **UCI Human Activity Recognition (HAR):** A time-series classification task using smartphone sensor data.
2. **LMSYS-Chat-1M:** A conversational language-modelling task using text sequences.

The evaluation considers model quality, training time, inference latency, throughput, GPU memory usage and energy consumption.

## Objectives

- Compare CfC, LSTM and Transformer under different sequence-modelling workloads.
- Evaluate model quality across time-series classification and language modelling.
- Measure training and inference performance.
- Analyse GPU memory usage and hardware-level energy consumption.
- Understand the trade-offs between predictive performance and computational efficiency.

## Datasets

### 1. UCI Human Activity Recognition (HAR)

- Task: Human activity classification.
- Input: 128 time steps with 9 inertial sensor channels.
- Classes: 6 human activities.
- Evaluation: Official train/test split, with a validation subset held out from the training subjects.

### 2. LMSYS-Chat-1M

- Task: Conversational language modelling.
- Source: LMSYS-Chat-1M dataset.
- Approximately 1 million records were scanned during preprocessing.
- 294,972 conversations were retained after filtering.
- Tokenizer vocabulary: 8,192 tokens.
- Training data: 60 million tokens.
- Validation data: 1 million tokens.
- Test data: 1 million tokens.

## Model Architectures

All three architectures were implemented and evaluated:

| Model | Description |
|---|---|
| CfC | Continuous-time recurrent neural network |
| LSTM | Recurrent neural network designed to capture long-term dependencies |
| Transformer | Attention-based sequence-modelling architecture |

The models were configured with approximately comparable parameter counts within each experiment to support a fairer architectural comparison.

## Experimental Setup

**Hardware:** NVIDIA Tesla T4 GPU.

The experiments measure:

| Category | Metrics |
|---|---|
| Model quality | HAR test accuracy, LMSYS test perplexity |
| Training | Training time |
| Inference | Latency and throughput |
| Memory | Peak GPU memory usage |
| Energy | Training energy where recorded and inference energy |

HAR experiments used FP32 precision, while LMSYS experiments used FP16 mixed precision. The workloads and measurement protocols therefore differ and should not be interpreted as perfectly identical cross-domain benchmarks.

## Key Results

### HAR

| Model | Test Accuracy |
|---|---:|
| LSTM | 89.04% |
| Transformer | 90.53% |
| CfC | 67.59% |

The Transformer achieved the highest test accuracy and the shortest training time. CfC used the least peak GPU memory during training but required substantially more training time.

### LMSYS

| Model | Test Perplexity |
|---|---:|
| LSTM | 49.83 |
| Transformer | 15.34 |
| CfC | 45.32 |

The Transformer achieved the lowest test perplexity, highest inference throughput and lowest measured inference energy. CfC had substantially higher inference latency and training time.

### Overall Observation

Across the evaluated workloads, the Transformer delivered the strongest overall combination of model quality and computational efficiency. The expected efficiency advantages of CfC were not observed in these experiments, although it demonstrated lower GPU memory usage during HAR training.

## Repository Structure

```text
AFML_Zenith/
│
├── HAR.py
├── models.py
├── models-2.py
├── train_text.py
├── prep_data.py
├── bench_lm.py
├── bench_seqlen.py
├── measure_updated.py
├── profile_ops.py
├── plot_bench.py
├── p3_pipeline.py
├── requirements.txt
│
├── vaishali_lmsys/
│   ├── prep_data.py
│   ├── train_text.py
│   ├── bench_seqlen.py
│   └── measure_updated.py
│
├── results/
│   ├── text/
│   │   ├── ppl_curves.png
│   │   ├── inference_energy.png
│   │   ├── inference_gpu_power.png
│   │   ├── inference_latency.png
│   │   ├── inference_throughput.png
│   │   └── CSV result files
│   │
│   ├── cross_domain_comparison/
│   │   ├── cross_domain_model_quality.png
│   │   ├── cross_domain_training_time.png
│   │   ├── cross_domain_inference_throughput.png
│   │   ├── cross_domain_inference_energy.png
│   │   ├── cross_domain_gpu_memory.png
│   │   └── CSV comparison files
│   │
│   └── vaishali_measurement_results/
│       ├── PROJECT_REPORT.md
│       └── Colab and laptop analysis
│
├── docs/
│   └── kaggle_t4_requirements.txt
│
└── requirements.txt
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Maansi-Sathish/AFML_Zenith.git
cd AFML_Zenith
```

Install the listed Python dependencies:

```bash
pip install -r requirements.txt
```

The experiments require a compatible PyTorch environment. GPU-based benchmarking and energy measurement require access to a supported NVIDIA GPU and its monitoring interface.

## Important Experimental Limitations

- LMSYS training energy was not recorded during the final training runs. Therefore, a complete training-energy comparison across both datasets is not available.
- HAR training energy and LMSYS inference energy were measured and are reported separately.
- Energy-above-idle measurements depend on the measured idle-power baseline.
- HAR and LMSYS use different precision settings, input formats, batch configurations and evaluation metrics.
- LMSYS inference benchmarking measures fixed-window forward passes, not autoregressive text generation.
- Results should be interpreted within the specific experimental configurations and hardware used.

## Technologies Used

- Python
- PyTorch
- NumPy
- NCPS (CfC implementation)
- NVIDIA Management Library (NVML)
- NVIDIA Tesla T4
- Matplotlib

## Contributors

This project was completed collaboratively. Contributions included dataset preparation, model implementation, training, hardware measurement, benchmarking, result analysis and report preparation.

## Repository

[GitHub – AFML_Zenith](https://github.com/Maansi-Sathish/AFML_Zenith)

---

**Project focus:** Understanding whether the efficiency characteristics of sequence-modelling architectures remain consistent across different data domains through controlled experiments and hardware-level evaluation.
