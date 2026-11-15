
# LMSYS Text Domain — Experimental Results

This directory contains the results of the LMSYS conversational
text-domain experiments.

## Experiments

1. Model training and evaluation
2. Test perplexity comparison
3. Inference latency and throughput
4. GPU memory measurement
5. Five-repeat inference energy measurement
6. Performance visualization graphs

## Models

- LSTM
- Transformer
- CfC

## Hardware

NVIDIA Tesla T4 GPU.

## Important notes

- Training used 60 million tokens.
- Validation and test splits contained 1 million tokens each.
- Inference energy was estimated using sampled nvidia-smi GPU power.
- Energy-above-idle values are approximate.
- Raw datasets and model checkpoints are stored separately.
