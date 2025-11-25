# Tunix Gemma Hackathon Implementation Plan

## Goal
Fine-tune Gemma 2B using Tunix on Kaggle TPUs to produce high-quality reasoning traces for Math and Creative Logic problems.

## Proposed Changes

### Dataset Generation
#### [NEW] data_generation.py
- Script to generate or format synthetic data.
- **Format**: `<reasoning>Step-by-step logic...</reasoning><answer>Final Answer</answer>`
- **Sources**: GSM8K (Math), Synthetic Logic Puzzles.

### Training Pipeline
#### [NEW] train.py
- **Library**: Tunix (JAX/Flax based).
- **Model**: Gemma 2B (Open Weights).
- **Config**: LoRA for efficient fine-tuning on TPU.
- **Loss**: Causal Language Modeling (CLM) loss on the reasoning + answer tokens.

### Inference
#### [NEW] inference.py
- Script to load the fine-tuned model and generate responses with reasoning traces.

## Verification Plan
### Automated Tests
- Run `data_generation.py` and verify output format.
- Run a single training step in `train.py` to verify TPU compatibility.