# 🧠 Teaching Gemma to Reason with Tunix

![Tunix + Gemma](https://img.shields.io/badge/Built%20With-Tunix-blueviolet) ![Gemma](https://img.shields.io/badge/Model-Gemma%202B-green) ![License](https://img.shields.io/badge/License-MIT-blue)

A hackathon project that fine-tunes Google's **Gemma 2B** model to solve math problems with explicit **Chain-of-Thought (CoT)** reasoning, powered by the **Tunix** library on TPUs.

## 🚀 Project Overview

Standard language models often jump straight to an answer, making them prone to hallucination and calculation errors. This project enforces a structured reasoning process by fine-tuning Gemma to output a `<reasoning>` trace before the final `<answer>`.

**Key Features:**
- **Structured Output**: Forces the model to "show its work" using XML-like tags.
- **Efficient Training**: Utilizes **Tunix** (JAX/Flax) for high-performance training on TPUs.
- **Synthetic Data Pipeline**: Includes a generator for creating infinite math reasoning datasets.

## 🛠️ Methodology

### 1. Dataset Generation
We generate synthetic arithmetic problems (Addition, Subtraction, Multiplication, Division) paired with step-by-step natural language explanations.
- **Format**: 
  ```xml
  Prompt: Solve 24 + 15
  Completion: <reasoning>To solve 24 + 15, I will add the tens...</reasoning><answer>39</answer>
  ```

### 2. Fine-Tuning with Tunix
- **Base Model**: `google/gemma-2b`
- **Optimization**: LoRA (Low-Rank Adaptation) for memory efficiency.
- **Framework**: Tunix (built on JAX/Flax) for seamless TPU scaling.

## 📂 Project Structure

```bash
├── data_generation.py    # Generates synthetic math problems with reasoning traces
├── train.py              # Main training script using Tunix (JAX/Flax)
├── inference.py          # Script to load the model and test reasoning capabilities
├── math_reasoning_data.jsonl # Generated dataset (ignored in git)
└── README.md             # Project documentation
```

## 💻 Installation & Usage

### Prerequisites
- Python 3.10+
- JAX & Flax (TPU/GPU support recommended)
- Tunix Library

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install flax tunix optax
```

### 1. Generate Data
Create the synthetic dataset for training.
```bash
python data_generation.py
```
*Output: `math_reasoning_data.jsonl` (1000 samples)*

### 2. Train the Model
Fine-tune Gemma on the generated data.
```bash
python train.py
```
*Saves checkpoints to `checkpoints/best_model`*

### 3. Run Inference
Test the model's ability to reason on new problems.
```bash
python inference.py
```

## 📊 Results

**Prompt:** `Solve 12 * 4`

**Model Output:**
> `<reasoning>` To solve 12 * 4, I will multiply 12 by 4. 10 times 4 is 40. 2 times 4 is 8. 40 plus 8 is 48. `</reasoning>` `<answer>` 48 `</answer>`

## 🤝 Acknowledgments
- **Google DeepMind** for the open-weights Gemma model.
- **Tunix Team** for the efficient JAX-based training library.
