# Teaching Gemma to Reason: A Tunix Approach

## Introduction
In this project, we fine-tuned Google's Gemma 2B model to not only solve math and logic problems but to explicitly explain its reasoning step-by-step. By leveraging the **Tunix** library on TPUs, we enabled the model to generate `<reasoning>` traces before providing the final `<answer>`.

## Methodology
### Dataset
We generated a synthetic dataset of arithmetic and logic problems. Each sample follows a strict format:
- **Prompt**: The problem statement.
- **Completion**: `<reasoning>...step-by-step logic...</reasoning><answer>...final result...</answer>`

### Model & Training
- **Base Model**: Gemma 2B (Open Weights)
- **Library**: Tunix (JAX/Flax)
- **Optimization**: We used LoRA (Low-Rank Adaptation) to efficiently fine-tune the model on Kaggle's TPU v5e.
- **Objective**: Causal Language Modeling (CLM) loss was applied to the entire completion, encouraging the model to learn the structure of reasoning.

## Results
The fine-tuned model demonstrates a significant improvement in interpretability. Unlike the base model, which often outputs just the answer, our model consistently provides a logical derivation.

**Example Output:**
> **Prompt:** Solve 24 + 15
> **Model:** <reasoning>To solve 24 + 15, I will add the tens and ones separately. 20 + 10 = 30. 4 + 5 = 9. 30 + 9 = 39.</reasoning><answer>39</answer>

## Conclusion
This project highlights the power of open tools like Tunix and Gemma. By enforcing a reasoning structure during training, we created a more transparent and reliable AI assistant.