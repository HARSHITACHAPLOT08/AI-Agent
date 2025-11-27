import os
import json
import jax
import jax.numpy as jnp
import optax

import tunix 
from tunix.models import GemmaConfig, GemmaModel
from flax.training import train_state


BATCH_SIZE = 8
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
MAX_LENGTH = 512
MODEL_NAME = "gemma_2b"

def load_data(file_path):
    """Loads the dataset from a JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def preprocess_batch(batch, tokenizer):
    """Tokenizes and formats the batch for training."""
    inputs = [x['prompt'] for x in batch]
    targets = [x['completion'] for x in batch]
    full_texts = [f"{p} {t}" for p, t in zip(inputs, targets)]
    
    encodings = tokenizer(
        full_texts, 
        padding="max_length", 
        truncation=True, 
        max_length=MAX_LENGTH, 
        return_tensors="jax"
    )
    return encodings

def train_step(state, batch):
    """Performs a single training step."""
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['input_ids'])
        shift_logits = logits[..., :-1, :]
        shift_labels = batch['input_ids'][..., 1:]
        loss = optax.softmax_cross_entropy_with_integer_labels(shift_logits, shift_labels).mean()
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

def main():
    print(f"Initializing {MODEL_NAME} with Tunix...")
    
    config = GemmaConfig.from_pretrained(MODEL_NAME)
    model = GemmaModel(config)
    tokenizer = tunix.AutoTokenizer.from_pretrained(MODEL_NAME)
    
    
    # FIX 1: Initialize parameters properly (Random init shown here; check Tunix docs for pretrained loading)
    print("Initializing model parameters...")
    key = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, MAX_LENGTH), dtype=jnp.int32)
    variables = model.init(key, dummy_input)
    params = variables['params']
    
    tx = optax.adamw(LEARNING_RATE)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    
    data = load_data("math_reasoning_data.jsonl")
    print(f"Loaded {len(data)} training samples.")
    
    for epoch in range(NUM_EPOCHS):
        print(f"Starting Epoch {epoch + 1}/{NUM_EPOCHS}")
        for i in range(0, len(data), BATCH_SIZE):
            batch_raw = data[i:i+BATCH_SIZE]
            batch = preprocess_batch(batch_raw, tokenizer)
            state, loss = train_step(state, batch)
            if i % 10 == 0:
                print(f"  Step {i}, Loss: {loss:.4f}")
                
    print("Saving model to checkpoints/best_model...")
    # FIX 2: Correct indentation
    tunix.save_checkpoint("checkpoints/best_model", state.params)
    
    print("Training complete!")

if __name__ == "__main__":
    main()
