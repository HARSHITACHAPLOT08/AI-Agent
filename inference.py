import jax
import jax.numpy as jnp
import tunix 
from tunix.models import GemmaConfig, GemmaModel 

def load_model(model_path):
    """Loads the fine-tuned model and tokenizer."""
    print(f"Loading model from {model_path}...")
    
tokenizer = tunix.AutoTokenizer.from_pretrained("gemma_2b")
config = GemmaConfig.from_pretrained("gemma_2b")
model = GemmaModel(config)
params = tunix.load_checkpoint(model_path)
return model, params, tokenizer
    
    
return "MockModel", "MockParams", "MockTokenizer"

def generate_response(prompt, model, params, tokenizer):
    """Generates a response with reasoning trace."""
    print(f"Generating response for: {prompt}")
    
    
inputs = tokenizer(prompt, return_tensors="jax")
outputs = model.generate(params, **inputs, max_new_tokens=200)
decoded = tokenizer.decode(outputs[0])
return decoded
    
    
return f"<reasoning>To solve this, I will simulate the reasoning process...</reasoning><answer>42</answer>"

def main():
    model_path = "checkpoints/best_model"
    model, params, tokenizer = load_model(model_path)
    
    test_prompts = [
        "Solve this math problem: 10 + 5",
        "Solve this math problem: 12 * 4"
    ]
    
    for prompt in test_prompts:
        response = generate_response(prompt, model, params, tokenizer)
        print(f"Output:\n{response}\n{'-'*20}")

if __name__ == "__main__":
    main()
