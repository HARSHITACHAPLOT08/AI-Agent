import random
import json

def generate_math_problem():
    """Generates a simple arithmetic problem with a reasoning trace."""
    a = random.randint(1, 100)
    b = random.randint(1, 100)
    op = random.choice(['+', '-', '*', '/'])
    
    if op == '+':
        answer = a + b
        reasoning = f"To solve {a} + {b}, I will add the two numbers. {a} plus {b} equals {answer}."
    elif op == '-':
        # Ensure positive result for simplicity
        if a < b: a, b = b, a
        answer = a - b
        reasoning = f"To solve {a} - {b}, I will subtract {b} from {a}. {a} minus {b} equals {answer}."
    elif op == '*':
        a = random.randint(1, 20) # Keep multiplication simple
        b = random.randint(1, 20)
        answer = a * b
        reasoning = f"To solve {a} * {b}, I will multiply the two numbers. {a} times {b} is {answer}."
    elif op == '/':
        b = random.randint(1, 12)
        answer = random.randint(1, 12)
        a = b * answer # Ensure clean division
        reasoning = f"To solve {a} / {b}, I will divide {a} by {b}. {a} divided by {b} is {answer}."

    return {
        "prompt": f"Solve this math problem: {a} {op} {b}",
        "completion": f"<reasoning>{reasoning}</reasoning><answer>{answer}</answer>"
    }

def generate_dataset(num_samples=1000, output_file="math_reasoning_data.jsonl"):
    print(f"Generating {num_samples} samples...")
    with open(output_file, 'w') as f:
        for _ in range(num_samples):
            sample = generate_math_problem()
            f.write(json.dumps(sample) + "\n")
    print(f"Dataset saved to {output_file}")

if __name__ == "__main__":
    generate_dataset()
