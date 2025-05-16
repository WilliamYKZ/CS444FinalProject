from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List
import json
from tqdm import tqdm

def load_model_and_tokenizer(model_path: str):
    """Load model and tokenizer from the given path."""
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def generate_response(model, tokenizer, prompt: str, max_length: int = 512):
    """Generate a response for the given prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the original prompt from the response
    response = response[len(prompt):].strip()
    return response

def run_benchmark(model_paths: List[str], test_prompts: List[str]):
    """Run benchmark comparison between models."""
    results = {}
    
    for model_path in model_paths:
        print(f"\nEvaluating model: {model_path}")
        model, tokenizer = load_model_and_tokenizer(model_path)
        model_responses = []
        
        for prompt in tqdm(test_prompts):
            response = generate_response(model, tokenizer, prompt)
            model_responses.append({
                "prompt": prompt,
                "response": response
            })
        
        results[model_path] = model_responses
        
        # Free up memory
        del model
        torch.cuda.empty_cache()
    
    return results

if __name__ == "__main__":
    # Define paths to your models
    model_paths = [
        "/home/exouser/Desktop/0_ModelCheckpoint/SFT_Model_Checkpoint/Checkpoint-5000",
        "/home/exouser/Desktop/0_ModelCheckpoint/RL_Model_Checkpoint/Checkpoint_2400"
    ]
    
    # Define test prompts
    test_prompts = [
        "Write a function to calculate the fibonacci sequence.",
        "Explain how to implement binary search.",
        "What are the best practices for writing clean code?",
        "How do I handle exceptions in Python?",
        # Add more test prompts here
    ]
    
    # Run benchmark
    results = run_benchmark(model_paths, test_prompts)
    
    # Save results
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2) 