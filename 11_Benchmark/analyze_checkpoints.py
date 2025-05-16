from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

def load_model_and_tokenizer(model_path: str):
    """Load model and tokenizer from the given path."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer

def generate_and_measure(model, tokenizer, prompt: str, max_length: int = 512):
    """Generate response and return input/output token lengths."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = len(inputs.input_ids[0])
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(prompt):].strip()
    output_length = len(tokenizer.encode(response))
    
    # Get token IDs
    input_token_ids = inputs.input_ids[0].tolist()
    output_token_ids = outputs[0].tolist()
    
    # Create detailed info for logging
    detail_info = {
        'prompt': prompt,
        'input_length': input_length,
        'output_length': output_length,
        'input_token_ids': input_token_ids,
        'output_token_ids': output_token_ids,
        'response': response
    }
    
    return input_length, output_length, detail_info

def analyze_checkpoints(checkpoints, test_prompts):
    results = {}
    all_details = []
    
    for checkpoint in checkpoints:
        print(f"\nAnalyzing checkpoint: {checkpoint}")
        model, tokenizer = load_model_and_tokenizer(checkpoint)
        
        input_lengths = []
        output_lengths = []
        checkpoint_details = []
        
        for prompt in tqdm(test_prompts):
            in_len, out_len, details = generate_and_measure(model, tokenizer, prompt)
            input_lengths.append(in_len)
            output_lengths.append(out_len)
            
            # Add checkpoint info to details
            details['checkpoint'] = checkpoint.split('/')[-1]
            checkpoint_details.append(details)
            
        results[checkpoint.split('/')[-1]] = {
            'input_lengths': input_lengths,
            'output_lengths': output_lengths
        }
        all_details.extend(checkpoint_details)
        
        # Free up memory
        del model
        torch.cuda.empty_cache()
    
    # Save detailed information to txt file
    with open('token_details.txt', 'w') as f:
        f.write("Token Analysis Details\n")
        f.write("=" * 50 + "\n\n")
        
        for detail in all_details:
            f.write(f"Checkpoint: {detail['checkpoint']}\n")
            f.write(f"Prompt: {detail['prompt']}\n")
            f.write(f"Input length: {detail['input_length']}\n")
            f.write(f"Output length: {detail['output_length']}\n")
            f.write("Input token IDs: " + str(detail['input_token_ids']) + "\n")
            f.write("Output token IDs: " + str(detail['output_token_ids']) + "\n")
            f.write(f"Response: {detail['response'][:100]}...\n")
            f.write("-" * 50 + "\n\n")
    
    return results

def plot_cdf(results, filename='token_length_cdf.png'):
    """Plot CDF of token lengths for each checkpoint."""
    plt.figure(figsize=(12, 8))
    
    colors = ['b', 'r', 'g']  # Different colors for each checkpoint
    for (checkpoint, data), color in zip(results.items(), colors):
        # Combine input and output lengths for overall distribution
        token_lengths = data['output_lengths']
        
        # Sort lengths for CDF
        sorted_lengths = np.sort(token_lengths)
        # Calculate probabilities
        probabilities = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
        
        # Plot CDF
        plt.plot(sorted_lengths, probabilities, 
                label=f'Checkpoint {checkpoint}',
                color=color,
                marker='o',
                markersize=6,
                alpha=0.6)
    
    plt.xlabel('Token Length')
    plt.ylabel('Cumulative Probability')
    plt.title('CDF of Output Token Lengths Across Checkpoints')
    plt.grid(True)
    plt.legend()
    
    # Save to specific folder
    save_path = "/home/exouser/Desktop/11_Benchmark/checkpoint_comparison/" + filename
    plt.savefig(save_path)
    plt.close()

def run_multiple_experiments(checkpoints, test_prompts, num_runs=10, keep_last=5):
    """Run multiple experiments and save individual and averaged plots."""
    all_results = []
    
    # Run experiments multiple times
    for run in range(num_runs):
        print(f"\nRunning experiment {run + 1}/{num_runs}")
        results = analyze_checkpoints(checkpoints, test_prompts)
        all_results.append(results)
        
        # Plot individual experiment
        plot_results(results, f'checkpoint_comparison_{run+1}.png')
        # Add CDF plot for each run
        plot_cdf(results, f'token_length_cdf_{run+1}.png')
    
    # Average the last N experiments
    last_n_results = all_results[-keep_last:]
    averaged_results = {}
    
    for checkpoint in checkpoints:
        checkpoint_name = checkpoint.split('/')[-1]
        averaged_results[checkpoint_name] = {
            'input_lengths': [],
            'output_lengths': []
        }
        
        # For each prompt position
        for i in range(len(test_prompts)):
            input_lengths = [run[checkpoint_name]['input_lengths'][i] for run in last_n_results]
            output_lengths = [run[checkpoint_name]['output_lengths'][i] for run in last_n_results]
            
            # Calculate averages
            avg_input = sum(input_lengths) / len(input_lengths)
            avg_output = sum(output_lengths) / len(output_lengths)
            
            averaged_results[checkpoint_name]['input_lengths'].append(avg_input)
            averaged_results[checkpoint_name]['output_lengths'].append(avg_output)
    
    # Plot averaged results
    plot_results(averaged_results, 'checkpoint_comparison_last_five.png', 
                title='Average Input vs Output Length (Last 5 Runs)')
    # Add CDF plot for averaged results
    plot_cdf(averaged_results, 'token_length_cdf_last_five.png')
    
    return averaged_results

def plot_results(results, filename, title='Input vs Output Length Across Different Checkpoints'):
    plt.figure(figsize=(12, 8))
    
    colors = ['b', 'r', 'g']  # Different colors for each checkpoint
    for (checkpoint, data), color in zip(results.items(), colors):
        # Plot scatter points
        plt.scatter(
            data['input_lengths'],
            data['output_lengths'],
            label=f'Checkpoint {checkpoint}',
            alpha=0.6,
            c=color
        )
        
        # Add trend line
        z = np.polyfit(data['input_lengths'], data['output_lengths'], 1)
        p = np.poly1d(z)
        plt.plot(data['input_lengths'], p(data['input_lengths']), 
                linestyle='--', alpha=0.8, c=color)
    
    plt.xlabel('Input Length (tokens)')
    plt.ylabel('Output Length (tokens)')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    
    # Save to specific folder
    save_path = "/home/exouser/Desktop/11_Benchmark/checkpoint_comparison/" + filename
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    # Define checkpoint paths
    checkpoints = [
        "/home/exouser/Desktop/0_ModelCheckpoint/RL_Model_Checkpoint/Checkpoint_200",
        "/home/exouser/Desktop/0_ModelCheckpoint/RL_Model_Checkpoint/Checkpoint_1200",
        "/home/exouser/Desktop/0_ModelCheckpoint/RL_Model_Checkpoint/Checkpoint_2400"
    ]
    
    # Define test prompts with varying lengths
    test_prompts = [
        "What is Python?",
        "Explain how to use a for loop in Python with an example.",
        "Can you explain the concept of object-oriented programming and provide a detailed example with multiple classes?",
        "Write a comprehensive guide on implementing a binary search tree, including insertion, deletion, and traversal methods.",
        "Explain the differences between various sorting algorithms (bubble sort, quick sort, merge sort) and analyze their time complexities with examples.",
        # Add more prompts as needed
    ]
    
    # Run multiple experiments
    averaged_results = run_multiple_experiments(checkpoints, test_prompts, num_runs=10, keep_last=5)
    print("\nAnalysis complete! Check the generated PNG files for visualizations.") 