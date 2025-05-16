#!/usr/bin/env python
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np

from datasets import load_dataset
from transformers import AutoTokenizer, pipeline
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from peft import LoraConfig
from accelerate import Accelerator



def build_dataset(tokenizer, dataset_name="lvwerra/stack-exchange-paired", size=100000):
    """Build a dataset just like in your RL training script."""
    # Feel free to copy exactly from your training script,
    # or adapt this to your custom dataset pipeline.
    train_dataset = load_dataset(dataset_name, data_dir="data/rl", split="train", verification_mode="no_checks")
    train_dataset = train_dataset.select(range(size))
    original_columns = train_dataset.column_names

    num_proc = 1  # adjust as needed for multiprocessing

    def preprocess_function(examples):
        new_examples = {"query": [], "input_ids": []}
        for question in examples["question"]:
            query = "Question: " + question + "\n\nAnswer: "
            tokenized_question = tokenizer(query, truncation=True)
            new_examples["query"].append(query)
            new_examples["input_ids"].append(tokenized_question["input_ids"])
        return new_examples

    ds = train_dataset.map(preprocess_function, batched=True, num_proc=num_proc, remove_columns=original_columns)
    ds = ds.filter(lambda x: len(x["input_ids"]) < 512, batched=False, num_proc=num_proc)
    ds.set_format(type="torch")
    return ds


def collator(data):
    return {
        key: [item[key] for item in data]
        for key in data[0].keys()
    }



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Path to your RLHF actor checkpoint")
    parser.add_argument("--reward_model_name", type=str, required=True, help="Path to your reward model checkpoint")
    parser.add_argument("--tokenizer_name", type=str, required=True, help="Path to your tokenizer")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for the comparison experiment")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum generation length for full generation")
    parser.add_argument("--partial_ratio", type=float, default=0.9, help="Ratio for partial generation (e.g., 0.8 = 80%)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cpu or cuda)")
    parser.add_argument("--output_dir", type=str, default="adv_experiment/", help="Where to save the figure")
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build or load a small dataset
    dataset = build_dataset(tokenizer)
    
    # Take just one batch for demonstration
    # (You can also iterate if you want multiple batches)
    sample_batch = collator([dataset[i] for i in range(args.batch_size)])
    
    # Create the PPO configuration
    config = PPOConfig(
        model_name=args.model_name,
        steps=1,  # We just need a single "step" for demonstration
        batch_size=args.batch_size,
        mini_batch_size=1,
        ppo_epochs=1,
        log_with=None,
    )

    # Setup LoRA (if used in training)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Load Actor model
    current_device = Accelerator().local_process_index
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
    args.model_name,
    device_map="auto",
    #torch_dtype=torch.float16,  # or load_in_8bit=True
    load_in_8bit=True,
    )

    # Initialize PPOTrainer (this will hold the model, dataset is optional here)
    ppo_trainer = PPOTrainer(
        config=config,
        model=model,
        ref_model=None,  # you could load a reference model if used for KL
        tokenizer=tokenizer,
        dataset=None,     # not strictly needed for a one-off test
        data_collator=collator,
    )

    # Build your reward pipeline
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model=args.reward_model_name,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype=torch.float16,
        model_kwargs={"load_in_8bit": True},  # Put it here
        return_token_type_ids=False,
    )

    if sentiment_pipe.model.config.pad_token_id is None:
        sentiment_pipe.model.config.pad_token_id = sentiment_pipe.model.config.eos_token_id

    # -------------------------------------------------------------------------
    # 1) PARTIAL GENERATION (80% of max_length)
    # -------------------------------------------------------------------------
    partial_len = int(args.partial_ratio * args.max_length)
    partial_sampler = LengthSampler(partial_len, partial_len + 1)  # fixed length

    generation_kwargs_partial = {
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
    }

    question_tensors = sample_batch["input_ids"]
    response_tensors_partial = ppo_trainer.generate(
        question_tensors,
        return_prompt=False,
        length_sampler=partial_sampler,
        **generation_kwargs_partial,
    )
    partial_texts = tokenizer.batch_decode(response_tensors_partial, skip_special_tokens=True)

    # Rewards
    combined_texts_partial = [
        q + resp for q, resp in zip(sample_batch["query"], partial_texts)
    ]
    rewards_partial = []
    pipe_outputs = sentiment_pipe(combined_texts_partial, truncation=True, return_all_scores=True)
    for output in pipe_outputs:
        # Example: You might pick output[0]["score"] if there's only 1 label, or
        # a difference if it's 2-class. Adjust as needed for your reward model.
        r = output[0]["score"]  
        rewards_partial.append(torch.tensor(r))

    # PPO step for partial generation to get advantage
    # (Be aware this updates model weights by default.)
    stats_partial = ppo_trainer.step(question_tensors, response_tensors_partial, rewards_partial)
    adv_partial = stats_partial["ppo/policy/advantages"]  # shape ~ (batch_size * seq_len,)

    # Convert to numpy and reshape
    adv_partial = np.array(adv_partial)  # if it's already numpy, might skip this
    adv_partial = adv_partial.reshape((args.batch_size, -1))
    adv_partial = adv_partial.mean(axis=1)  # shape (8,)



    # -------------------------------------------------------------------------
    # 2) FULL GENERATION (100% of max_length)
    # -------------------------------------------------------------------------
    # (Re-load the original checkpoint if you want to avoid model updates from partial step.)
    # For simplicity, we'll skip the re-load here, but you can do it if needed.
    
    full_sampler = LengthSampler(args.max_length, args.max_length + 1)
  # fixed length

    generation_kwargs_full = {
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
    }

    response_tensors_full = ppo_trainer.generate(
        question_tensors,
        return_prompt=False,
        length_sampler=full_sampler,
        **generation_kwargs_full,
    )
    full_texts = tokenizer.batch_decode(response_tensors_full, skip_special_tokens=True)

    combined_texts_full = [
        q + resp for q, resp in zip(sample_batch["query"], full_texts)
    ]
    rewards_full = []
    pipe_outputs = sentiment_pipe(combined_texts_full, truncation=True, return_all_scores=True)
    for output in pipe_outputs:
        r = output[0]["score"]
        rewards_full.append(torch.tensor(r))

    stats_full = ppo_trainer.step(question_tensors, response_tensors_full, rewards_full)
    adv_full = stats_full["ppo/policy/advantages"]

    adv_full = np.array(adv_full).reshape((args.batch_size, -1)).mean(axis=1)


    # -------------------------------------------------------------------------
    # 3) PLOT RESULTS
    # -------------------------------------------------------------------------
    x = list(range(1, args.batch_size + 1))  # e.g. [1..8] for batch size 8
    plt.figure(figsize=(8, 5))
    plt.plot(x, adv_partial, marker='o', color='red', label='80% Generation')
    plt.plot(x, adv_full,   marker='x', color='blue', label='Full Generation')

    plt.xlabel("Data index in batch")
    plt.ylabel("Advantage")
    plt.title("Advantage Comparison: 80% vs Full Generation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    save_path = f"{args.output_dir}/advantage_comparison.png"
    plt.savefig(save_path)
    print(f"[INFO] Figure saved to {save_path}")


if __name__ == "__main__":
    main()



'''
python Experiment.py \
  --model_name /home/exouser/Desktop/0_ModelCheckpoint/Full_RL_Model/CheckPoint_300 \
  --reward_model_name /home/exouser/Desktop/DeepSeek-R1-Distill-Qwen-7B \
  --tokenizer_name /home/exouser/Desktop/0_ModelCheckpoint/Full_RL_Model/CheckPoint_300 \
  --batch_size 4 \
  --max_length 128 \
  --partial_ratio 0.9 \
  --device cuda \
  --output_dir 15_AdvantageExperiment/

'''