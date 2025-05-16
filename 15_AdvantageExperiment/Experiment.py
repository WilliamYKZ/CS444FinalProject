#!/usr/bin/env python

import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from tqdm import tqdm
from transformers import (
    pipeline,
    HfArgumentParser,
    AutoTokenizer,
)
from dataclasses import dataclass, field
from typing import Optional
from trl import AutoModelForCausalLMWithValueHead, PPOTrainer, PPOConfig


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="gpt2")
    tokenizer_name: Optional[str] = field(default="gpt2")
    reward_model_name: Optional[str] = field(default=None)
    batch_count: Optional[int] = field(default=10)


def compute_value(model_with_value_head, input_ids):
    with torch.no_grad():
        out = model_with_value_head.forward_value(input_ids)
        values = out[:, -1]
    return values

def partial_generation_experiment(
    ppo_trainer,
    tokenizer,
    sentiment_pipe,
    model,
    batch_count=10,
    fractions=(0.3, 0.6, 0.9, 1.0),
    reward_baseline=0.0,
):
    all_advantages = {f: [] for f in fractions}
    partial_adv_map = {f: [] for f in fractions}

    dataloader = ppo_trainer.dataloader
    for step_idx, batch in tqdm(enumerate(dataloader), total=batch_count, desc="Partial Gen Experiment"):
        if step_idx >= batch_count:
            break

        question_tensors = batch["input_ids"].to(ppo_trainer.accelerator.device)
        full_generation = ppo_trainer.generate(
            question_tensors,
            return_prompt=False,
            top_k=0,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        
        batch_size = full_generation.size(0)

        for i in range(batch_size):
            full_tokens = full_generation[i]
            prompt_len = question_tensors[i].ne(tokenizer.pad_token_id).sum().item()
            total_len = full_tokens.ne(tokenizer.pad_token_id).sum().item()
            new_tokens_len = total_len - prompt_len

            for f in fractions:
                keep_new_tokens = max(1, int(new_tokens_len * f))
                partial_len = prompt_len + keep_new_tokens
                partial_tokens = full_tokens[:partial_len].unsqueeze(0)
                partial_text = tokenizer.decode(partial_tokens[0], skip_special_tokens=True)

                pipe_outputs = sentiment_pipe([partial_text], return_all_scores=True)
                partial_reward = pipe_outputs[0][0]["score"] - reward_baseline

                partial_value = compute_value(model, partial_tokens).item()
                advantage = partial_reward - partial_value

                all_advantages[f].append(advantage)
                partial_adv_map[f].append(advantage)

    plot_fraction_advantages(all_advantages, fractions)
    
    full_advances = np.array(partial_adv_map[1.0])
    for f in fractions:
        if f == 1.0:
            continue
        partial_advances = np.array(partial_adv_map[f])
        pearson_corr, pearson_p = st.pearsonr(partial_advances, full_advances)
        spearman_corr, spearman_p = st.spearmanr(partial_advances, full_advances)
        
        print(f"Fraction: {f:.0%}")
        print(f"  Pearson Corr vs Full: {pearson_corr:.3f}, p={pearson_p:.2e}")
        print(f"  Spearman Corr vs Full: {spearman_corr:.3f}, p={spearman_p:.2e}")
        print("-" * 40)

def plot_fraction_advantages(all_advantages, fractions):
    fractions_list = []
    means = []
    stds = []
    
    for f in fractions:
        arr = np.array(all_advantages[f])
        fractions_list.append(f)
        means.append(arr.mean())
        stds.append(arr.std())
    
    plt.figure(figsize=(7,5))
    plt.errorbar(
        fractions_list, means, yerr=stds, fmt='-o',
        capsize=3, label='Mean Advantage ±1 std'
    )
    plt.title("Partial vs. Full Generation Advantage")
    plt.xlabel("Fraction of Generated Tokens")
    plt.ylabel("Advantage (Reward - Value)")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    
    # Load your model
    model = AutoModelForCausalLMWithValueHead.from_pretrained(script_args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name)

    # Reward model
    sentiment_pipe = pipeline("sentiment-analysis", model=script_args.reward_model_name)

    # Construct your PPO trainer
    ppo_config = PPOConfig(model_name=script_args.model_name, steps=1000)
    ppo_trainer = PPOTrainer(ppo_config, model, tokenizer=tokenizer, dataset=None)

    partial_generation_experiment(
        ppo_trainer=ppo_trainer,
        tokenizer=tokenizer,
        sentiment_pipe=sentiment_pipe,
        model=model,
        batch_count=script_args.batch_count,
        fractions=(0.3, 0.6, 0.9, 1.0),
        reward_baseline=0.0,
    )
