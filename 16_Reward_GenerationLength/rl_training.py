# rlhf_training.py

from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import Adafactor, AutoTokenizer, HfArgumentParser, pipeline, set_seed
import wandb
wandb.init(project="Checkpoint_Reward") 

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from trl.core import LengthSampler


@dataclass
class ScriptArguments:
    # Model & Tokenizer
    model_name: Optional[str] = field(default="", metadata={"help": "Policy model checkpoint (or HF Hub name)"})
    tokenizer_name: Optional[str] = field(default="", metadata={"help": "Tokenizer name or path"})
    reward_model_name: Optional[str] = field(default="", metadata={"help": "Reward model path or name"})
    
    # Training
    log_with: Optional[str] = field(default=None, metadata={"help": "Set to 'wandb' for wandb logging"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "Learning rate for PPO"})
    output_max_length: Optional[int] = field(default=128, metadata={"help": "Max length for generated responses"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "PPO minibatch size"})
    batch_size: Optional[int] = field(default=8, metadata={"help": "Batch size per training step"})
    ppo_epochs: Optional[int] = field(default=4, metadata={"help": "Number of PPO epochs per step"})
    gradient_accumulation_steps: Optional[int] = field(default=4, metadata={"help": "Gradient accumulation steps"})
    adafactor: Optional[bool] = field(default=False, metadata={"help": "Use Adafactor optimizer"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "Use early stopping"})
    target_kl: Optional[float] = field(default=0.15, metadata={"help": "KL target for early stopping"})
    reward_baseline: Optional[float] = field(default=0.0, metadata={"help": "Subtract this baseline from reward"})
    batched_gen: Optional[bool] = field(default=False, metadata={"help": "Use batched generation"})
    save_freq: Optional[int] = field(default=100, metadata={"help": "Steps to save the model checkpoint"})
    output_dir: Optional[str] = field(default="runs/", metadata={"help": "Where to save model checkpoints"})
    seed: Optional[int] = field(default=0, metadata={"help": "Random seed"})
    steps: Optional[int] = field(default=100, metadata={"help": "Total PPO training steps"})
    init_kl_coef: Optional[float] = field(default=0.2, metadata={"help": "Initial KL penalty coefficient"})
    adap_kl_ctrl: Optional[bool] = field(default=True, metadata={"help": "Use adaptive KL control (else linear)"})
    load_in_8bit: Optional[bool] = field(default=True, metadata={"help": "Load policy model in 8-bit mode"})


parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
reward_model_name = script_args.reward_model_name
dataset_name = "lvwerra/stack-exchange-paired"

config = PPOConfig(
    steps=script_args.steps,               # total PPO steps
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    batch_size=script_args.batch_size,
    mini_batch_size=script_args.mini_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optimize_cuda_cache=True,
    early_stopping=script_args.early_stopping,
    target_kl=script_args.target_kl,
    ppo_epochs=script_args.ppo_epochs,
    seed=script_args.seed,
    init_kl_coef=script_args.init_kl_coef,
    adap_kl_ctrl=script_args.adap_kl_ctrl,
)

print(f"Using total PPO steps: {config.total_ppo_epochs}")

# 1. Load dataset
train_dataset = load_dataset("lvwerra/stack-exchange-paired", data_dir="data/rl", split="train", verification_mode="no_checks")
train_dataset = train_dataset.select(range(100000))
original_columns = train_dataset.column_names

# 2. Pipeline args
sent_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": 16,
    "truncation": True,
}

# 3. Tokenizer
tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name)
if getattr(tokenizer, "pad_token", None) is None:
    tokenizer.pad_token = tokenizer.eos_token

# 4. Build dataset
def build_dataset(tokenizer):
    num_proc = 24

    def preprocess_function(examples):
        new_examples = {"query": [], "input_ids": []}
        for question in examples["question"]:
            query = "Question: " + question + "\n\nAnswer: "
            tokenized_question = tokenizer(query, truncation=True)
            new_examples["query"].append(query)
            new_examples["input_ids"].append(tokenized_question["input_ids"])
        return new_examples

    ds = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    ds = ds.filter(lambda x: len(x["input_ids"]) < 512, batched=False, num_proc=num_proc)
    ds.set_format(type="torch")
    return ds

dataset = build_dataset(tokenizer)

def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}

set_seed(config.seed)

# 5. Load model with LoRA
current_device = Accelerator().local_process_index
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,  # can point to e.g. "/path/to/checkpoint-2000"
    load_in_8bit=script_args.load_in_8bit,
    device_map={"": current_device},
    peft_config=lora_config,
)

# 6. Optional Adafactor
optimizer = None
if script_args.adafactor:
    optimizer = Adafactor(
        filter(lambda p: p.requires_grad, model.parameters()),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        lr=config.learning_rate,
    )

ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer,
)

# 7. Reward model (pipeline)
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"

sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=reward_model_name,
    tokenizer=tokenizer,
    device_map={"": current_device},
    model_kwargs={"load_in_8bit": script_args.load_in_8bit},
    return_token_type_ids=False,
)
if sentiment_pipe.model.config.pad_token_id is None:
    sentiment_pipe.model.config.pad_token_id = sentiment_pipe.model.config.eos_token_id

# 8. Generation parameters
generation_kwargs = {
    "top_k": 50,
    "top_p": 0.95,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    #"eos_token_id": 100_000,
}
output_min_length = 32
output_max_length = script_args.output_max_length
output_length_sampler = LengthSampler(output_min_length, output_max_length)

# 9. Training loop
for step, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    # Stop after total PPO steps
    if step >= config.total_ppo_epochs:
        break

    # (A) Generate responses
    question_tensors = batch["input_ids"]
    response_tensors = ppo_trainer.generate(
        question_tensors,
        return_prompt=False,
        length_sampler=output_length_sampler,
        **generation_kwargs,
    )
    batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

    # (B) Compute rewards
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    # Typically pipe_outputs is a list of lists (for each text, the model might return multiple classes)
    rewards = []
    for output in pipe_outputs:
        # If it's a single label or the first label in a multi-label setting, use output[0]["score"]
        r = output[0]["score"] - script_args.reward_baseline
        rewards.append(torch.tensor(r))

    # (C) PPO step
    stats = ppo_trainer.step(question_tensors, response_tensors, rewards)

    # ---------------------------
    # LOG PER-SAMPLE REWARD & LENGTH
    # ---------------------------
    # 1) Log each sample's reward vs. step
    for i, rew in enumerate(rewards):
        wandb.log({"reward_per_sample": rew.item(), "step": step})

    # 2) Log each sample's generation length vs. step
    gen_lengths = [len(resp) for resp in response_tensors]
    for i, length in enumerate(gen_lengths):
        wandb.log({"gen_length_per_sample": length, "step": step})

    # ---------------------------
    # (OPTIONAL) LOG AVERAGES
    # ---------------------------
    avg_reward = torch.stack(rewards).mean().item()
    wandb.log({"avg_reward": avg_reward, "step": step})

    avg_gen_length = sum(gen_lengths) / len(gen_lengths)
    wandb.log({"avg_generation_length": avg_gen_length, "step": step})
    
    # if "train/mean_advantage" in stats:
    #     wandb.log({"mean_advantage": stats["train/mean_advantage"], "step": step})
    # elif "ppo/advantages_mean" in stats:
    #     wandb.log({"mean_advantage": stats["ppo/advantages_mean"], "step": step})

    # # 2) Per-sample advantage (if available). For example, "train/advantages" might be a list.
    # # You need to confirm how your TRL version returns them (print(stats) to check).
    # if "train/advantages" in stats and isinstance(stats["train/advantages"], list):
    #     for i, adv in enumerate(stats["train/advantages"]):
    #         wandb.log({"advantage_per_sample": adv, "step": step})
    # elif "ppo/advantages" in stats and isinstance(stats["ppo/advantages"], list):
    #     for i, adv in enumerate(stats["ppo/advantages"]):
    #         wandb.log({"advantage_per_sample": adv, "step": step})
    

    # (D) Log other stats from PPO
    # ppo_trainer.log_stats will log stats to wandb too
    ppo_trainer.log_stats(stats, batch, rewards)

    # (E) Save model periodically
    if script_args.save_freq and step > 0 and (step % script_args.save_freq == 0):
        save_path = f"{script_args.output_dir}/step_{step}"
        ppo_trainer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")







'''
accelerate launch --num_processes 1 /home/exouser/Desktop/16_Reward_GenerationLength/rl_training.py \
    --log_with=wandb  \
    --model_name /home/exouser/Desktop/0_ModelCheckpoint/Full_RL_Model/CheckPoint_500 \
    --reward_model_name /home/exouser/Desktop/DeepSeek-R1-Distill-Qwen-7B \
    --tokenizer_name /home/exouser/Desktop/Qwen2.5-3B-Instruct \
    --output_max_length 128  \
    --save_freq 20  \
    --batch_size 8  \
    --gradient_accumulation_steps 8 \
    --ppo_epochs 4 \
    --seed 42  \
    --learning_rate 1.4e-5  \
    --early_stopping False   \
    --output_dir ./my-rlhf-output  \
    --steps 720
'''