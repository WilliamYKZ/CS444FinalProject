import os
import csv
import torch
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import (
    Adafactor,
    AutoTokenizer,
    HfArgumentParser,
    pipeline,
    set_seed,
    BitsAndBytesConfig
)
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from trl.core import LengthSampler

tqdm.pandas()

@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="", metadata={"help": "the model name"})
    tokenizer_name: Optional[str] = field(default="", metadata={"help": "the tokenizer name"})
    reward_model_name: Optional[str] = field(default="", metadata={"help": "the reward model name"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    output_max_length: Optional[int] = field(default=128, metadata={"help": "maximum length for generation"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
    ppo_epochs: Optional[int] = field(default=4, metadata={"help": "the number of PPO epochs"})
    gradient_accumulation_steps: Optional[int] = field(default=4, metadata={"help": "the number of gradient accumulation steps"})
    adafactor: Optional[bool] = field(default=False, metadata={"help": "whether to use the adafactor optimizer"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "KL target for early stopping"})
    reward_baseline: Optional[float] = field(default=0.0, metadata={"help": "a baseline value subtracted from the reward"})
    batched_gen: Optional[bool] = field(default=False, metadata={"help": "whether to use batched text generation"})
    save_freq: Optional[int] = field(default=None, metadata={"help": "frequency (in steps) to save the model"})
    output_dir: Optional[str] = field(default="runs/", metadata={"help": "directory to save the model"})
    seed: Optional[int] = field(default=0, metadata={"help": "the seed"})
    steps: Optional[int] = field(default=20000, metadata={"help": "number of epochs"})
    init_kl_coef: Optional[float] = field(default=0.2, metadata={"help": "initial KL penalty coefficient"})
    adap_kl_ctrl: Optional[bool] = field(default=True, metadata={"help": "use adaptive KL control"})
    load_in_8bit: Optional[bool] = field(default=True, metadata={"help": "whether to load the model in 8bit"})

parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

reward_model_name = script_args.reward_model_name
dataset_name = "lvwerra/stack-exchange-paired"

config = PPOConfig(
    steps=script_args.steps,
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

# Load and prepare the dataset, then restrict to 80 examples.
train_dataset = load_dataset("lvwerra/stack-exchange-paired", data_dir="data/rl", split="train", verification_mode="no_checks")
train_dataset = train_dataset.select(range(100000))
original_columns = train_dataset.column_names

# Define pipeline arguments for the reward model.
sent_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": 16,
    "truncation": True,
}

tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name)
if getattr(tokenizer, "pad_token", None) is None:
    tokenizer.pad_token = tokenizer.eos_token

def build_dataset(tokenizer, dataset_name="lvwerra/stack-exchange-paired"):
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
# Restrict dataset to 80 examples.
dataset = dataset.select(range(80))

# Updated collator that pads tensor sequences to equal length.
def collator(data):
    collated = {}
    for key in data[0]:
        if isinstance(data[0][key], torch.Tensor):
            collated[key] = pad_sequence([d[key] for d in data], batch_first=True)
        else:
            collated[key] = [d[key] for d in data]
    return collated

# Set seed for deterministic evaluation.
set_seed(config.seed)

current_device = Accelerator().local_process_index

# Setup BitsAndBytesConfig if 8bit is enabled.
quant_config = BitsAndBytesConfig(load_in_8bit=True) if script_args.load_in_8bit else None

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    quantization_config=quant_config,
    device_map={"": current_device},
    peft_config=lora_config,
)

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

device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"

# For the reward model pipeline, pass quantization_config via model_kwargs if available.
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=reward_model_name,
    device_map={"": current_device},
    model_kwargs={"quantization_config": quant_config} if quant_config is not None else {},
    tokenizer=tokenizer,
    return_token_type_ids=False,
)
if sentiment_pipe.model.config.pad_token_id is None:
    sentiment_pipe.model.config.pad_token_id = sentiment_pipe.model.config.eos_token_id

generation_kwargs = {
    "top_k": 1,
    "top_p": 0.95,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": 100_000,
}
output_min_length = 32
output_max_length = script_args.output_max_length
output_length_sampler = LengthSampler(output_min_length, output_max_length)

# Prepare CSV files for logging experiment outputs.
os.makedirs(script_args.output_dir, exist_ok=True)
full_log_path = os.path.join(script_args.output_dir, "full_generation_results.csv")
trunc_log_path = os.path.join(script_args.output_dir, "truncated_results.csv")

full_log_f = open(full_log_path, "w", newline="")
trunc_log_f = open(trunc_log_path, "w", newline="")

full_writer = csv.DictWriter(full_log_f, fieldnames=["epoch", "prompt", "full_response", "full_reward", "full_critic_value", "full_advantage"])
full_writer.writeheader()

trunc_writer = csv.DictWriter(trunc_log_f, fieldnames=["epoch", "prompt", "truncated_response", "trunc_reward", "trunc_critic_value", "trunc_advantage"])
trunc_writer.writeheader()

# Training loop – with 80 examples total and batch_size=8 (10 iterations).
for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    # Get the padded tensor for input_ids.
    question_tensors = batch["input_ids"]
    # Convert the padded tensor into a list of 1D tensors.
    query_list = [q for q in question_tensors]

    response_tensors = ppo_trainer.generate(
        query_list,
        return_prompt=False,
        length_sampler=output_length_sampler,
        **generation_kwargs,
    )
    # Convert response_tensors (a list of tensors) into a padded tensor.
    response_tensors = torch.nn.utils.rnn.pad_sequence(response_tensors, batch_first=True)

    batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

    # Compute reward for full generation using the sentiment analysis pipeline.
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    rewards = [torch.tensor(output[0]["score"] - script_args.reward_baseline) for output in pipe_outputs]

    # Use query_list (a list of 1D tensors) when calling step.
    stats = ppo_trainer.step(query_list, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

    with torch.no_grad():
        # Full generation: concatenate question and response.
        full_input_ids = torch.cat([question_tensors.to(device), response_tensors.to(device)], dim=1)
        outputs_full = model(full_input_ids)
        full_critic_values = outputs_full.values[:, -1]

    for i in range(len(batch["query"])):
        prompt_text = batch["query"][i]
        full_response_text = batch["response"][i]
        full_reward_val = rewards[i].item()
        full_critic_val = full_critic_values[i].item()
        full_advantage = full_reward_val - full_critic_val

        full_writer.writerow({
            "epoch": epoch,
            "prompt": prompt_text,
            "full_response": full_response_text,
            "full_reward": full_reward_val,
            "full_critic_value": full_critic_val,
            "full_advantage": full_advantage,
        })

        # Process truncated (80%) responses.
        full_response_tokens = tokenizer.encode(full_response_text)
        trunc_length = max(1, int(0.8 * len(full_response_tokens)))
        truncated_tokens = full_response_tokens[:trunc_length]
        truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)

        # Recalculate reward using the truncated response (prompt + truncated response).
        combined_text = prompt_text + truncated_text
        pipe_output = sentiment_pipe(combined_text, **sent_kwargs)
        trunc_reward = float(pipe_output[0][0]["score"] - script_args.reward_baseline)

        # Recalculate critic value for the truncated response.
        question_tokens = question_tensors[i].unsqueeze(0).to(device)
        truncated_response_tensor = torch.tensor(truncated_tokens).unsqueeze(0).to(device)
        truncated_input_ids = torch.cat([question_tokens, truncated_response_tensor], dim=1)
        outputs_trunc = model(truncated_input_ids)
        trunc_critic_val = outputs_trunc.values[:, -1].item()

        trunc_advantage = trunc_reward - trunc_critic_val

        trunc_writer.writerow({
            "epoch": epoch,
            "prompt": prompt_text,
            "truncated_response": truncated_text,
            "trunc_reward": trunc_reward,
            "trunc_critic_value": trunc_critic_val,
            "trunc_advantage": trunc_advantage,
        })

    if script_args.save_freq and epoch and epoch % script_args.save_freq == 0:
        ppo_trainer.save_pretrained(os.path.join(script_args.output_dir, f"step_{epoch}"))


# Close CSV files after training.
full_log_f.close()
trunc_log_f.close()





'''
accelerate launch --num_processes 1 /home/exouser/Desktop/20_Advantage/Experiment.py \
    --log_with=wandb  \
    --model_name /home/exouser/Desktop/0_ModelCheckpoint/Full_RL_Model/CheckPoint_500 \
    --reward_model_name /home/exouser/Desktop/DeepSeek-R1-Distill-Qwen-7B \
    --tokenizer_name /home/exouser/Desktop/Qwen2.5-3B-Instruct \
    --output_max_length 1024  \
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