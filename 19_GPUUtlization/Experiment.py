# Experiment.py
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Optional
import time
import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import Adafactor, AutoTokenizer, HfArgumentParser, pipeline, set_seed
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from trl.core import LengthSampler

# ----------------- ADDED IMPORTS FOR GPU MEASUREMENT & PLOTTING ----------------- #
import pynvml
import matplotlib.pyplot as plt
import threading
# ------------------------------------------------------------------------------- #

# Initialize NVML once at the start
pynvml.nvmlInit()

@dataclass
class ScriptArguments:
    """
    The name of the Causal LM model we wish to fine-tune with PPO.
    """
    model_name: Optional[str] = field(default="", metadata={"help": "the model name"})
    tokenizer_name: Optional[str] = field(default="", metadata={"help": "the tokenizer name"})
    reward_model_name: Optional[str] = field(default="", metadata={"help": "the reward model name"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    output_max_length: Optional[int] = field(default=128, metadata={"help": "maximum length for generation"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
    ppo_epochs: Optional[int] = field(default=4, metadata={"help": "the number of ppo epochs"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    adafactor: Optional[bool] = field(default=False, metadata={"help": "whether to use the adafactor optimizer"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
    reward_baseline: Optional[float] = field(
        default=0.0,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    batched_gen: Optional[bool] = field(default=False, metadata={"help": "whether to use the batched text gen"})
    save_freq: Optional[int] = field(default=None, metadata={"help": "n steps to save the model"})
    output_dir: Optional[str] = field(default="runs/", metadata={"help": "Directory to save the model"})
    seed: Optional[int] = field(default=0, metadata={"help": "the seed"})
    steps: Optional[int] = field(default=20000, metadata={"help": "number of total steps"})
    init_kl_coef: Optional[float] = field(
        default=0.2,
        metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},
    )
    adap_kl_ctrl: Optional[bool] = field(default=True, metadata={"help": "Use adaptive KL control"})
    load_in_8bit: Optional[bool] = field(default=True, metadata={"help": "whether to load the model in 8bit"})

# Parse arguments
parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

# PPO Configuration
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

# Load dataset
train_dataset = load_dataset(
    "lvwerra/stack-exchange-paired", data_dir="data/rl", split="train", verification_mode="no_checks"
)
train_dataset = train_dataset.select(range(100000))
original_columns = train_dataset.column_names

# Pipeline arguments
sent_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": 16,
    "truncation": True,
}

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name)
if getattr(tokenizer, "pad_token", None) is None:
    tokenizer.pad_token = tokenizer.eos_token

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

# Set seed
set_seed(config.seed)

# Get current device
accelerator = Accelerator()
current_device = accelerator.local_process_index

# Build the model (Actor) with LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    load_in_8bit=script_args.load_in_8bit,
    device_map={"": current_device},
    peft_config=lora_config,
)

# (Optional) Build optimizer
optimizer = None
if script_args.adafactor:
    optimizer = Adafactor(
        filter(lambda p: p.requires_grad, model.parameters()),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        lr=config.learning_rate,
    )

# Build PPO trainer
ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer,
)

# Build the Reward Model pipeline
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"

reward_model_name = script_args.reward_model_name
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=reward_model_name,
    device_map={"": current_device},
    model_kwargs={"load_in_8bit": script_args.load_in_8bit},
    tokenizer=tokenizer,
    return_token_type_ids=False,
)
if sentiment_pipe.model.config.pad_token_id is None:
    sentiment_pipe.model.config.pad_token_id = sentiment_pipe.model.config.eos_token_id

# Generation arguments
generation_kwargs = {
    "top_k": 1,            # must be an integer (not 1.0)
    "top_p": 0.95,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": 100_000,
}

output_min_length = 32
output_max_length = script_args.output_max_length
output_length_sampler = LengthSampler(output_min_length, output_max_length)

# ---------- GPU UTILIZATION HELPER ---------- #
def get_average_gpu_utilization_during(func, *args, **kwargs):
    """
    Runs 'func(*args, **kwargs)' in a thread while polling GPU utilization.
    Returns (average_gpu_percent, func_return_value).
    """
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    samples = []
    result_container = [None]

    def _wrapper():
        result_container[0] = func(*args, **kwargs)

    thread = threading.Thread(target=_wrapper)
    thread.start()

    while thread.is_alive():
        util_struct = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_util = util_struct.gpu  # in percent
        samples.append(gpu_util)
        time.sleep(0.05)  # Poll every 50ms

    thread.join()
    avg_util = (sum(samples) / len(samples)) if samples else 0.0
    return avg_util, result_container[0]

# Track GPU usage
actor_gpu_util = 0.0
reward_gpu_util = 0.0
critic_gpu_util = 0.0
other_gpu_util = 0.0

# ------- We do a single batch for demonstration ------- #
for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    if epoch >= 1:
        break

    question_tensors = batch["input_ids"]

    # 1) ACTOR STAGE
    def actor_stage():
        return ppo_trainer.generate(
            question_tensors,
            return_prompt=False,
            length_sampler=output_length_sampler,
            **generation_kwargs,
        )

    actor_gpu_util, response_tensors = get_average_gpu_utilization_during(actor_stage)

    # 2) REWARD STAGE
    def reward_stage():
        decoded_responses = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        texts = [q + r for q, r in zip(batch["query"], decoded_responses)]
        return sentiment_pipe(texts, **sent_kwargs)

    reward_gpu_util, pipe_outputs = get_average_gpu_utilization_during(reward_stage)

    # 3) CRITIC (VALUE) STAGE
    def critic_stage():
        # Convert pipeline outputs to numeric rewards
        rewards = [
            torch.tensor(o[0]["score"] - script_args.reward_baseline) for o in pipe_outputs
        ]
        stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
        return stats

    critic_gpu_util, stats = get_average_gpu_utilization_during(critic_stage)

    # 4) OTHER STAGE
    def other_stage():
        # example "other" step
        time.sleep(0.2)
    other_gpu_util, _ = get_average_gpu_utilization_during(other_stage)

    # Print or log GPU usage
    print(f"Actor GPU Util:  {actor_gpu_util:.2f}%")
    print(f"Reward GPU Util: {reward_gpu_util:.2f}%")
    print(f"Critic GPU Util: {critic_gpu_util:.2f}%")
    print(f"Other GPU Util:  {other_gpu_util:.2f}%")

    break  # exit after one batch

# --- CREATE THE BAR CHART --- #
components = ["Actor", "Reward", "Critic", "Other"]
utils = [actor_gpu_util, reward_gpu_util, critic_gpu_util, other_gpu_util]

plt.bar(components, utils)
plt.xlabel("RLHF Component")
plt.ylabel("Average GPU Utilization (%)")
plt.title("GPU Utilization by Stage (Single Batch)")
plt.tight_layout()
plt.savefig("gpu_util_bar_chart.png")
plt.show()

print("Done! Chart saved as gpu_util_bar_chart.png")




'''

  
accelerate launch --num_processes 1 Experiment.py \
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
    --steps 120



'''