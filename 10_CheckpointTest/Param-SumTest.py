import torch
from transformers import AutoModelForCausalLM

base_path = "/home/exouser/Desktop/0_ModelCheckpoint/Full_RL_Model/CheckPoint_500"  # original base
merged_path = "/home/exouser/Desktop/0_ModelCheckpoint/Full_RL_Model/CheckPoint_1200"  # your newly merged folder

model_base = AutoModelForCausalLM.from_pretrained(base_path)
model_merged = AutoModelForCausalLM.from_pretrained(merged_path)

sum_base = 0
sum_merged = 0

for p1, p2 in zip(model_base.parameters(), model_merged.parameters()):
    sum_base += p1.data.float().sum().item()
    sum_merged += p2.data.float().sum().item()

print("Base model param sum:", sum_base)
print("Merged SFT param sum:", sum_merged)
