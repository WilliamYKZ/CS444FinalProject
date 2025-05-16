import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --------------------------------------------------------------------
# 1) Replace these paths with your own paths
# --------------------------------------------------------------------
BASE_MODEL_PATH = "/home/exouser/Desktop/0_ModelCheckpoint/Full_RL_Model/CheckPoint_500"  # original, un-fine-tuned llama
MERGED_MODEL_PATH = "/home/exouser/Desktop/0_ModelCheckpoint/Full_RL_Model/CheckPoint_1200"  # newly merged SFT weights

# --------------------------------------------------------------------
# 2) Load both models and tokenizers
# --------------------------------------------------------------------
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, device_map="auto")
base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

print("Loading merged SFT model...")
merged_model = AutoModelForCausalLM.from_pretrained(MERGED_MODEL_PATH, device_map="auto")
merged_tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH)


# --------------------------------------------------------------------
# 4) Short generation test
# --------------------------------------------------------------------
prompt = "Explain why reinforcement learning is important in AI."

print("\n----- Generating with CheckPoint_500 model -----")
inputs_base = base_tokenizer(prompt, return_tensors="pt").to(base_model.device)
with torch.no_grad():
    gen_tokens_base = base_model.generate(
        **inputs_base,
        max_new_tokens=1024,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )
output_base = base_tokenizer.decode(gen_tokens_base[0], skip_special_tokens=True)
print(output_base)

print("\n----- Generating with CheckPoint_1200 model -----")
inputs_merged = merged_tokenizer(prompt, return_tensors="pt").to(merged_model.device)
with torch.no_grad():
    gen_tokens_merged = merged_model.generate(
        **inputs_merged,
        max_new_tokens=1024,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )
output_merged = merged_tokenizer.decode(gen_tokens_merged[0], skip_special_tokens=True)
print(output_merged)

print("\nDone. Compare the outputs above to see differences between base vs. SFT.\n")
