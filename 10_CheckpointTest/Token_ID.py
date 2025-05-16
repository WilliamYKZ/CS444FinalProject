import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --------------------------------------------------------------------
# 1) Replace these paths with your own paths
# --------------------------------------------------------------------
BASE_MODEL_PATH = "/home/exouser/Desktop/0_ModelCheckpoint/SFT_Model_Checkpoint/Checkpoint-5000"  # original, un-fine-tuned llama
MERGED_MODEL_PATH = "/home/exouser/Desktop/0_ModelCheckpoint/RL_Model_Checkpoint/Checkpoint_200"  # newly merged SFT weights

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

print("\n----- Generating with base model -----")
inputs_base = base_tokenizer(prompt, return_tensors="pt").to(base_model.device)

# Print the input token IDs for the base model
print("Base model input token IDs:", inputs_base["input_ids"].tolist())

with torch.no_grad():
    gen_tokens_base = base_model.generate(
        **inputs_base,
        max_new_tokens=50,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )

# Print the generated token IDs for the base model
print("Base model generated token IDs:", gen_tokens_base[0].tolist())

output_base = base_tokenizer.decode(gen_tokens_base[0], skip_special_tokens=True)
print("Base model output text:")
print(output_base)

print("\n----- Generating with merged SFT model -----")
inputs_merged = merged_tokenizer(prompt, return_tensors="pt").to(merged_model.device)

# Print the input token IDs for the merged model
print("Merged SFT model input token IDs:", inputs_merged["input_ids"].tolist())

with torch.no_grad():
    gen_tokens_merged = merged_model.generate(
        **inputs_merged,
        max_new_tokens=50,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )

# Print the generated token IDs for the merged SFT model
print("Merged SFT model generated token IDs:", gen_tokens_merged[0].tolist())

output_merged = merged_tokenizer.decode(gen_tokens_merged[0], skip_special_tokens=True)
print("Merged SFT model output text:")
print(output_merged)

print("\nDone. Compare the outputs (and token IDs) above to see differences between base vs. SFT.\n")
