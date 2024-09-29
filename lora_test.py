# # Install Unsloth (if not already installed)
# !pip install unsloth
# # Upgrade to the latest nightly version
# !pip uninstall unsloth -y && pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Import necessary libraries
import torch
import torch.nn as nn
from unsloth import FastLanguageModel
import json

# torch.set_default_device('cpu')


# Step 1: Initialize the FastLanguageModel and Tokenizer
model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"  # Choose a supported 4-bit model
max_seq_length = 2048
dtype = None  # Auto-detection
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Step 2: Add LoRA Adapters
lora_config = {
    "r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

model = FastLanguageModel.get_peft_model(
    model,
    **lora_config,
)

# Optional: Freeze all model parameters except LoRA adapters
for name, param in model.named_parameters():
    if any(target in name for target in lora_config["target_modules"]):
        param.requires_grad = True
    else:
        param.requires_grad = False

# Step 3: Set Up the Optimizer
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# Step 4: Create Dummy Inputs
dummy_prompt = "How's the weather today?"
encoded_input = tokenizer.encode_plus(
    dummy_prompt,
    return_tensors="pt",
    max_length=max_seq_length,
    truncation=True,
    padding="max_length",
)

input_ids = encoded_input["input_ids"]  # Shape: (1, max_seq_length)
attention_mask = encoded_input["attention_mask"]  # Shape: (1, max_seq_length)

# Move inputs to the same device as the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)

# Step 5: Perform a Forward Pass
outputs = model(input_ids=input_ids, attention_mask=attention_mask)
# Assuming the model returns logits
logits = outputs.logits  # Shape: (1, max_seq_length, vocab_size)

# Step 6: Compute a Dummy Loss (e.g., Mean of Logits)
dummy_loss = logits.mean()

# Step 7: Backward Pass and Optimizer Step
optimizer.zero_grad()
dummy_loss.backward()
optimizer.step()

# Step 8: Verify Parameter Updates
# For demonstration, print the first few gradients of LoRA parameters
print("Sample gradients from LoRA adapters:")
for name, param in model.named_parameters():
    if param.requires_grad and param.grad is not None:
        print(f"{name}: {param.grad.abs().mean().item():.6f}")
        break  # Print only the first gradient for brevity

print("Feasibility Test Completed Successfully!")
