from model import GPT, GPTConfig, inject_lora, count_trainable
import torch
<<<<<<< HEAD:lora/lora_sanity_check.py
from model_lora import GPT, GPTConfig, freeze_base_params, count_trainable
=======
>>>>>>> lora:LoRA/lora_sanity_check.py

# Basismodell
model_base = GPT(GPTConfig())
model_base.eval()

# LoRA-Modell
model_lora = GPT(GPTConfig())
model_lora = inject_lora(model_lora, lora_rank=4)
model_lora.load_state_dict(model_base.state_dict(), strict=False)
model_lora.eval()

# Check 1: only LoRA params are trainable
print("=== Trainable parameters ===")
for name, param in model_lora.named_parameters():
    if param.requires_grad:
        print(f"  {name}: {param.shape}")
print(f"Total trainable: {count_trainable(model_lora)}")

# Check 2: output identical to base model (B initialised to zero)
x = torch.randint(0, GPTConfig().vocab_size, (1, 16))

with torch.no_grad():
    out_base, _ = model_base(x)
    out_lora, _ = model_lora(x)

max_difference = (out_base - out_lora).abs().max().item()
print(f"\n=== Output difference ===")
print(f"Max difference: {max_difference:.2e}")