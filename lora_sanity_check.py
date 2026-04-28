import torch
from model import GPT, GPTConfig, freeze_base_params, count_trainable

# Load the same checkpoint for both models
ckpt = torch.load('nanoGPT/out-shakespeare-5-320-1/ckpt.pt', map_location='cpu')

# Load with LoRA
config_lora = GPTConfig(**{**ckpt['model_args'], 'lora_rank': 4})
model_lora = GPT(config_lora)
model_lora.load_state_dict(ckpt['model'], strict=False)
freeze_base_params(model_lora)

# Check 1: only LoRA params are trainable
print("=== Trainable parameters ===")
for name, param in model_lora.named_parameters():
    if param.requires_grad:
        print(f"  {name}: {param.shape}")
print(f"Total trainable: {count_trainable(model_lora)}")

# Check 2: output is identical to model without LoRA (B=0 at init)
config_base = GPTConfig(**{**ckpt['model_args'], 'lora_rank': 0})
model_base = GPT(config_base)
model_base.load_state_dict(ckpt['model'], strict=True)

x = torch.randint(0, config_lora.vocab_size, (1, 32))
with torch.no_grad():
    out_lora, _ = model_lora(x)
    out_base, _ = model_base(x)

print("\n=== Output identity check ===")
print(f"Max diff: {(out_lora - out_base).abs().max().item()}")  # should be ~0.0
