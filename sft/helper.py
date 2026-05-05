import os
import shutil
import numpy as np
import torch
from peft import get_peft_model, LoraConfig


def prep_data_sub(
    basepath: str = "nanoGPT",
    split: float = 0.5,
) -> None:
    if split < 0 or split > 1:
        raise ValueError(
            "Split is outside of range 0-1. Please fix that.\n" +
            f"Actual Value: {split}"
        )

    # check if subset already exists
    if os.path.exists(
        os.path.join(
            basepath,
            "data",
            "shakespeare_char",
            f"train-{int(split * 100)}.bin"
        )
    ):
        print(f"A split of {split} already exists, not creating another one.")
        return

    # load
    data = np.memmap(
        os.path.join(
            basepath,
            "data",
            "shakespeare_char",
            "train-orig.bin"
        ), 
        dtype = np.uint16, 
        mode = "r"
    )

    # subset and save
    data_subset = data[:int(len(data) * split)]
    data_subset.tofile(
        os.path.join(
            basepath,
            "data",
            "shakespeare_char",
            f"train-{int(split * 100)}.bin"
        )
    )


def use_subset(
    basepath: str,
    split: float | int = 0.5
) -> None:
    
    # check if split is in a valid range
    if split > 0 and split <= 1:
        id = int(split * 100)
    elif split > 0 and split <= 100 and isinstance(split, int):
        id = split
    else:
        raise ValueError("Split is not in range ]0,1] or ]0,100] (and int).")

    if os.path.exists(
        os.path.join(basepath, "train.bin")
    ):
        # remove the train.bin first
        os.remove(
            os.path.join(basepath, "train.bin")
        )

    # aaaand copy
    shutil.copyfile(
        src = os.path.join(
            basepath,
            f"train-{id}.bin"
        ),
        dst = os.path.join(
            basepath,
            "train.bin"
        )
    )


SPECIAL_TOKENS = ['@', '|', '<', '>']
NUM_SPECIAL_TOKENS = len(SPECIAL_TOKENS)  # 4

def inject_lora(
    model, 
    lora_rank: int = 4
):
    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        target_modules=["c_attn"],
        lora_dropout=0.0,
        bias="none",
    )
    return get_peft_model(model, config)


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def resize_token_embeddings(
    model, 
    num_new_tokens: int
    ):
    """Extend token embeddings and the tied lm_head weight by *num_new_tokens*.

    Both ``model.transformer.wte`` (the input embedding) and
    ``model.lm_head`` (the output projection that shares its weight with wte
    in standard GPT-2) are resized in-place.  New rows are initialised with
    the mean of the existing embedding rows so that the new tokens start in a
    plausible region of weight space.

    Args:
        model: A ``GPT`` instance (unwrapped from DDP if applicable).
        num_new_tokens: Number of additional vocabulary entries to add.
    """
    if num_new_tokens <= 0:
        return model, model.config.vocab_size

    old_vocab_size = model.config.vocab_size
    new_vocab_size = old_vocab_size + num_new_tokens

    # --- input embedding (wte) -------------------------------------------
    old_wte = model.transformer.wte  # nn.Embedding
    old_weight = old_wte.weight.data  # (old_vocab, n_embd)

    new_wte = torch.nn.Embedding(new_vocab_size, old_wte.embedding_dim)
    new_wte.to(device=old_weight.device, dtype=old_weight.dtype)

    # copy existing weights
    new_wte.weight.data[:old_vocab_size] = old_weight
    # initialise new rows with the mean of existing embeddings
    mean_embedding = old_weight.mean(dim=0, keepdim=True)  # (1, n_embd)
    new_wte.weight.data[old_vocab_size:] = mean_embedding.expand(num_new_tokens, -1)

    model.transformer.wte = new_wte

    # --- output projection (lm_head) -------------------------------------
    # In nanoGPT lm_head is an nn.Linear(n_embd, vocab_size, bias=False)
    # whose .weight is *tied* to wte.weight at forward time.  We replace the
    # Linear layer and re-tie the weights so the parameter is shared exactly
    # as in the original architecture.
    old_lm_head = model.lm_head  # nn.Linear
    new_lm_head = torch.nn.Linear(old_lm_head.in_features, new_vocab_size, bias=False)
    new_lm_head.to(device=old_lm_head.weight.device, dtype=old_lm_head.weight.dtype)

    new_lm_head.weight = model.transformer.wte.weight  # tie weights

    model.lm_head = new_lm_head

    # --- update config so checkpoints reflect the new vocab size ----------
    model.config.vocab_size = new_vocab_size

    print(f"resize_token_embeddings: vocab {old_vocab_size} -> {new_vocab_size} "
          f"(added {num_new_tokens} special token(s): {SPECIAL_TOKENS})")
    
    return model, new_vocab_size