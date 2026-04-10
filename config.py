import torch
import platform
from dataclasses import dataclass
from textwrap import dedent



if torch.cuda.is_available:
    device = "cuda"
elif torch.mps.is_available:
    device = "mps"
else:
    device = "cpu"

if platform.system() == "Linux" and torch.cuda.is_available():
    compile = True
else:
    compile = False

@dataclass
class GPTConfiguration:
    device: str = device
    compile: bool = compile
    n_layer: int = 6
    n_head: int = 6
    n_embed: int = 384

    def __str__(
            self
    ) -> str:
        longstr = f"""
            # Baseline configuration
            out_dir = 'out-shakespeare-baseline'
            eval_interval = 250
            eval_iters = 200
            log_interval = 10
            always_save_checkpoint = False
            wandb_log = False
            wandb_project = 'nanoGPT-assignment'
            wandb_run_name = 'baseline'
            dataset = 'shakespeare_char'
            gradient_accumulation_steps = 1
            batch_size = 64
            block_size = 256
            n_layer = {self.n_layer}
            n_head = {self.n_head}
            n_embd = {self.n_embed}
            dropout = 0.2
            learning_rate = 1e-3
            max_iters = 5000
            lr_decay_iters = 5000
            min_lr = 1e-4
            beta2 = 0.99
            warmup_iters = 100
            weight_decay = 1e-1
            device = {self.device} # change to 'cuda' if you have a GPU
            compile = {self.compile} # set True only on Linux with GPU
        """
        return dedent(longstr)
