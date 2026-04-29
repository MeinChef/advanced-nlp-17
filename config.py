import torch
import platform
from dataclasses import dataclass
from textwrap import dedent


@dataclass
class GPTConfiguration:
    n_layer: int = 6
    n_head: int = 6
    n_embed: int = 384
    lr: float = 1e-3
    device: str = "cpu"
    compile: bool = False
    name: str = "baseline"

    def __str__(
        self
    ) -> str:
        longstr = f"""
            # Baseline configuration
            out_dir = 'out-shakespeare-{self.name}'
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
            learning_rate = {self.lr}
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
    
    def set_backend(
        self
    ) -> None:
        if torch.cuda.is_available:
            self.device = "cuda"
        elif torch.mps.is_available:
            self.device = "mps"
        else:
            self.device = "cpu"

    def set_compile(
        self
    ) -> None:
        if platform.system() == "Linux" and self.device == "cuda":
            self.compile = True
        else:
            self.compile = False

    # TODO:
    #   - function write to file
    #   - filename in class