import torch
import platform
from dataclasses import dataclass
from textwrap import dedent
import os


@dataclass
class GPTConfiguration:
    n_layer: int = 6
    n_head: int = 6
    n_embed: int = 384
    lr: float = 1e-3
    dataset: str = "shakespeare_char"
    eval_iters: int = 200
    eval_interval: int = 250
    eval_samples: int = 5
    batch_size: int = 64
    max_iters: int = 5000
    warmup_iters: int = 200
    save_checkpoints: bool = False
    init_from: str = "scratch"
    device: str = "cpu"
    compile: bool = False
    name: str = "baseline"

    def __str__(
        self
    ) -> str:
        longstr = f"""
            # Baseline configuration
            out_dir = 'out-shakespeare-{self.name}'
            eval_interval = {self.eval_interval}
            eval_iters = {self.eval_iters}
            log_interval = 10
            always_save_checkpoint = {self.save_checkpoints}
            init_from = '{self.init_from}'
            wandb_log = False
            wandb_project = 'nanoGPT-assignment'
            wandb_run_name = 'baseline'
            dataset = '{self.dataset}'
            gradient_accumulation_steps = 1
            batch_size = {self.batch_size}
            block_size = 256
            n_layer = {self.n_layer}
            n_head = {self.n_head}
            n_embd = {self.n_embed}
            dropout = 0.2
            learning_rate = 1e-3
            max_iters = {self.max_iters}
            lr_decay_iters = 100
            min_lr = 1e-4
            beta2 = 0.99
            warmup_iters = {self.warmup_iters}
            weight_decay = 1e-1
            device = '{self.device}'
            compile = {self.compile} 
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

    def write(
        self,
        basepath: str = os.path.dirname(__file__)
    ) -> None:
        fullpath = os.path.join( 
            basepath,
            "config",
            f"train-shakespeare-char-{self.name}.py"
        )

        with open(fullpath, "w") as f:
            f.write(str(self))

    # TODO:
    #   - function write to file
    #   - filename in class