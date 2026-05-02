from part_2_prepare_sft import prepare_training
from part_2_evaluation import evaluate_model
from config import GPTConfiguration
import shutil
import os
import subprocess
import sys
import torch
import time


if __name__ == "__main__":
    models = ['task1', 'task2', 'multi', 'char']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    nanopath = os.path.join(
        os.path.dirname(__file__),
        "nanoGPT"
    )

    # preparing the data
    for model in models:
        prepare_training(model)

    # move the custom training file to the nanoGPT folder
    shutil.copy(
        src = os.path.join(
            os.path.dirname(__file__),
            "sft",
            "train_sft.py"
        ),
        dst = os.path.join(
            os.path.dirname(__file__),
            "nanoGPT",
            "train_sft.py"
        )
    )

    # train the models
    # COMPUTATIONALLY EXPENSIVE
    for model in models:
        now = time.time()
        cfg = GPTConfiguration(
            n_layer = 5,
            n_head = 5,
            n_embed = 320,
            eval_iters = 50,
            eval_interval = 50, 
            max_iters = 7600 + 2500, # starts at the last checkpoint from previous
            save_checkpoints = False,

            init_from = "resume",
            lr = 1e-3 if model == "char" else 1e-4,
            dataset = f"shakespeare_{model}",
            name = model
        )
        cfg.set_backend()
        cfg.write(
            os.path.join(
                os.path.dirname(__file__), 
                "nanoGPT"
            )
        )

        current_outpath = os.path.join(
            os.path.dirname(__file__),
            "logs",
            f"out-shakespeare-{cfg.name}"
        )
        os.makedirs(
            current_outpath,
            exist_ok = True
        )

        # and copy the best model to that
        os.makedirs(
            os.path.join(
                nanopath,
                f"out-shakespeare-{cfg.name}"
            ),
            exist_ok = True
        )

        shutil.copy(
            src = os.path.join(
                os.path.dirname(__file__),
                "nanoGPT",
                "out-shakespeare-5-320-1",
                "ckpt.pt"
            ),
            dst = os.path.join(
                os.path.dirname(__file__),
                "nanoGPT",
                f"out-shakespeare-{cfg.name}",
                "ckpt.pt"
            )
        )

        # training
        with open(
            os.path.join(
                current_outpath,
                "train.out"
            ),
            "w+"
        ) as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "train_sft",
                    os.path.join(
                        os.path.dirname(__file__),
                        "nanoGPT",
                        "config",
                        f"train-shakespeare-char-{model}.py"
                    )
                ],
                cwd = os.path.join(
                    os.path.dirname(__file__), 
                    "nanoGPT"
                ),
                text = True,
                encoding = "utf-8",
                stdout = log,
                check = True
            )
        
        # and evaluate them
        evaluate_model(
            model = model,
            device = device
        )
        print(f"Done with model {model}! Took only like {time.time()-now}s")