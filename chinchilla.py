from config import GPTConfiguration
from itertools import product
import subprocess
import os
import sys
import numpy as np

def prep_data_subs(
    base_path: str = "nanoGPT",
    split: float = 0.5,
) -> None:
    if split < 0 or split > 1:
        raise ValueError(
            "Split is outside of range 0-1. Please fix that.\n" +
            f"Actual Value: {split}"
        )

    # TODO: check if subset already exists

    # load
    data = np.memmap(
        os.path.join(
            base_path,
            "data",
            "train.bin"
        ), 
        dtype = np.uint16, 
        mode = "r"
    )

    # subset and save
    data_subset = data[:int(len(data) * split)]
    data_subset.tofile(
        os.path.join(
            base_path,
            "data",
            f"train-{int(split * 100)}.bin"
        )
    )



if __name__ == "__main__":
    nanoGPTpath = os.path.join(
        os.path.dirname(__file__),
        "nanoGPT"
    )

    model_configs = [
        {"layers": 2, "heads": 2, "embed": 64},
        {"layers": 4, "heads": 4, "embed": 128},
        {"layers": 5, "heads": 5, "embed": 256},
        {"layers": 6, "heads": 6, "embed": 512},
    ]

    data_subsets = [0.125, 0.25, 0.5, 1]
    # TODO: create subsets, delete train.bin, and copy the needed subset to train.bin

    for model_config, data_subset in product(model_configs, data_subsets):
        
        cfg = GPTConfiguration(
            n_head = model_config["heads"],
            n_layer = model_config["layers"],
            n_embed = model_config["embed"],
            max_iters = 5,                        # TODO: make this depend on dataset size, such that model sees fixed number of tokens for each ds
            eval_iters = 20,
            eval_interval = 5,
            save_checkpoints = True,
            name = f"{model_config["layers"]}-{model_config["embed"]}-{data_subset}"
        )

        # this overrides the values you passed earlier, if you passed any
        # config1.set_backend()
        # config1.set_compile()


        cfg.write(
            basepath = nanoGPTpath
        )

        current_outpath = os.path.join(
            os.path.dirname(__file__),
            f"out-shakespeare-{cfg.name}"
        )
        
        os.makedirs(
            current_outpath,
            exist_ok = True,
        )



        # TODO: check if the log files exist and are complete (or try loading checkpoint to resume from)
        #       and then resume training instead of re-doing the whole thing

        # training
        with open(
            os.path.join(
                current_outpath,
                "train.log"
            ),
            "w+"
        ) as log:
            
            # the GPT-init process will report paramters, but rounded and without embedding.
            # to change that please go into nanoGPT/model.py and change line 148 to
            # print(f"number of parameters: {self.get_num_params(non_embedding = False)}")
            # that will report the exact number of parameters to stdout / train.log

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "train",
                    os.path.join(
                        nanoGPTpath,
                        "config",
                        f"train-shakespeare-char-{cfg.name}.py"
                    )
                ],
                cwd = nanoGPTpath,
                text = True,
                encoding = "utf-8",
                stdout = log,
                check = True
            )

        # sampling (generating outputs)
        with open(
            os.path.join(
                current_outpath,
                "samples.log"
            ),
            "w+"                                        # creates file if it doesn't exist
        ) as file:
            
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "sample",
                    f"--out_dir=out-shakespeare-{cfg.name}",
                    f"--num_samples={cfg.eval_samples}",
                    f"--device={cfg.device}"
                ],
                cwd = nanoGPTpath,
                text = True,
                encoding = "utf-8",
                stdout = file,
                check = True
            )