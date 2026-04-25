from config import GPTConfiguration
from itertools import product
import subprocess
import os
import sys
import shutil
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

    # check if subset already exists
    if os.path.exists(
        os.path.join(
            base_path,
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
            base_path,
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
            base_path,
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
    for split in data_subsets:
        prep_data_subs(split = split)

    for model_config, data_subset in product(model_configs, data_subsets):
        
        # TODO: make max_iters depend on dataset size, such that model sees fixed number of tokens for each ds
        # as per my understanding, we just have to keep that constant across all models
        # should we take model size in account here?

        cfg = GPTConfiguration(
            n_head = model_config["heads"],
            n_layer = model_config["layers"],
            n_embed = model_config["embed"],
            max_iters = 5,
            eval_iters = 20,
            eval_interval = 5,
            save_checkpoints = True,
            name = f"{model_config['layers']}-{model_config['embed']}-{data_subset}"
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

        # move the split we want to use to train.bin
        use_subset(
            basepath = os.path.join(
                nanoGPTpath,
                "data",
                "shakespeare_char"
            ),
            split = data_subset
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