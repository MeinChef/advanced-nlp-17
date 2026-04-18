from config import GPTConfiguration
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

    config1 = GPTConfiguration(
        n_head = 2,
        n_layer = 2,
        n_embed = 128,
        max_iters = 100,
        eval_iters = 20,
        eval_interval = 50,
        save_checkpoints = True,
        name = "trials"
    )

    config1.write(
        basepath = nanoGPTpath
    )

    # this overrides the values you passed earlier, if you passed any
    # config1.set_backend()
    # config1.set_compile()

    current_outpath = os.path.join(
        os.path.dirname(__file__),
        f"out-shakespeare-{config1.name}"
    )
    
    os.makedirs(
        current_outpath,
        exist_ok = True,
    )

    # training
    with open(
        os.path.join(
            current_outpath,
            "train.log"
        ),
        "w+"
    ) as log:
        
        subprocess.run(
            [
                sys.executable,
                "-m",
                "train",
                os.path.join(
                    nanoGPTpath,
                    "config",
                    f"train-shakespeare-char-{config1.name}.py"
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
                f"--out_dir=out-shakespeare-{config1.name}",
                f"--num_samples={config1.eval_samples}",
                f"--device={config1.device}"
            ],
            cwd = nanoGPTpath,
            text = True,
            encoding = "utf-8",
            stdout = file,
            check = True
        )