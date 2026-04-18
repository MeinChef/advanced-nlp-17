from config import GPTConfiguration
import subprocess
import os
import sys


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
        check = True
    )


    with open(
        os.path.join(
            nanoGPTpath,
            f"out-shakespeare-{config1.name}",
            "samples"
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