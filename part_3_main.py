import os
import sys
import time
import shutil
import subprocess
from config import GPTConfiguration
from part_2_evaluation import evaluate_model

if __name__ == "__main__":
    experiments = [
        "task1-r1",
        "task1-r2",
        "task1-r4",
        "task1-r8",
        "task1-r16",
        "task2-r4",
        "multi-r4"
    ]

    nanopath = os.path.join(
        os.path.dirname(__file__),
        "nanoGPT"
    )

    # move the custom training file to the nanoGPT folder
    shutil.copy(
        src = os.path.join(
            os.path.dirname(__file__),
            "sft",
            "helper.py"
        ),
        dst = os.path.join(
            nanopath,
            "helper.py"
        )
    )
    shutil.copy(
        src = os.path.join(
            os.path.dirname(__file__),
            "sft",
            "train_sft.py"
        ),
        dst = os.path.join(
            nanopath,
            "train_sft.py"
        )
    )

    # experiments woop woop
    for exp in experiments:
        now = time.time()

        # configuration
        cfg = GPTConfiguration(
            max_iters = 7600 + 2000,
            lora_rank = 0, # int(exp[7:]),           # everything after -r
            lr = 1e-4,
            save_checkpoints = False,
            dataset = f"shakespeare_{exp[:5]}", # tasknames are all 5 long
            init_from = "resume",
            name = exp
        )
        cfg.set_backend()
        cfg.set_compile()
        cfg.write(
            basepath = nanopath
        )

        # create log dir
        current_outpath = os.path.join(
            os.path.dirname(__file__),
            "logs",
            f"out-shakespeare-{cfg.name}"
        )
        os.makedirs(
            current_outpath,
            exist_ok = True
        )

        # and copy the best model to resume folder
        os.makedirs(
            os.path.join(
                nanopath,
                f"out-shakespeare-{cfg.name}"
            ),
            exist_ok = True
        )
        shutil.copy(
            src = os.path.join(
                nanopath,
                "out-shakespeare-5-320-1",
                "ckpt.pt"
            ),
            dst = os.path.join(
                nanopath,
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
                        nanopath,
                        "config",
                        f"train-shakespeare-char-{cfg.name}.py"
                    )
                ],
                cwd = nanopath,
                text = True,
                encoding = "utf-8",
                stdout = log,
                check = True
            )

        evaluate_model(
            model = exp[:5],
            device = cfg.device
        )
        print(f"Done with model {exp}! Took only like {time.time()-now}s")