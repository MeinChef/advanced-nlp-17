import os
import sys
import math
import time
import shutil
import subprocess
from argparse import ArgumentParser
from itertools import product
from config import GPTConfiguration
from sft.helper import prep_data_sub, use_subset
from sft.data_prep import prepare_training
from eval import evaluate_model

CONFIG_MAP = {
    "pre": (
        [
            {"layers": 2, "heads": 2, "embed": 64 * 2, "params":    119_168},  #    119,168
            {"layers": 4, "heads": 4, "embed": 64 * 4, "params":  3_230_208},  #  3,230,208
            {"layers": 5, "heads": 5, "embed": 64 * 5, "params":  6_250_240},  #  6,250,240
            {"layers": 6, "heads": 6, "embed": 64 * 6, "params": 10_745_088},  # 10,745,088
        ],
        [0.125, 0.25, 0.5, 1.0],
    ),

    "sft": ["task1", "task2", "multi"],

    "lora": [
        "task1-r1",
        "task1-r2",
        "task1-r4",
        "task1-r8",
        "task1-r16",
        "task2-r4",
        "multi-r4",
    ],
}

def main(task: str) -> None:
    todos = CONFIG_MAP[task]
    nanopath = os.path.join(
        os.path.dirname(__file__),
        "nanoGPT"
    )

    if task == "pre":
        for split in todos[1]:
            prep_data_sub(
                basepath = nanopath,
                split = split
            )
        todos = product(todos[0], todos[1])
    else:
        for x in CONFIG_MAP["sft"]:
            prepare_training(x)


    for model in todos:
        then = time.time()
        if task == "pre":
            # unpack
            model_config, data_subset = model
            # block size is context size
            # ergo the network "sees" batch_size x block_size tokens per batch.
            # which is 64*256 = 16384
            max_iters = math.ceil(model_config["params"] / 16384 * 20)
            
            cfg = GPTConfiguration(
                n_head = model_config["heads"],
                n_layer = model_config["layers"],
                n_embed = model_config["embed"],
                max_iters = max_iters,
                eval_iters = 5,
                eval_interval = 50,
                save_checkpoints = True,
                name = f"{model_config['layers']}-{model_config['embed']}-{data_subset}"
            )

            # and copy the subset to use
            use_subset(
                basepath = os.path.join(
                    nanopath,
                    "data",
                    "shakespeare_char"
                ),
                split = data_subset
            )

        else:
            # create configuration for the best model
            cfg = GPTConfiguration(
                n_layer = 5,
                n_head = 5,
                n_embed = 320,
                lr = 1e-3,

                eval_iters = 50,
                eval_interval = 50, 
                max_iters = 7600 + 2000, # starts at the last checkpoint from previous

                lora_rank = int(model[7:]) if len(model) > 5 else 0,  # everything after -r

                save_checkpoints = False,
                init_from = "resume",
                dataset = f"shakespeare_{model[:5]}",
                name = model                        # type: ignore
            )

            # and copy the model to resume folder
            os.makedirs(
                os.path.join(
                    nanopath,
                    f"out-shakespeare-{cfg.name}"
                ),
                exist_ok = True
            )

            source = os.path.join(
                nanopath,
                "out-shakespeare-5-320-1",
                "ckpt.pt"
            )
            if os.path.exists(source):
                shutil.copy(
                    src = source,
                    dst = os.path.join(
                        nanopath,
                        f"out-shakespeare-{cfg.name}",
                        "ckpt.pt"
                    )
                )
            else:
                raise FileNotFoundError(
                    f"Could not find a pre-trained model in {source}.\n" +
                    "Please run the script with --pre first or copy some model in above folder."
                )

        cfg.set_backend()
        cfg.set_compile()
        cfg.write(
            basepath = nanopath
        )

        # create logdir
        current_outpath = os.path.join(
            os.path.dirname(__file__),
            "logs",
            f"out-shakespeare-{cfg.name}"
        )
        os.makedirs(
            current_outpath,
            exist_ok = True,
        )

        # and run the training
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

        if task == "pre":
            # sampling (generating outputs)
            with open(
                os.path.join(
                    current_outpath,
                    "samples.out"
                ),
                "w+"
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
                    cwd = nanopath,
                    text = True,
                    encoding = "utf-8",
                    stdout = file,
                    check = True
                )
        else:
            evaluate_model(
                model = model[:5],                  # type: ignore
                device = cfg.device
            )

        print(f"Succesfully trained model {model}!")
        print(f"Wall Time: {round(time.time() - then, 4)}s")



def parse_args():
    parser = ArgumentParser(
        description="Training mode selector"
    )

    group = parser.add_mutually_exclusive_group(required = True)

    group.add_argument(
        "--pre-training", "--pre",
        dest="mode",
        action="store_const",
        const="pre",
        help="Run pre-training mode"
    )

    group.add_argument(
        "--fine-tune", "--sft",
        dest="mode",
        action="store_const",
        const="sft",
        help="Run fine-tuning mode"
    )

    group.add_argument(
        "--lora", "--lora",
        dest="mode",
        action="store_const",
        const="lora",
        help="Run LoRA mode"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args.mode)