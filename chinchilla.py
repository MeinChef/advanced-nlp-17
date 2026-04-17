from config import GPTConfiguration
import subprocess
import os


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
            "python",
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