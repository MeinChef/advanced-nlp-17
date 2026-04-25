import subprocess
import os
import sys
import re
import shutil

def ask_user_for_confirmation(prompt: str) -> bool:
    while True:
        user_input = input(prompt).strip().lower()
        if user_input == "y":
            return True
        elif user_input == "n":
            return False
        else:
            print("Invalid input. please use only 'y' or 'n'.")

def in_conda() -> bool:
    return bool(
            os.environ.get("CONDA_DEFAULT_ENV") or 
            os.environ.get("CONDA_PREFIX")
        )   

def in_venv() -> bool:
    # Prüfen, ob das Skript in einer virtuellen Umgebung läuft
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

if __name__ == "__main__":

    if in_conda():
        print("Detected conda environment:", os.environ.get("CONDA_DEFAULT_ENV"))
    elif in_venv():
        print("Detected virtual environment:", sys.prefix)
    else:
        print("No conda environment detected. Aborting...")
        raise EnvironmentError()
    print("Starting Setup...")
    if not os.path.exists(
        os.path.join(
            os.path.dirname(__file__),
            "nanoGPT"
        )
    ):
        print("Cloning the nanoGPT repository...")
        subprocess.check_call(
            [
                "git",
                "clone",
                "https://github.com/karpathy/nanoGPT.git"
            ]
        )

        print("Done!")

    else:
        print("NanoGPT repository already cloned or folder exists. Skipping...")

    print("Installing dependencies...")
    if ask_user_for_confirmation("Blackwell GPU? (y/n): "):
        subprocess.check_call(
            [
                sys.executable, 
                "-m", 
                "pip", 
                "install", 
                "-r", 
                os.path.join(
                    os.path.dirname(__file__), 
                    "requirements_blackwell.txt"
                )
            ]
        )
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "torch",
                "torchvision",
                "torchaudio",
                "--index-url",
                "https://download.pytorch.org/whl/cu129",
            ]
        )
    else:
        subprocess.check_call(
            [
                sys.executable, 
                "-m", 
                "pip", 
                "install", 
                "-r", 
                os.path.join(
                    os.path.dirname(__file__), 
                    "requirements.txt"
                )
            ]
        )
    print("Done!")

    datapath =  os.path.join(
        os.path.dirname(__file__),
        "nanoGPT",
        "data",
        "shakespeare_char"
    )

    if not os.path.exists(
        os.path.join(
            datapath,
            "train.bin"
        )
    ):
        print("Generating train and test data...")
        import nanoGPT.data.shakespeare_char.prepare
        shutil.copyfile(
            src = os.path.join(
                datapath,
                "train.bin"
            ),
            dst = os.path.join(
                datapath,
                "train-orig.bin"
            )
        )
        print("Done!")

    else:
        print("Train and test data already prepared, skipping...")

    print("Editing nanoGPT/model.py to report exact model parameters...")
    if os.path.exists(
        os.path.join(
            os.path.dirname(__file__),
            "nanoGPT",
            "model.py"
        )
    ):
        filepth = os.path.join(
            os.path.dirname(__file__),
            "nanoGPT",
            "model.py"
        )

        replacement = "        print(\"number of parameters:\", self.get_num_params(non_embedding=False))"


        with open(filepth, "r") as file:
            data = file.readlines()
        
        if "number of parameters:" in data[147]:
            data[147] = replacement
        else: 
            for n, line in enumerate(data):
                if re.search("number of parameters:", line):
                    data[n] = replacement

        with open(filepth, "w") as file:
            file.writelines(data)

        print("Succesfully edited model.py!")
    else:
        print("Could not find model.py. Skipping ...")

    print("Setup complete!")