import subprocess
import os
import sys

def in_venv() -> bool:
    # Prüfen, ob das Skript in einer virtuellen Umgebung läuft
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

if __name__ == "__main__":

    if in_venv():
        print("Detected virtual environment:", sys.prefix)
    else:
        print("No virtual environment detected. Aborting...")
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

    if not os.path.exists(
        os.path.join(
            os.path.dirname(__file__),
            "nanoGPT",
            "data",
            "shakespeare_char",
            "train.bin"
        )
    ):
        print("Generating train and test data...")
        import nanoGPT.data.shakespeare_char.prepare
        print("Done!")

    else:
        print("Train and test data already prepared, skipping...")

    print("Setup complete!")