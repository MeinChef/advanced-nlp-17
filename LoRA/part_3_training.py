import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

def train(experiment: str):
    """
    Function to run a LoRA fine-tuning experiment.
    Parameter:
        - experiment (str): choose which experiment to run
            'exp4_taskA_rank4'
            'exp4_taskB_rank4'
            'exp5_taskA_rank1'
            'exp5_taskA_rank2'
            'exp5_taskA_rank4'
            'exp5_taskA_rank8'
            'exp5_taskA_rank16'
            'exp6_fullFT'
            'exp6_lora_rank4'
            'exp6_lora_multitask'
    """

    valid_experiments = [
        'exp4_taskA_rank4',
        'exp4_taskB_rank4',
        'exp5_taskA_rank1',
        'exp5_taskA_rank2',
        'exp5_taskA_rank4',
        'exp5_taskA_rank8',
        'exp5_taskA_rank16',
        'exp6_fullFT',
        'exp6_lora_rank4',
        'exp6_lora_multitask',
    ]

    if experiment not in valid_experiments:
        print(f"'{experiment}' is not a supported experiment. Choose from: {valid_experiments}")
        return

    rootpth = Path(os.path.dirname(__file__),).parent

    # Ensure logs directory exists
    os.makedirs(
        os.path.join(
            rootpth,
            "logs"
        ), 
        exist_ok = True
    )
    os.makedirs(
        os.path.join(
            rootpth,
            "logs",
            f"out-lora-{experiment}"
        )
    )

    # Get the LoRA directory (where this script is)
    lora_dir = os.path.dirname(os.path.abspath(__file__))

    
    # Run training with timestamp logging (Windows-compatible)
    log_file = os.path.join(
        rootpth,
        "logs",
        f"out-lora-{experiment}",
        f"{experiment}.out"
    )
    try:
        with open(log_file, 'w+') as log:
            process = subprocess.Popen(
                [sys.executable, 'train_lora.py', f'LoRA/config/training_{experiment}.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                cwd=lora_dir  # Run from LoRA directory
            )
            
            # Stream output with timestamps
            for line in process.stdout:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                timestamped_line = f'[{timestamp}] {line}'
                print(timestamped_line, end='')
                log.write(timestamped_line)
                log.flush()
            
            process.wait()
            if process.returncode != 0:
                print(f"Training failed with return code {process.returncode}", file=sys.stderr)
    except Exception as e:
        print(f"Error running training: {e}", file=sys.stderr)
