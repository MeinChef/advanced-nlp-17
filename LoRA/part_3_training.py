import os

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

    os.system(
        f'cd nanoGPT && python finetune_lora.py config/training_{experiment}.py '
        f'| ts | tee logs/{experiment}.log'
    )
