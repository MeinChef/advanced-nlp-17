import re
import os

def prepare_configs(device_type: str, has_logs: bool = True):
    """
    Prepare config files for LoRA fine-tuning experiments (Part 3).
    Parameters:
        - device_type (str): 'cuda' for Linux GPU, otherwise 'cpu'
        - has_logs (bool): if False, creates fresh log files
    """

    # Base config for LoRA fine-tuning (mirrors finetune_lora.py defaults)
    file_content_standard = (
        "# LoRA fine-tuning base configuration\n"
        "out_dir = 'out/lora/base'\n"
        "eval_interval = 200\n"
        "eval_iters = 200\n"
        "log_interval = 10\n"
        "always_save_checkpoint = False\n"
        "wandb_log = False\n"
        "wandb_project = 'nanoGPT-lora'\n"
        "wandb_run_name = 'lora'\n"
        "dataset = 'shakespeare_char'\n"
        "gradient_accumulation_steps = 1\n"
        "batch_size = 32\n"
        "block_size = 256\n"
        "n_layer = 6\n"
        "n_head = 6\n"
        "n_embd = 384\n"
        "dropout = 0.1\n"
        "learning_rate = 1e-4\n"
        "max_iters = 2000\n"
        "lr_decay_iters = 2000\n"
        "min_lr = 1e-5\n"
        "beta2 = 0.99\n"
        "warmup_iters = 100\n"
        "weight_decay = 1e-1\n"
        "lora_rank = 4\n"
        "init_from = 'resume'\n"
        "device = 'cpu'\n"
        "compile = False"
    )

    file_content_linux_gpu = file_content_standard\
        .replace("device = 'cpu'", "device = 'cuda'")\
        .replace("compile = False", "compile = True")

    file_content = file_content_linux_gpu if device_type == 'cuda' else file_content_standard

    # Each experiment: (name, dataset, lora_rank)
    experiments_variants = [
        # Experiment 4: single-task LoRA
        ('exp4_taskA_rank4',      'task_a',  4),
        ('exp4_taskB_rank4',      'task_b',  4),
        # Experiment 5: rank ablation on Task A
        ('exp5_taskA_rank1',      'task_a',  1),
        ('exp5_taskA_rank2',      'task_a',  2),
        #('exp5_taskA_rank4',      'task_a',  4),
        ('exp5_taskA_rank8',      'task_a',  8),
        ('exp5_taskA_rank16',     'task_a', 16),
        # Experiment 6: LoRA vs full fine-tuning
        #('exp6_fullFT',           'task_a',  0),   # lora_rank=0 = full fine-tuning
        #('exp6_lora_rank4',       'task_a',  4),
        ('exp6_lora_multitask',   'task_ab', 4),
    ]

    os.makedirs('config', exist_ok=True) 

    for name, dataset, lora_rank in experiments_variants:
        new_content = re.sub(r'(dataset = ).*',   rf"\g<1>'{dataset}'",        file_content)
        new_content = re.sub(r'(lora_rank = ).*', rf'\g<1>{lora_rank}',        new_content)
        new_content = re.sub(r'(out_dir = )\S+',  rf"\g<1>'out/lora/{name}'",  new_content)
        new_content = re.sub(r'(wandb_run_name = ).*', rf"\g<1>'{name}'",      new_content)

        with open(f'config/training_{name}.py', 'w') as f:
            f.write(new_content)

    if not has_logs:
        try:
            os.makedirs('logs', exist_ok=True)
        except:
            print('folder either already exists or could not be created')
        finally:
            for name, _, _ in experiments_variants:
                with open(f'logs/{name}.log', 'w+') as f:
                    pass