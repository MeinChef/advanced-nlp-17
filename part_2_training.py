import os
import re

def train(model: str):
    """
    Function to train a model
    Parameter:
        - model (str): choose which model to train
            'Task 1'
            'Task 2'
            'Multi-Task'
            'Pre-Trained'
    """

    if model == 'Task 1':
        os.system(f"cd {os.path.join(os.path.dirname(__file__), 'nanoGPT')} && python train.py config/training_shakespeare_task1.py | ts | tee {os.path.join(os.path.dirname(__file__), 'part_2_logs/shakespeare_task1.log')}")
    elif model == 'Task 2':
        os.system(f"cd {os.path.join(os.path.dirname(__file__), 'nanoGPT')} && python train.py config/training_shakespeare_task2.py | ts | tee {os.path.join(os.path.dirname(__file__), 'part_2_logs/shakespeare_task2.log')}")
    elif model == 'Multi-Task':
        os.system(f"cd {os.path.join(os.path.dirname(__file__), 'nanoGPT')} && python train.py config/training_shakespeare_multitask.py | ts | tee {os.path.join(os.path.dirname(__file__), 'part_2_logs/shakespeare_multitask.log')}")
    elif model == 'Pre-Trained':
        os.system(f"cd {os.path.join(os.path.dirname(__file__), 'nanoGPT')} && python train.py config/training_shakespeare_char.py | ts | tee {os.path.join(os.path.dirname(__file__), 'part_2_logs/shakespeare_char.log')}")
    else:
        print('not a supported model for training')