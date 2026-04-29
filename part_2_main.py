import part_2_prepare_sft
# import part_2_train_config # replace with GPTConfiguration
import part_2_training
import part_2_evaluation
from config import GPTConfiguration
import shutil
import os
import torch


if __name__ == "__main__":
    models = ['task1', 'task2', 'multi', 'pre']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # preparing the data
    for model in models:
        part_2_prepare_sft.prepare_training(model)

    # TODO: create logs folder
    # prepare the training files
    # part_2_train_config.prepare_configs(
    #     device_type = device,
    #     has_logs = False
    # ) 

    # IF has_logs FALSE WILL CREATE LOG FILES BUT ERASE ALL PREVIOUS ONES

    # train the models
    # COMPUTATIONALLY EXPENSIVE
    for model in models:
        cfg = GPTConfiguration(
            n_layer = 5,
            n_head = 5,
            n_embed = 320,
            lr = 1e-3 if model == "pre" else 1e-4,
            dataset = f"shakespeare_{model if model != 'pre' else 'char'}",
            name = model
        )
        cfg.set_backend()
        cfg.write(
            os.path.join(
                os.path.dirname(__file__), 
                "nanoGPT"
            )
        )

        # TODO: do that with subprocess.run (and also here)
        part_2_training.train(model)

    new_dest = shutil.move(
        os.path.join(
            os.path.dirname(__file__), 
            'nanoGPT',
            'part_2_logs'
        ), 
        os.path.dirname(__file__)
    )

    # create file if didn't exist
    with open(os.path.join(
        os.path.dirname(__file__), 
        'part_2_logs',
        'accuracies.log'
    ), 'a+') as f:
        f.close()

    # evaluate the models
    for model in models:
        part_2_evaluation.evaluate_model(
            model = model,
            device = device
        )