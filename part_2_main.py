import part_2_prepare_sft
import part_2_train_config
import part_2_training
import part_2_evaluation


models = ['Task 1', 'Task 2', 'Multi-Task', 'Pre-Trained']
#preparing the data
for model in models:
    part_2_prepare_sft.prepare_training(model)

#prepare the training files
part_2_train_config.prepare_configs(device_type='cuda', has_logs=False) #IF FALSE WILL CREATE LOG FILES BUT ERASE ALL PREVIOUS ONES

#train the models
#COMPUTATIONALLY EXPENSIVE
for model in models:
    part_2_training.train(model)

#evaluate the models
for model in models:
    part_2_evaluation.evaluate_model(model)