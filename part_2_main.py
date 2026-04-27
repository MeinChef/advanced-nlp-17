import part_2_prepare_sft
import part_2_train_config
import part_2_training

#preparing the data
part_2_prepare_sft.prepare_training('Task 1')
part_2_prepare_sft.prepare_training('Task 2')
part_2_prepare_sft.prepare_training('Multi-Task')
part_2_prepare_sft.prepare_training('Pre-Trained')

#prepare the training files
part_2_train_config.prepare_configs(device_type='cuda', has_logs=True)

#train the models
part_2_training.train('Task 2')