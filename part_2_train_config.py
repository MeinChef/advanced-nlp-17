import re
import os

def prepare_configs(device_type: str, has_logs: bool = True):
    """
    prepare the config files for the training
    Parameters:
        - device_type (str): use 'cuda' if you support cuda and Linux, otherwise it will use the cpu
        - has_logs (bool): if set to True, the function will not overwrite previous log files
    """
    #writing the file with all model and training parameters, NOTE: set to gpu with linux
    file_content_standard = '# Baseline configuration \nout_dir = \'out-shakespeare-baseline\' \neval_interval = 250 \neval_iters = 200 \nlog_interval = 10 \nalways_save_checkpoint = False \nwandb_log = False \nwandb_project = \'nanoGPT-assignment\' \nwandb_run_name = \'baseline\' \ndataset = \'shakespeare_char\' \ngradient_accumulation_steps = 1 \nbatch_size = 64 \nblock_size = 256 \nn_layer = 5 \nn_head = 5 \nn_embd = 310 \ndropout = 0.2 \nlearning_rate = 1e-3 \nmax_iters = 1500 \nlr_decay_iters = 5000 \nmin_lr = 1e-4 \nbeta2 = 0.99 \nwarmup_iters = 100 \nweight_decay = 1e-1 \ndevice = \'cpu\' # change to \'cuda\' if you have a GPU \ncompile = False # set True only on Linux with GPU'
    file_content_linux_gpu = '# Baseline configuration \nout_dir = \'out-shakespeare-baseline\' \neval_interval = 250 \neval_iters = 200 \nlog_interval = 10 \nalways_save_checkpoint = False \nwandb_log = False \nwandb_project = \'nanoGPT-assignment\' \nwandb_run_name = \'baseline\' \ndataset = \'shakespeare_char\' \ngradient_accumulation_steps = 1 \nbatch_size = 64 \nblock_size = 256 \nn_layer = 5 \nn_head = 5 \nn_embd = 310 \ndropout = 0.2 \nlearning_rate = 1e-3 \nmax_iters = 1500 \nlr_decay_iters = 5000 \nmin_lr = 1e-4 \nbeta2 = 0.99 \nwarmup_iters = 100 \nweight_decay = 1e-1 \ndevice = \'cuda\' # change to \'cuda\' if you have a GPU \ncompile = True # set True only on Linux with GPU'
    if device_type == 'cuda':
        file_content = file_content_linux_gpu
    else:
        file_content = file_content_standard

    #all the new parameters in a list
    experiments_variants = ['shakespeare_task1', 'shakespeare_task2', 'shakespeare_multitask', 'shakespeare_char']

    for experiment in experiments_variants:
        #modify the content from baseline with the new parameter
        if experiment != 'shakespeare_char':
            new_file_content = re.sub(r'(learning_rate = ).*', rf'\g<1>1e-4', file_content)
            #print(new_file_content)
        else:
            new_file_content = re.sub(r'(learning_rate = ).*', rf'\g<1>1e-3', file_content)
        
        new_file_content = re.sub(r'(dataset = ).*', rf"\g<1>'{experiment}'", new_file_content)
        #change the model storage folder
        new_file_content = re.sub(r'(out_dir = )\S+', rf"\g<1>'out-{experiment}'", new_file_content)
        #when modifying the number of embeddings we have to modify the number of heads too
        #write the parameters to a new config file
        with open(f'nanoGPT/config/training_{experiment}.py', 'w') as file:
            file.write(new_file_content)
        file.close()

    if not has_logs:
        try:
            os.mkdir('nanoGPT/logs') #try to create the log folder if it doesn't exist
        except:
            print('folder either already exists or could not be created')
        finally: #create one log file for each model for training later on
            for experiment in experiments_variants:
                with open(f'nanoGPT/logs/{experiment}.log', 'w+') as file:
                    pass