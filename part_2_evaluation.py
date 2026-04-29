import numpy as np
import os
import pickle
import re

def decode_val_bin(val_bin_path, meta_path, num_chars=None):
    """
    Decode val.bin back to plain text using character-level tokenizer.
    
    Args:
        val_bin_path: path to val.bin
        meta_path:    path to meta.pkl (created alongside train.bin/val.bin)
        num_chars:    how many characters to decode (None = all)
    
    Returns:
        decoded string
    """

    # Load the vocab mappings
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    
    itos = meta['itos']  # int -> char
    data = np.fromfile(val_bin_path, dtype=np.uint16) # Load the binary data
    
    if num_chars is not None:
        data = data[:num_chars]
    
    # Decode
    decoded = ''.join(itos[i] for i in data)
    return decoded

def parse_entry(token: str, dialogue: str):
    match = re.match(rf'({token}\s*.*?\s*<)\s*([A-Z_]+(?:\s+[A-Z_]+)*)\s*>', dialogue, re.DOTALL)
    if not match:
        #print(dialogue)
        return ('', '')
    text = match.group(1).strip()
    label = match.group(2)
    return (text, label)

def evaluate_model(model: str):
    # Usage
    if model == 'Task 1':
        val_preprocessed_dataset = decode_val_bin(os.path.join(os.path.dirname(__file__), 'nanoGPT/data/shakespeare_task1/val.bin'), os.path.join(os.path.dirname(__file__), 'nanoGPT/data/shakespeare_task1/meta.pkl'))
    elif model == 'Task 2':
        val_preprocessed_dataset = decode_val_bin(os.path.join(os.path.dirname(__file__), 'nanoGPT/data/shakespeare_task2/val.bin'), os.path.join(os.path.dirname(__file__), 'nanoGPT/data/shakespeare_task2/meta.pkl'))
    elif model == 'Multi-Task':
        decode_val_bin(os.path.join(os.path.dirname(__file__), 'nanoGPT/data/shakespeare_multitask/val.bin'), os.path.join(os.path.dirname(__file__), 'nanoGPT/data/shakespeare_multitask/meta.pkl'))
    elif model == 'Pre-Trained':
        print(f'accuracy of the {model}-model:')
        print(f'- Task 1: 0%')
        print(f'- Task 2: 0%')
        return 0
    else:
        print('not a supported model for training')
    
    val_processed_dataset = val_preprocessed_dataset.split('\n\n')
    #print(val_processed_dataset)
    if model == 'Task 1':
        val_dataset = [parse_entry('@', entry) for entry in val_processed_dataset]#.remove(('', ''))
    elif model == 'Task 2':
        val_dataset = [parse_entry('|', entry) for entry in val_processed_dataset]#.remove(('', ''))
    elif model == 'Multi-Task':
        val_dataset = [parse_entry('@', entry) for entry in val_processed_dataset]#.remove(('', ''))
        val_dataset += [parse_entry('|', entry) for entry in val_processed_dataset]#.remove(('', ''))
    
    #print(val_dataset[:20])
    terminal_output = ""
    correct = [0, 0] #task 1, task 2-specific correct
    total = [0, 0]
    for sample_input in val_dataset:
        if model == 'Task 1':
            out_dir = 'task1'
        elif model == 'Task 2':
            out_dir = 'task2'
        elif model == 'Multi-Task':
            out_dir = 'multitask'
        terminal_output = os.popen(f"cd {os.path.join(os.path.dirname(__file__), 'nanoGPT')} && python sample.py --out_dir=out-shakespeare_{out_dir} --device=cuda --num_samples=1 --max_new_tokens=30 --start=\"{sample_input[0]}\"").read()
        try:
            match = re.search(rf'<\s*([A-Z_]+)\s*>', terminal_output, re.DOTALL).group(1).strip()
        except:
            match = ''
        
        #print(f'Real Label: {sample_input[1]}, Generated Label: {match}')
        
        if match == sample_input[1]:
            if sample_input[1] in ["VERSE", "PROSE"]:
                correct[1] += 1
                total[1] += 1
            else:
                correct[0] += 1
                total[0] += 1
        else:
            if sample_input[1] in ["VERSE", "PROSE"]:
                total[1] += 1
            else:
                total[0] += 1
    
    try:
        accuracy_task1 = correct[0] / total[0]
    except:
        accuracy_task1 = 0
    try:
        accuracy_task2 = correct[1] / total[1]
    except:
        accuracy_task2 = 0
    print(f'accuracy of the {model}-model:')
    print(f'- Task 1: {accuracy_task1 * 100}%')
    print(f'- Task 2: {accuracy_task2 * 100}%')

    with open(os.path.join(os.path.dirname(__file__), 'part_2_logs/accuracies.log'), 'a') as f:
        f.write(str([model, accuracy_task1, accuracy_task2]) + '\n')
    f.close()