import numpy as np
import os
import pickle
import re
from sft.sample import TextGenerator

def decode_val_bin(
        val_bin_path, 
        meta_path, 
        num_chars = None
    ) -> str:
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
    data = np.fromfile(val_bin_path, dtype = np.uint16) # Load the binary data
    
    if num_chars is not None:
        data = data[:num_chars]
    
    # Decode
    decoded = ''.join(itos[i] for i in data)
    return decoded

def parse_entry(
        token: str, 
        dialogue: str
    ) -> tuple[str, str]:
    
    match = re.match(
        rf'({token}\s*.*?\s*<)\s*([A-Z_]+(?:\s+[A-Z_]+)*)\s*>', 
        dialogue, 
        re.DOTALL
    )
    
    if not match:
        #print(dialogue)
        return ('', '')
    
    text = match.group(1).strip()
    label = match.group(2)
    return (text, label)

def evaluate_model(
        model: str,
        device: str = "cuda"
    ) -> None:
    # Usage

    if model == 'pre':
        print(f'accuracy of the {model}-model:')
        print('- task1: 0%')
        print('- task2: 0%')
        return 
    
    val_preprocessed_dataset = decode_val_bin(
        val_bin_path = os.path.join(
            os.path.dirname(__file__), 
            'nanoGPT',
            'data',
            f'shakespeare_{model}',
            'val.bin'
        ), 
        meta_path = os.path.join(
            os.path.dirname(__file__), 
            'nanoGPT',
            'data',
            f'shakespeare_{model}',
            'meta.pkl'
        )
    )
    
    val_processed_dataset = val_preprocessed_dataset.split('\n\n')
    #print(val_processed_dataset)
    if model == 'task1':
        val_dataset = [parse_entry('@', entry) for entry in val_processed_dataset]#.remove(('', ''))
    elif model == 'task2':
        val_dataset = [parse_entry('|', entry) for entry in val_processed_dataset]#.remove(('', ''))
    elif model == 'multi':
        val_dataset = [parse_entry('@', entry) for entry in val_processed_dataset]#.remove(('', ''))
        val_dataset += [parse_entry('|', entry) for entry in val_processed_dataset]#.remove(('', ''))
    else:
        val_dataset = val_preprocessed_dataset

    # init GPT-Model because subprocess is very overhead-y
    generator = TextGenerator(
        out_dir = os.path.join(
            os.path.dirname(__file__),
            "nanoGPT",
            f"out-shakespeare-{model}"
        ),
        device = device
    )
    print("Using path: ",
        os.path.join(
            os.path.dirname(__file__),
            'nanoGPT',
            f'out-shakespeare-{model}'
        )
    )

    correct = [0, 0] #task 1, task 2-specific correct
    total = [0, 0]
    for sample_input in val_dataset[:len(val_dataset) // 5]:
        if not sample_input[0] or not sample_input[1]:
            continue
        
        terminal_output = generator.generate(
            sample_input[0],
            max_new_tokens = 30
        )
        # proc = subprocess.run(
        #     [
        #         sys.executable,
        #         "-m",
        #         "sample",
        #         f"--out_dir=out-shakespeare-{model}",
        #         "--num_samples=1",
        #         f"--device={device}",
        #         "--max_new_tokens=30",
        #         f"--start='{sample_input[0]}'"
        #     ],
        #     cwd = os.path.join(
        #         os.path.dirname(__file__),
        #         "nanoGPT"
        #     ),
        #     encoding = "utf-8",
        #     stdout = subprocess.PIPE
        # )
        # terminal_output = proc.stdout

        match = re.search(
            r'<\s*([A-Z_]+)\s*>',
            terminal_output, 
            re.DOTALL
        )

        if match:
            match = match.group(1).strip()
        else:
            match = ''
        
        print(f'Real Label: {sample_input[1]}, Generated Label: {match}')
        
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
    
    if total[0] == 0:
        accuracy_task1 = 0
    else:
        accuracy_task1 = correct[0] / total[0]
    if total[1] == 0:
        accuracy_task2 = 0
    else:
        accuracy_task2 = correct[1] / total[1]

    print(f'accuracy of the {model}-model:')
    print(f'- task1: {accuracy_task1 * 100}%')
    print(f'- task2: {accuracy_task2 * 100}%')

    with open(
        os.path.join(
            os.path.dirname(__file__), 
            'logs',
            f'out-shakespeare-{model}',
            'accuracies.log'
            ), 
            'a+'
        ) as f:
        f.write(str([model, accuracy_task1, accuracy_task2]) + '\n')