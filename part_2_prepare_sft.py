from collections import Counter
from email.mime import text # not accessed - needed?
import re
import numpy as np
import random
#from part_2_prose_vs_verse_classifier import ProseVerseClassifier
import os
import pickle
import requests

def sft_prepare_task1(data):
    speakers = []
    for line in data.splitlines():
        match = re.match(r'^([A-Z\s]+):$', line)
        if match:
            speakers.append(match.group(1))

    speaker_counts = Counter(speakers)
    common_speakers = np.array(speaker_counts.most_common(10)).T[0]
    print(common_speakers)

    all_dialogues = data.split('\n\n')
    speaker_identification_dataset = []
    for speaker in common_speakers:
        for block in all_dialogues:
            if block.startswith(speaker + ':'):
                speaker_token = '@' #replaces [SPEAKER]
                answer_token = '<' #replaces [ANSWER]
                end_token = '>' #replaces [END]
                input = f'{speaker_token} ' + block[len(speaker) + 1:].strip() + f' {answer_token} ' + speaker + f' {end_token}'
                speaker_identification_dataset.append(input)
    random.shuffle(speaker_identification_dataset)

    full_dataset = '\n\n'.join(speaker_identification_dataset)
    print(len(speaker_identification_dataset))

    return full_dataset

def sft_prepare_task2(data):
    all_dialogues = data.split('\n\n')
    #clf = ProseVerseClassifier()
    VerseVsProse_identification_dataset = []
    vowels = ['a', 'i', 'u', 'e', 'o', 'y']
    classify_token = '|' #replaces [CLASSIFY]
    answer_token = '<' #replaces [ANSWER]
    end_token = '>' #replaces [END]

    for block in all_dialogues:
        lines = block.split('\n')
        lines = lines[1:]
        vowel_count_list = []
        if len(lines) >= 3 and len(lines) <= 5:
            for line in lines:
                char_counter = Counter(line)
                vowel_count = 0
                for vowel in vowels:
                    vowel_count += char_counter[vowel]
                vowel_count_list.append(vowel_count)
            
            if np.any(np.abs(np.diff(np.array(vowel_count_list))) > 3):
                dialogue_classification = 'PROSE'
            elif np.all(np.abs(np.diff(np.array(vowel_count_list))) <= 3):
                dialogue_classification = 'VERSE'
            block_without_speaker = '\n'.join(lines)
            input = f'{classify_token} ' + block_without_speaker + f' {answer_token} ' + dialogue_classification + f' {end_token}'
            VerseVsProse_identification_dataset.append(input)
            #list_of_texts.append(block_without_speaker)

    full_dataset = '\n\n'.join(VerseVsProse_identification_dataset)
    print(len(VerseVsProse_identification_dataset))

    return full_dataset

def sft_prepare_multitask(data):
    speaker_identification_dataset = sft_prepare_task1(data).split('\n\n')
    VerseVsProse_identification_dataset = sft_prepare_task2(data).split('\n\n')
    combined_dataset = speaker_identification_dataset + VerseVsProse_identification_dataset
    random.shuffle(combined_dataset)
    full_dataset = '\n\n'.join(combined_dataset)
    return full_dataset


# Prepare the Shakespeare dataset for character-level language modeling.
# So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
# Will save train.bin, val.bin containing the ids, and meta.pkl containing the
# encoder and decoder and some other related info.

def prepare_training(task: str):
    """
    Prepare training based on the task  with SFT given by modifying the dataset for said task
    Parameters:
        task (str): 4 choices
            'task1': trained for speaker identification
            'task3': trained for verse vs prose identification
            'multi': trained for both at the same time
            'pre': trained without any specific task
    """
    # download the tiny shakespeare dataset
    input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
    if not os.path.exists(input_file_path):
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        with open(input_file_path, 'w') as f:
            f.write(requests.get(data_url).text)

    with open(input_file_path, 'r') as f:
        data = f.read()
    print(f"length of dataset in characters: {len(data):,}")

    # get all the unique characters that occur in this text
    if task == 'task1':
        chars = sorted(list(set(data)) + ['@', '<', '>'])
    elif task == 'task2':
        chars = sorted(list(set(data)) + ['|', '<', '>'])
    elif task == 'multi':
        chars = sorted(list(set(data)) + ['@', '|', '<', '>'])
    elif task == 'pre':
        chars = sorted(list(set(data)))
    else:
        raise ValueError(
            "Parameter 'task' not recognised. Expected 'task1', 'task2', 'multi', or 'pre'\n"
            f"Got {task} instead"
        )
    
    vocab_size = len(chars)
    print("all the unique characters:", ''.join(chars))
    print(f"vocab size: {vocab_size:,}")

    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    #print(stoi)
    itos = { i:ch for i,ch in enumerate(chars) }
    def encode(s):
        return [stoi[c] for c in s] # encoder: take a string, output a list of integers
    def decode(L):
        return ''.join([itos[i] for i in L]) # decoder: take a list of integers, output a string

    # create the train and test splits
    if task == 'task1':
        preprocessed_data = sft_prepare_task1(data)
    elif task == 'task2':
        preprocessed_data = sft_prepare_task2(data)
    elif task == 'multi':
        preprocessed_data = sft_prepare_multitask(data)
    elif task == 'pre':
        preprocessed_data = data
    #print(preprocessed_data)
    n = len(preprocessed_data)
    train_data = preprocessed_data[:int(n*0.8)] # make sure the validation set is not too small
    val_data = preprocessed_data[int(n*0.8):] # make sure the validation set is not too small

    # encode both to integers
    train_ids = encode(train_data)
    val_ids = encode(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # export to bin files
    train_ids = np.array(train_ids, dtype = np.uint16)
    val_ids = np.array(val_ids, dtype = np.uint16)

    # save the meta information as well, to help us encode/decode later
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }

    basepath = os.path.join(
        os.path.dirname(__file__),
        "nanoGPT",
        "data",
    )
    if task == 'task1':
        longpath = os.path.join(
            basepath,
            "shakespeare_task1"
        )
        if not os.path.exists(longpath):
            os.mkdir(longpath)
        
    elif task == 'task2':
        longpath = os.path.join(
            basepath,
            "shakespeare_task2"
        )
        if not os.path.exists(longpath):
            os.mkdir(longpath)

    elif task == 'multi':
        longpath = os.path.join(
            basepath,
            "shakespeare_multitask"
        )
        if not os.path.exists(longpath):
            os.mkdir(longpath)
    elif task == 'pre':
        longpath = os.path.join(
            basepath,
            "shakespeare_char"
        )
        if not os.path.exists(longpath):
            os.mkdir(longpath)
        

    # save train and test
    train_ids.tofile(
        os.path.join(longpath, "train.bin")
    )
    val_ids.tofile(
        os.path.join(longpath, "val.bin")
    )

    # and metadata
    with open(os.path.join(longpath, "meta.pkl"), 'wb') as f:
        pickle.dump(meta, f)

# length of dataset in characters:  1115394
# all the unique characters:
#  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# vocab size: 65
# train has 1003854 tokens
# val has 111540 tokens
