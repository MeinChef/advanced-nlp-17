from collections import Counter
from email.mime import text
import re
import numpy as np
import random

def sft_prepare(data):
    speakers = []
    for line in data.splitlines():
        match = re.match(r'^([A-Z\s]+):$', line)
        if match:
            speakers.append(match.group(1))

    speaker_counts = Counter(speakers)
    common_speakers = np.array(speaker_counts.most_common(10)).T[0]

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

    return full_dataset


"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np

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
chars = sorted(list(set(data)) + ['@', '<', '>'])
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
print(stoi)
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
sft_data = sft_prepare(data)
n = len(sft_data)
train_data = sft_data[:int(n*0.8)] # make sure the validation set is not too small
val_data = sft_data[int(n*0.8):] # make sure the validation set is not too small

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# length of dataset in characters:  1115394
# all the unique characters:
#  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# vocab size: 65
# train has 1003854 tokens
# val has 111540 tokens
