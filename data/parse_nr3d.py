import os
import json, pickle
import pandas as pd

from functools import reduce
from collections import Counter
from typing import List
from tqdm import tqdm
from ast import literal_eval

scannet_meta_root = os.path.join('scannet', 'meta_data')
scannetv2_train = open(os.path.join(scannet_meta_root, 'scannetv2_train.txt'), 'r').read().split('\n')
scannetv2_val = open(os.path.join(scannet_meta_root, 'scannetv2_val.txt'), 'r').read().split('\n')

def parse_tokens(sentence: str) -> List[str]:
    sentence = sentence.lower()
    check_special_token = lambda char: (
        (ord(char) <= ord('z') and ord(char) >= ord('a')) or \
        (ord(char) <= ord('9') and ord(char) >= ord('0'))
    )
    sentence = ''.join(
        char if check_special_token(char) else ' ' + char + ' '  for char in sentence
    )
    tokens = list(filter(lambda token: token != '', sentence.split(' ')))
    return tokens


## organize nr3d dataset

df = pd.read_csv('nr3d.csv')
df.tokens = df["tokens"].apply(literal_eval)

nr3d_train, nr3d_val = [], []

for _, row in tqdm(df.iterrows()):
    entry = {
        "scene_id": row["scan_id"],
        "object_id": str(row["target_id"]),
        "object_name": row["instance_type"],
        "ann_id": str(row["assignmentid"]),
        "description": row["utterance"].lower(),
        "token": parse_tokens(row["utterance"])
    }
    if entry['scene_id'] in scannetv2_train:
        nr3d_train.append(entry)
    elif entry['scene_id'] in scannetv2_val:
        nr3d_val.append(entry)


nr3d_train_scene_list = sorted(set(corpus['scene_id'] for corpus in nr3d_train))
nr3d_val_scene_list = sorted(set(corpus['scene_id'] for corpus in nr3d_val))

with open('nr3d_train.json', "w") as f:
    json.dump(nr3d_train, f, indent=4)

with open('nr3d_val.json', "w") as f:
    json.dump(nr3d_val, f, indent=4)
    
with open('nr3d_train.txt', 'w') as f:
    f.write('\n'.join(nr3d_train_scene_list))
    
with open('nr3d_val.txt', 'w') as f:
    f.write('\n'.join(nr3d_val_scene_list))


## build vocabulary
if not os.path.isfile('nr3d_vocabulary.json'):
    glove = pickle.load(open('glove.p', "rb"))
    all_words = reduce(lambda x, y: x + y, [data["token"] for data in nr3d_train])
    word_counter = Counter(all_words)
    word_counter = sorted(
        [(k, v) for k, v in word_counter.items() if k in glove], 
        key=lambda x: x[1], reverse=True
    )
    word_list = [k for k, _ in word_counter]
    
    # build vocabulary
    word2idx, idx2word = {}, {}
    spw = ["pad_", "unk", "sos", "eos"] # NOTE distinguish padding token "pad_" and the actual word "pad"
    for i, w in enumerate(word_list):
        shifted_i = i + len(spw)
        word2idx[w] = shifted_i
        idx2word[shifted_i] = w
    
    # add special words into vocabulary
    for i, w in enumerate(spw):
        word2idx[w] = i
        idx2word[i] = w
    
    vocab = {
        "word2idx": word2idx,
        "idx2word": idx2word
    }
    json.dump(vocab, open('nr3d_vocabulary.json', "w"), indent=4)
