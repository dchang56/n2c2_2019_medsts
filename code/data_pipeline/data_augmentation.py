import json
import csv
import os
import pandas as pd
import re
import numpy as np
import pickle
import argparse

               
def extract_strings(mappings, sentence):
    strings = []
    for phrase in mappings:
        if phrase['mapping']:
            for m in phrase['mapping']:
                text = m['preferred'].lower().split()
                strings += text
                
                if 'Clinical Drug' in m['semtypes']:
                    strings += ['clinical drug']
                if 'Pharmacologic Substance' in m['semtypes']:
                    strings += ['pharmacologic substance']
    out = " ".join(strings)
    out = re.sub(r'[^\w\s]', ' ', out)
    out = out.split()
    out = [t for t in out if t not in banned_words and t not in sentence]
    out = list(set(out))
    out = " ".join(out)
    return out

def augment_sentences(mm_output, data):

    assert len(mm_output) == len(data)*2
    i = 0
    j = 0
    augment_strings = []
    while i < len(mm_output)-1:
        aug = extract_strings(mm_output[i], data[j]['sentence1'])
        data[j]['augment1'] = aug
        i += 1
        aug = extract_strings(mm_output[i], data[j]['sentence2'])
        data[j]['augment2'] = aug
        i += 1
        j += 1


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--logits_dir', type=str, required=False)
args = parser.parse_args()

data_dir = args.data_dir
logits_dir = args.logits_dir

dev_mapping_path = 'filtered_dev.p'
train_mapping_path = 'filtered_train.p'

with open(os.path.join(data_dir, dev_mapping_path), 'rb') as f:
    dev_mapping = pickle.load(f)
with open(os.path.join(data_dir, train_mapping_path), 'rb') as f:
    train_mapping = pickle.load(f)
dev_path = os.path.join(data_dir, 'dev.jsonl')
train_path = os.path.join(data_dir, 'train.jsonl')

with open(dev_path, 'r') as f:
    dev = []
    for line in f:
        example = json.loads(line)
        dev.append(example)
with open(train_path, 'r') as f:
    train = []
    for line in f:
        example = json.loads(line)
        train.append(example)

## add ensemble logits as label_soft
if logits_dir:
    logits = pd.read_csv(logits_dir, index_col=0)
    best_val = 0
    for n in range(15, len(logits.columns)*4//5, 5):
        top_cols = logits.corr().sort_values(by='label', axis=1, ascending=False).columns[1:n+1]
        logits['ensemble'] = logits[top_cols].mean(axis=1)
        val = logits.corr().iloc[0, -1].item()
        if val > best_val:
            final_n = n

    top_cols = logits.corr().sort_values(by='label', axis=1, ascending=False).columns[1:n+1]
    logits['ensemble'] = logits[top_cols].mean(axis=1)
    preds = logits['ensemble']

    t = 0
    for example in train:
        for pid, logit in preds.items():
            if example['pid'] == pid:
                t+=1
                example['label_soft'] = round(logit, 4)
            if 'label_soft' not in example.keys():
                example['label_soft'] = example['label']


banned_words = ['contextual', 'qualifier', 'value', 'action', '-', 'patients', 'of', 'person', 'feels', 'time', 'to', \
               'contents', 'escalation', 'cavity', 'region', 'medical', 'discussion', 'procedure', 'unit', 'dose', 'appearance', \
               'feelings', 'has', 'does', 'finding', 'function', 'in', 'qualitative', 'changing', 'publication', 'educational', \
                'by', 'for', 'with', 'from', 'continuous', 'transducers', 'process', 'needs', 'individual', 'reporting', 'chief', \
               'relationships', '6', '10', 'syncytial', 'human', 'masks', 'muscle', 'training', 'virus',]

augment_sentences(dev_mapping, dev)
augment_sentences(train_mapping, train)

with open(dev_path, 'w') as f:
    for line in dev:
        json.dump(line, fp=f)
        f.write('\n')
with open(train_path, 'w') as f:
    for line in train:
        json.dump(line, fp=f)
        f.write('\n')

























