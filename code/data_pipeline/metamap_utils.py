import json
import csv
import os
import re
import numpy as np
import argparse

"""
preprocess: lowercase, replace single digits w words, remove list of words, remove numbers, puncs and whitespace,
    add commas after male and female, location
"""

def generate_metamap_input_files_clean(data_dir):
    train_file = os.path.join(data_dir, 'train.jsonl')
    dev_file = os.path.join(data_dir, 'dev.jsonl')
    train_mm_input_file = os.path.join(data_dir, 'train_mm_input.txt')
    dev_mm_input_file = os.path.join(data_dir, 'dev_mm_input.txt')

    with open(train_file, 'r') as f:
        lines = []
        for line in f:
            example = json.loads(line)
            sentence1 = preprocess_mm_input(example['sentence1'])
            sentence2 = preprocess_mm_input(example['sentence2'])
            lines.append(sentence1)
            lines.append(sentence2)
    with open(train_mm_input_file, 'w') as writer:
        for line in lines:
            writer.write(line)
            writer.write('\n')
            writer.write('\n')
        writer.write('\n')

    with open(dev_file, 'r') as f:
        lines = []
        for line in f:
            example = json.loads(line)
            sentence1 = preprocess_mm_input(example['sentence1'])
            sentence2 = preprocess_mm_input(example['sentence2'])
            lines.append(sentence1)
            lines.append(sentence2)
    with open(dev_mm_input_file, 'w') as writer:
        for line in lines:
            writer.write(line)
            writer.write('\n')
            writer.write('\n')
        writer.write('\n')

def preprocess_mm_input(sent):
    digit_to_word = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4':'four','5':'five',
                     '6':'six','7':'seven','8':'eight','9':'nine', '10': 'ten'}
    stopwords = ['mg','mcg','ml', 'liter', "tablets", "tablet",'capsules', 'capsule']
    sent = sent.lower()
    nums = re.findall(r'\s[0-9]\s', sent)
    nums = [num.strip() for num in nums]
    for num in nums:
        sent = sent.replace(num, digit_to_word[num])
    sent = re.sub(r'[^\w\s]', ' ', sent) #dash should be replace w a space to prevent collapsing separate terms, that confuses metamap
    filter_words = re.compile('|'.join(map(re.escape, stopwords)))
    sent = filter_words.sub('', sent)
    sent = re.sub(r'[\d]*', '', sent)
    sent = re.sub(r'[\s]+', ' ', sent)
    sent = re.sub('male and female', 'male and female,', sent)
    sent = re.sub('location', ', location,', sent)
    sent = re.sub('discussed', 'discussion', sent)
    sent = re.sub('discussing', 'discussion', sent)
    sent = re.sub('discuss ', 'discussion', sent)
    sent = re.sub('as well as', 'and', sent)
    sent = re.sub('albuterol', ', albuterol,', sent)
    sent = re.sub('vitamin b', ', vitamin b,', sent)
    sent = re.sub('d ay', 'day', sent)
    sent = re.sub('call back', 'call', sent)
    sent = re.sub('immunization', ', immunization,', sent)
    sent = re.sub('diabetes', ', diabetes,', sent)
    sent = re.sub('release', '', sent)
    sent = re.sub('vaginal discharge', ', vaginal discharge,', sent)
    sent = re.sub('back-up', '', sent)
    sent = re.sub('back here', 'here', sent)
    sent = re.sub('suspicious', 'suspected', sent)
    sent = re.sub('care unit', 'care unit, pacu,', sent)
    
    return sent


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',
		required=True,
		type=str)
args = parser.parse_args()

data_dir = args.data_dir

generate_metamap_input_files_clean(data_dir)
















