import csv
import json
import os
import argparse
import logging
import random
import numpy as np
from sklearn.model_selection import KFold

random.seed(5)
medsts_path = '/home/dc925/project/data/seq_pair/MEDSTS'
type_annotation = os.path.join(medsts_path, 'annotated_data.csv')


def preprocess_medsts(medsts_path):
	original = os.path.join(medsts_path, 'train.txt')
	original_out = os.path.join(medsts_path, 'data.jsonl')

	type_labels_map = {'condition': 0, 'interaction':1, 'medication': 2, 'misc': 3}
	# original_c_out = os.path.join(medsts_discretized_path, 'data.jsonl')
	# os.makedirs(medsts_discretized_path, exist_ok=True)
	type_labels = {}
	with open(type_annotation, 'r') as f:
		reader = csv.reader(f)
		for line in reader:
			type_labels[line[0]] = line[4]
		type_labels.pop('')

	with open(original, 'r') as f:
		reader = csv.reader(f, delimiter='\t')
		lines = []
		# lines_c = []
		for i, line in enumerate(reader):
			sent1 = remove_slash_and_brackets(line[0])
			sent2 = remove_slash_and_brackets(line[1])
			label = float(line[2])
			#discretize
			if label >= 0 and label < 1.9:
				label_c = 0
			elif label >= 1.9 and label < 3.8:
				label_c = 1
			else:
				label_c = 2
			if label >= 0 and label < 3:
				label_c2 = 0
			elif label >= 3:
				label_c2 = 1

			type_string = type_labels[str(i)].lower()
			label_type = type_labels_map[type_string]
			lines.append({'sentence1':sent1,'sentence2':sent2,'label':label, 'pid':i, 'label_c':label_c, 'label_c2': label_c2, 'label_type': label_type})

	random.shuffle(lines)
	with open(original_out, 'w') as writer:
		for line in lines:
			json.dump(line, fp=writer)
			writer.write("\n")

def medsts_split(medsts_path, train_p=0.8, k=5):
	original = os.path.join(medsts_path, 'data.jsonl')
	with open(original, 'r') as f:
		lines = []
		for line in f:
			lines.append(json.loads(line))
		size = len(lines)
	dataset = np.array(lines)
	kf = KFold(n_splits=k)

	i = 0

	for train_index, dev_index in kf.split(dataset):
		k_fold_path = os.path.join(medsts_path, 'k_fold_{}'.format(i))
		os.makedirs(k_fold_path, exist_ok=True)
		train_out = os.path.join(k_fold_path, 'train.jsonl')
		dev_out = os.path.join(k_fold_path, 'dev.jsonl')

		train = dataset[train_index]
		dev = dataset[dev_index]

		with open(train_out, 'w') as train_file:
			for line in train:
				json.dump(line,fp=train_file)
				train_file.write('\n')
		with open(dev_out, 'w') as dev_file:
			for line in dev:
				json.dump(line, fp=dev_file)
				dev_file.write('\n')
		i += 1


def remove_slash_and_brackets(sentence):
	sentence = sentence.lower()
	sentence = sentence.replace('/', ' ')
	sentence = sentence.replace('[', ' ')
	sentence = sentence.replace(']', ' ')

	return sentence

preprocess_medsts(medsts_path)
medsts_split(medsts_path)
