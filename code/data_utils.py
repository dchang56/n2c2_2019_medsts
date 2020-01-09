import csv
import os
import logging
import json
import numpy as np
import torch
#testing
from tqdm import trange, tqdm
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)

class InputExample(object):
	def __init__(self, pid, text_a, text_b=None, aug_a=None, aug_b=None, label=None, label_c=None, label_c2=None, label_type=None, label_soft=None):
		self.pid = pid
		self.text_a = text_a
		self.text_b = text_b
		self.aug_a = aug_a
		self.aug_b = aug_b
		self.label = label
		self.label_c = label_c
		self.label_c2 = label_c2
		self.label_type = label_type
		self.label_soft = label_soft

class InputFeatures(object):
	def __init__(self, input_ids, input_mask, segment_ids, label_id, pid, label_soft_id=None):
		self.input_ids = input_ids
		self.input_mask = input_mask
		self.segment_ids = segment_ids
		self.label_id = label_id
		self.pid = pid
		self.label_soft_id = label_soft_id

class DataProcessor(object):
	def get_train_examples(self, data_dir):
		raise NotImplementedError()
	def get_dev_examples(self, data_dir):
		raise NotImplementedError()
	def get_labels(self):
		raise NotImplementedError()
	@classmethod
	def _read_tsv(cls, input_file, quotechar=None):
		with open(input_file, "r") as f:
			reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
			lines = []
			for line in reader:
				lines.append(line)
			return lines
	@classmethod
	def _read_jsonl(cls, input_file):
		with open(input_file, 'r') as f:
			lines = []
			for line in f:
				lines.append(json.loads(line))
			return lines

class MednliProcessor(DataProcessor):
	def get_train_examples(self, data_dir):
		return self._create_examples(
			self._read_jsonl(os.path.join(data_dir, "train.jsonl")))
	def get_dev_examples(self, data_dir):
		return self._create_examples(
			self._read_jsonl(os.path.join(data_dir, "dev.jsonl")))
	def get_labels(self):
		return ['contradiction', 'entailment', 'neutral']
	def _create_examples(self, lines):
		examples = []
		for (i, line) in enumerate(lines):
			pid = tokenization.convert_to_unicode(i)
			text_a = tokenization.convert_to_unicode(line['sentence1'])
			text_b = tokenization.convert_to_unicode(line['sentence2'])
			label = tokenization.convert_to_unicode(line['label'])
			examples.append(
				InputExample(pid=pid, text_a=text_a, text_b=text_b, label=label))
		return examples

class MedstsProcessor(DataProcessor):
	def get_train_examples(self, data_dir):
		return self._create_examples(
			self._read_jsonl(os.path.join(data_dir, 'train.jsonl')))
	def get_dev_examples(self, data_dir):
		return self._create_examples(
			self._read_jsonl(os.path.join(data_dir, 'dev.jsonl')))
	def get_labels(self):
		return ['None']
	def _create_examples(self, lines):
		examples = []
		for (i, line) in enumerate(lines):
			pid = line['pid']
			text_a = line['sentence1']
			aug_a = line['augment1']
			text_b = line['sentence2']
			aug_b = line['augment2']
			label = line['label']
			if 'label_soft' in line.keys():
				label_soft = line['label_soft']
				if type(label_soft) != float:
					print(label_soft)
				examples.append(
				InputExample(pid=pid, text_a=text_a, text_b=text_b, aug_a=aug_a, aug_b=aug_b, label=label, label_soft=label_soft))
			else:
				examples.append(
				InputExample(pid=pid, text_a=text_a, text_b=text_b, aug_a=aug_a, aug_b=aug_b, label=label))
		return examples

def convert_examples_to_features(examples, label_list,  max_seq_length, tokenizer, output_mode, augment):
	label_map = {label: i for i, label in enumerate(label_list)}
	features = []
	max_len = 0
	print('@@@@ converting examples, with augment {} @@@'.format(augment))
	for (ex_index, example) in enumerate(examples):
		tokens_a = tokenizer.tokenize(example.text_a)
		if augment:
			tokens_aug_a = tokenizer.tokenize(example.aug_a)
			tokens_a = tokens_a + tokens_aug_a


		tokens_b = None
		if example.text_b:
			tokens_b = tokenizer.tokenize(example.text_b)
			if augment:
				tokens_aug_b = tokenizer.tokenize(example.aug_b)
				tokens_b = tokens_b + tokens_aug_b
			seq_len = len(tokens_a) + len(tokens_b)
			_truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
		else:
			seq_len = len(tokens_a)
			if len(tokens_a) > max_seq_length - 2:
				tokens_a = tokens_a[:(max_seq_length-2)]
		if seq_len > max_len:
			max_len = seq_len

		tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
		segment_ids = [0] * len(tokens)
		if tokens_b:
			tokens += tokens_b + ["[SEP]"]
			segment_ids += [1] * (len(tokens_b)+1)
		input_ids = tokenizer.convert_tokens_to_ids(tokens)
		input_mask = [1] * len(input_ids)
		padding = [0] * (max_seq_length - len(input_ids))
		input_ids += padding
		input_mask += padding
		segment_ids += padding

		assert len(input_ids) == max_seq_length
		assert len(input_mask) == max_seq_length
		assert len(segment_ids) == max_seq_length


		label_id = float(example.label)
		if type(example.label_soft)==float:
			label_soft_id = float(example.label_soft)
			features.append(
				InputFeatures(
					input_ids=input_ids,
					input_mask=input_mask,
					segment_ids=segment_ids,
					label_id=label_id,
					pid=example.pid,
					label_soft_id=label_soft_id))
		else:
			features.append(
				InputFeatures(
					input_ids=input_ids,
					input_mask=input_mask,
					segment_ids=segment_ids,
					label_id=label_id,
					pid=example.pid))
	return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
	while True:
		total_length = len(tokens_a) + len(tokens_b)
		if total_length <= max_length:
			break
		if len(tokens_a) > len(tokens_b):
			tokens_a.pop()
		else:
			tokens_b.pop()


processors = {
	'medsts': MedstsProcessor,
	'mednli': MednliProcessor,
}

output_modes = {
	'medsts': 'regression',
	'mednli': 'classification',
}

