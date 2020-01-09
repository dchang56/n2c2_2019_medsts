import argparse
import os
import numpy as np
import random
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from tqdm import tqdm, trange
from scipy.stats import pearsonr
import pandas as pd
from data_utils import *
import logging
import pickle
from torch_geometric.data import DataLoader as GraphDataLoader

from itertools import cycle

def setup_parser():
	parser = argparse.ArgumentParser()
	# Required parameters
	parser.add_argument("--data_dir", type=str, required=True)
	parser.add_argument("--output_dir", type=str, required=True)
	parser.add_argument("--task_name", type=str, required=True)
	parser.add_argument("--config_name", default="", type=str,
						help="Pretrained config name or path if not the same as model_name")
	parser.add_argument("--text_encoder_checkpoint", type=str, required=True)

	parser.add_argument("--cache_dir", default="", type=str)
	parser.add_argument("--max_seq_length", default=128, type=int)
	parser.add_argument("--do_train", action='store_true')
	parser.add_argument("--do_eval", action='store_true')
	parser.add_argument("--evaluate_during_training", action='store_true')
	parser.add_argument("--do_lower_case", action='store_true')
	parser.add_argument("--per_gpu_train_batch_size", default=8, type=int)
	parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int)
	parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
	parser.add_argument("--learning_rate", default=5e-5, type=float,
						help="The initial learning rate for Adam.")
	parser.add_argument("--weight_decay", default=0.0, type=float)
	parser.add_argument("--adam_epsilon", default=1e-8, type=float)
	parser.add_argument("--max_grad_norm", default=1.0, type=float)
	parser.add_argument("--num_train_epochs", default=3.0, type=float)
	parser.add_argument("--max_steps", default=-1, type=int,
						help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
	parser.add_argument("--warmup_steps", default=0, type=int,
						help="Linear warmup over warmup_steps.")

	parser.add_argument('--logging_steps', type=int, default=100)
	parser.add_argument('--save_steps', type=int, default=300)
	parser.add_argument("--eval_all_checkpoints", action='store_true')
	parser.add_argument("--no_cuda", action='store_true')

	parser.add_argument('--overwrite_output_dir', action='store_true',
						help="Overwrite the content of the output directory")
	parser.add_argument('--overwrite_cache', action='store_true',
						help="Overwrite the cached training and evaluation sets")
	parser.add_argument('--seed', type=int, default=42)
	parser.add_argument('--df', action='store_true')
	parser.add_argument('--augment', action='store_true')

	parser.add_argument('--aux', type=str, default='none')

	parser.add_argument('--text_only', action='store_true')




	## For Graph Encoder
	parser.add_argument('--n_hidden', type=int, default=128)
	parser.add_argument('--min_score', type=float, default=0.01)
	parser.add_argument('--graph_lr', type=float, default=0.0015)
	parser.add_argument('--do_layernorm', action='store_true')

	parser.add_argument('--loss_factor', type=float, default=1.0)
	parser.add_argument('--kd', action='store_true')
	parser.add_argument('--margin', type=float, default=0.1)

	


	return parser

def accuracy(preds, labels):
	return (preds == labels).mean()

def compute_metrics(output_mode, preds, labels):
	assert len(preds) == len(labels)
	
	if output_mode == 'classification':
		return {'accuracy': (preds==labels).mean()}
	else:
		return {'pearson': pearsonr(preds, labels)[0]}

def get_bert_param_groups(model, args):
	no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
	if args.df:
		group1 = ['layer.0', 'layer.1.'] 
		group2 = ['layer.2', 'layer.3']
		group3 = ['layer.4', 'layer.5'] 
		group4 = ['layer.6', 'layer.7']
		group5 = ['layer.8', 'layer.9']
		group6 = ['layer.10', 'layer.11']
		group_all = ['layer.0', 'layer.1.', 'layer.2', 'layer.3', 'layer.4', 'layer.5', \
		'layer.6', 'layer.7', 'layer.8', 'layer.9', 'layer.10', 'layer.11']
		optimizer_grouped_parameters = [
			{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)], \
			'weight_decay': args.weight_decay},
			{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group1)], \
			'weight_decay': args.weight_decay, 'lr': args.learning_rate/2**5},
			{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group2)], \
			'weight_decay': args.weight_decay, 'lr': args.learning_rate/2**4},
			{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group3)], \
			'weight_decay': args.weight_decay, 'lr': args.learning_rate/2**3},
			{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group4)], \
			'weight_decay': args.weight_decay, 'lr': args.learning_rate/2**2},
			{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group5)], \
			'weight_decay': args.weight_decay, 'lr': args.learning_rate/2},
			{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group6)], \
			'weight_decay': args.weight_decay, 'lr': args.learning_rate},

			{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)], \
			'weight_decay': 0.0},
			{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group1)], \
			'weight_decay': 0.0, 'lr': args.learning_rate/2**5},
			{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group2)], \
			'weight_decay': 0.0, 'lr': args.learning_rate/2**4},
			{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group3)], \
			'weight_decay': 0.0, 'lr': args.learning_rate/2**3},
			{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group4)], \
			'weight_decay': 0.0, 'lr': args.learning_rate/2**2},
			{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group5)], \
			'weight_decay': 0.0, 'lr': args.learning_rate/2},
			{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group6)], \
			'weight_decay': 0.0, 'lr': args.learning_rate},
		]
	else:
		optimizer_grouped_parameters = [
		{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
		{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
		]            
	return optimizer_grouped_parameters

def set_seed(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.n_gpu > 0:
		torch.cuda.manual_seed_all(args.seed)

def load_and_cache_examples(args, task, tokenizer, evaluate=False, reverse=True):
	processor = processors[task]()
	output_mode = output_modes[task]
	cached_features_file = os.path.join(args.data_dir, 'cached_{}'.format('dev' if evaluate else 'train'))
	if os.path.exists(cached_features_file) and not args.overwrite_cache:
		features = torch.load(cached_features_file)
	else:
		label_list = processor.get_labels()
		examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
		
		##this is where you double reverse
		if reverse:
			examples_reverse = [InputExample(pid=example.pid, text_a=example.text_b, text_b=example.text_a, aug_a=example.aug_a, aug_b=example.aug_b, label=example.label, label_soft=example.label_soft) for example in examples]
		else:
			examples_reverse = []
		features = convert_examples_to_features(examples+examples_reverse, label_list, args.max_seq_length, tokenizer, output_mode, args.augment)
		torch.save(features, cached_features_file)

	all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
	all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
	all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
	all_pids = torch.tensor([int(f.pid) for f in features], dtype=torch.long)

	all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
	if features[0].label_soft_id:
		try:
			all_label_soft_ids = torch.tensor([f.label_soft_id for f in features], dtype=torch.float)
		except:
			print([f.label_soft_id for f in features])

		dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_pids, all_label_soft_ids)
	else:
		dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_pids)

	return dataset

def load_graph_examples(args, evaluate=False, reverse=True):
	data_dir = args.data_dir
	if evaluate:
		with open(os.path.join(data_dir, 'dev_dataset_a.p'), 'rb') as f:
			dataset_a = pickle.load(f)
		with open(os.path.join(data_dir, 'dev_dataset_b.p'), 'rb') as f:
			dataset_b = pickle.load(f)
		dataloader_a = GraphDataLoader(dataset_a, batch_size=args.eval_batch_size)
		dataloader_b = GraphDataLoader(dataset_b, batch_size=args.eval_batch_size)
	else:
		with open(os.path.join(data_dir, 'train_dataset_a.p'), 'rb') as f:
			dataset_a = pickle.load(f)
		with open(os.path.join(data_dir, 'train_dataset_b.p'), 'rb') as f:
			dataset_b = pickle.load(f)
		if reverse:
			dataloader_a = GraphDataLoader(dataset_a+dataset_b, batch_size=args.train_batch_size)
			dataloader_b = GraphDataLoader(dataset_b+dataset_a, batch_size=args.train_batch_size)
		else:
			dataloader_a = GraphDataLoader(dataset_a, batch_size=args.train_batch_size)
			dataloader_b = GraphDataLoader(dataset_b, batch_size=args.train_batch_size)

	return dataloader_a, dataloader_b

def bounded_kd_loss(logits, soft_target, target, margin=0.1):
	if torch.mean(abs(logits - target)) + margin > torch.mean(abs(soft_target - target)):
		ret = torch.mean(abs(logits - target))
	else:
		ret = 0
	return ret





def generate_errors(args, all_preds, out_label_ids, pids, data='dev'):
	d = {'pred': all_preds, 'label': out_label_ids, 'pid': pids}
	df = pd.DataFrame.from_dict(d)
	df['error'] = df['label'] - df['pred']
	df['abs_error'] = abs(df['error'])

	dev_data = os.path.join(args.data_dir, '{}.jsonl'.format(data))
	lines = []
	with open(dev_data, 'r') as f:
		for line in f:
			lines.append(json.loads(line))

	sent1s = []
	sent2s = []
	for p in list(df.pid):
		for ex in lines:
			if int(ex['pid']) == p:
				sent1, sent2 = ex['sentence1'], ex['sentence2']
				sent1s.append(sent1)
				sent2s.append(sent2)
	df['sent1'] = pd.Series(sent1s)
	df['sent2'] = pd.Series(sent2s)
	df = df.sort_values(by='abs_error', ascending=False)              
	return df
