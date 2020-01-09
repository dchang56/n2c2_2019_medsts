import os
import pandas as pd
from scipy.stats import pearsonr
import numpy as np
import json
import torch.nn.functional as F
import torch
import random
from tqdm import tqdm
from model import *
from utils import *
from data_utils import *
import torch
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)
from tqdm import tqdm, trange
from pytorch_transformers import (WEIGHTS_NAME, BertConfig, BertForSequenceClassification, BertModel, BertTokenizer)

import argparse

def predict(args, model, tokenizer, dataset, data='train'):
	task_name = args.task_name
	batch_size = 8
	args.train_batch_size = 8
	sampler = SequentialSampler(dataset)
	dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, drop_last=True)
	graph_dataloader_a, graph_dataloader_b = load_graph_examples(args, evaluate=True if data=='dev' else False, reverse=False)

	preds = None
	out_label_ids = None
	pids = None
	
	for batch, data_a, data_b in tqdm(zip(dataloader, graph_dataloader_a, graph_dataloader_b), desc='Predicting', total=len(dataloader)):
		model.eval()
		batch = tuple(t.to(args.device) for t in batch)
		data_a, data_b = data_a.to(args.device), data_b.to(args.device)
		with torch.no_grad():
			logits = model(batch[0], batch[1], batch[2], data_a, data_b, task_name, args.do_layernorm)

		if preds is None:
			preds = logits.detach().cpu().numpy()
			out_label_ids = data_a.label.detach().cpu().numpy()
			pids = batch[4].detach().cpu().numpy()
		else:
			preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
			out_label_ids = np.append(out_label_ids, data_a.label.detach().cpu().numpy(), axis=0)
			pids = np.append(pids, batch[4].detach().cpu().numpy(), axis=0)
	preds = np.squeeze(preds)
	errors = generate_errors(args, preds, out_label_ids, pids, data=data)
	result = compute_metrics('regression', preds, out_label_ids)
	
	return result, errors


def get_logits(args, checkpoint_path):
	processor = processors[args.task_name]()
	set_seed(args)
	label_list = processor.get_labels()
	num_labels = len(label_list)

	sd = torch.load(os.path.join(checkpoint_path, 'saved_model.pth'))
	graph_encoder_weights = {}
	classifier_weights = {}
	classifier_c_weights = {}
	classifier_c2_weights = {}
	classifier_type_weights = {}
	for k, v in sd.items():
		if k.startswith('graph'):
			graph_encoder_weights[k.replace('graph_encoder.','')] = v
		if k.startswith('classifier.'):
			classifier_weights[k.replace('classifier.', '')] = v
		if k.startswith('classifier_c.'):
			classifier_c_weights[k.replace('classifier_c.', '')] = v
		if k.startswith('classifier_c2.'):
			classifier_c2_weights[k.replace('classifier_c2.', '')] = v
		if k.startswith('classifier_type.'):
			classifier_type_weights[k.replace('classifier_type.','')] = v

	#load models
	config = BertConfig.from_pretrained(checkpoint_path)
	tokenizer = BertTokenizer.from_pretrained(checkpoint_path, do_lower_case=args.do_lower_case)
	text_encoder = BertModel.from_pretrained(checkpoint_path, config=config)

	graph_encoder = GraphEncoder(args.n_hidden, args.min_score)
	medsts_classifier = PairClassifier(config.hidden_size+args.n_hidden, 1)
	medsts_c_classifier = PairClassifier(config.hidden_size+args.n_hidden, 5)
	medsts_c2_classifier = PairClassifier(config.hidden_size+args.n_hidden, 2)
	medsts_type_classifier = PairClassifier(config.hidden_size+args.n_hidden, 4)

	graph_encoder.load_state_dict(graph_encoder_weights)
	medsts_classifier.load_state_dict(classifier_weights)
	medsts_c_classifier.load_state_dict(classifier_c_weights)
	medsts_c2_classifier.load_state_dict(classifier_c2_weights)
	medsts_type_classifier.load_state_dict(classifier_type_weights)

	model = MedstsNet(text_encoder, graph_encoder, medsts_classifier, medsts_c_classifier, medsts_c2_classifier, medsts_type_classifier)
	model.to(args.device);

	train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False, reverse=False)

	result, errors = predict(args, model, tokenizer, train_dataset, data='train')

	return errors

output_dir = '/home/dc925/project/medsts/output/'
clinicalbert_outputs = os.path.join(output_dir, 'clinicalbert_outputs')
ncbibert_outputs = os.path.join(output_dir, 'ncbi_bert_outputs')
mtdnnbase_outputs = os.path.join(output_dir, 'mt_dnn_base_outputs')
scibert_outputs = os.path.join(output_dir, 'scibert_outputs')

clinicalbert_runs = [os.path.join(clinicalbert_outputs, r) for r in os.listdir(clinicalbert_outputs) if r.startswith('run')]
ncbibert_runs = [os.path.join(ncbibert_outputs, r) for r in os.listdir(ncbibert_outputs) if r.startswith('run')]
mtdnnbase_runs = [os.path.join(mtdnnbase_outputs, r) for r in os.listdir(mtdnnbase_outputs) if r.startswith('run')]
scibert_runs = [os.path.join(scibert_outputs, r) for r in os.listdir(scibert_outputs) if r.startswith('run')]

output_dir2 = '/home/dc925/project/medsts/output/first_batch/'
clinicalbert_outputs_2 = os.path.join(output_dir2, 'clinicalbert_outputs')
ncbibert_outputs_2 = os.path.join(output_dir2, 'ncbi_bert_outputs')
scibert_outputs_2 = os.path.join(output_dir2, 'scibert_outputs')
mtdnnbase_outputs_2 = os.path.join(output_dir2, 'mt_dnn_base_outputs')
clinicalbert_runs_2 = [os.path.join(clinicalbert_outputs_2, r) for r in os.listdir(clinicalbert_outputs_2) if r.startswith('run')]
ncbibert_runs_2 = [os.path.join(ncbibert_outputs_2, r) for r in os.listdir(ncbibert_outputs_2) if r.startswith('run')]
scibert_runs_2 = [os.path.join(scibert_outputs_2, r) for r in os.listdir(scibert_outputs_2) if r.startswith('run')]
mtdnnbase_runs_2 = [os.path.join(mtdnnbase_outputs_2, r) for r in os.listdir(mtdnnbase_outputs_2) if r.startswith('run')]

output_dir3 = '/home/dc925/project/medsts/output/second_batch/'
clinicalbert_outputs_3 = os.path.join(output_dir3, 'clinicalbert_outputs')
ncbibert_outputs_3 = os.path.join(output_dir3, 'ncbi_bert_outputs')
scibert_outputs_3 = os.path.join(output_dir3, 'scibert_outputs')
mtdnnbase_outputs_3 = os.path.join(output_dir3, 'mt_dnn_base_outputs')
clinicalbert_runs_3 = [os.path.join(clinicalbert_outputs_3, r) for r in os.listdir(clinicalbert_outputs_3) if r.startswith('run')]
ncbibert_runs_3 = [os.path.join(ncbibert_outputs_3, r) for r in os.listdir(ncbibert_outputs_3) if r.startswith('run')]
scibert_runs_3 = [os.path.join(scibert_outputs_3, r) for r in os.listdir(scibert_outputs_3) if r.startswith('run')]
mtdnnbase_runs_3 = [os.path.join(mtdnnbase_outputs_3, r) for r in os.listdir(mtdnnbase_outputs_3) if r.startswith('run')]


output_dir4 = '/home/dc925/project/medsts/output/final_kd_1/'
clinicalbert_outputs_4 = os.path.join(output_dir4, 'clinicalbert_outputs')
ncbibert_outputs_4 = os.path.join(output_dir4, 'ncbi_bert_outputs')
scibert_outputs_4 = os.path.join(output_dir4, 'scibert_outputs')
mtdnnbase_outputs_4 = os.path.join(output_dir4, 'mt_dnn_base_outputs')
clinicalbert_runs_4 = [os.path.join(clinicalbert_outputs_4, r) for r in os.listdir(clinicalbert_outputs_4) if r.startswith('run')]
ncbibert_runs_4 = [os.path.join(ncbibert_outputs_4, r) for r in os.listdir(ncbibert_outputs_4) if r.startswith('run')]
scibert_runs_4 = [os.path.join(scibert_outputs_4, r) for r in os.listdir(scibert_outputs_4) if r.startswith('run')]
mtdnnbase_runs_4 = [os.path.join(mtdnnbase_outputs_4, r) for r in os.listdir(mtdnnbase_outputs_4) if r.startswith('run')]

output_dir5 = '/home/dc925/project/medsts/output/'
clinicalbert_outputs_5 = os.path.join(output_dir5, 'clinicalbert_aux_outputs')
ncbibert_outputs_5 = os.path.join(output_dir5, 'ncbi_bert_aux_outputs')
scibert_outputs_5 = os.path.join(output_dir5, 'scibert_aux_outputs')
mtdnnbase_outputs_5 = os.path.join(output_dir5, 'mt_dnn_base_aux_outputs')
clinicalbert_runs_5 = [os.path.join(clinicalbert_outputs_5, r) for r in os.listdir(clinicalbert_outputs_5) if r.startswith('run') and int(r.split('_')[-1])<8]
ncbibert_runs_5 = [os.path.join(ncbibert_outputs_5, r) for r in os.listdir(ncbibert_outputs_5) if r.startswith('run') and int(r.split('_')[-1])<8]
scibert_runs_5 = [os.path.join(scibert_outputs_5, r) for r in os.listdir(scibert_outputs_5) if r.startswith('run') and int(r.split('_')[-1])<8]
mtdnnbase_runs_5 = [os.path.join(mtdnnbase_outputs_5, r) for r in os.listdir(mtdnnbase_outputs_5) if r.startswith('run') and int(r.split('_')[-1])<8]

all_runs = ncbibert_runs + clinicalbert_runs + mtdnnbase_runs + scibert_runs + \
	clinicalbert_runs_2 + ncbibert_runs_2 + scibert_runs_2 + mtdnnbase_runs_2 + \
	clinicalbert_runs_3 + ncbibert_runs_3 + scibert_runs_3 + mtdnnbase_runs_3 + \
	clinicalbert_runs_4 + ncbibert_runs_4 + scibert_runs_4 + mtdnnbase_runs_4 +\
	clinicalbert_runs_5 + ncbibert_runs_5 + scibert_runs_5 + mtdnnbase_runs_5

parser = argparse.ArgumentParser()
parser.add_argument("--k", type=int, required=True)
args = parser.parse_args()
folds = [args.k]
kfold_model_outputs = []
for k in folds:
	print('starting')
	outputs = [os.path.join(r, 'k_fold_{}/checkpoint'.format(k)) for r in all_runs]
	for i, checkpoint_path in enumerate(tqdm(outputs)):
		args = torch.load(os.path.join(checkpoint_path, 'training_args.bin'))
		errors = get_logits(args, checkpoint_path)
		errors = errors.sort_values(by='pid').set_index('pid')
		if i==0:
			df = pd.concat([errors['label'], errors['sent1'], errors['sent2']], axis=1)
		df['pred{}'.format(i)] = errors['pred']
	df.to_csv('/home/dc925/project/medsts/logits/fold_{}_logits.csv'.format(k))
	print('fold {} done'.format(k))

























