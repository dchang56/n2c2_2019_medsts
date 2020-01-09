import argparse
import csv
import json
import logging
import os
import random
import sys
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from tqdm import tqdm, trange
from scipy.stats import pearsonr
from torch.nn import CrossEntropyLoss, MSELoss
from pytorch_pretrained_bert.modeling import WEIGHTS_NAME, CONFIG_NAME
from parallel import DataParallelModel, DataParallelCriterion, gather
from data_utils import *
from utils import *
from modeling import BertForMultiTask, BertConfig
# from optimization import BERTAdam
from pytorch_transformers import AdamW, WarmupLinearSchedule
import tokenization
from itertools import cycle
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
					datefmt = '%m/%d/%Y %H:%M:%S',
					level = logging.INFO)
logger = logging.getLogger(__name__)


def main():
	parser = setup_parser()
	args = parser.parse_args()
	logger.info('@@@@@ START @@@@@')
	device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
	n_gpu = torch.cuda.device_count()
	logger.info('device %s n_gpu %d', device, n_gpu)
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if n_gpu > 0:
		torch.cuda.manual_seed_all(args.seed)
	if not args.do_train and not args.do_eval:
		raise ValueError("At least one of `do_train` or `do_eval` must be True.")
	args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

	bert_config = BertConfig.from_json_file(args.bert_config_file)
	if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
		raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
	os.makedirs(args.output_dir, exist_ok=True)

	if args.tasks == 'all':
		task_names = ['medsts', 'mednli']
		data_dirs = ['MEDSTS', 'MEDNLI']
	elif args.tasks == 'single':
		task_names = ['medsts', 'mednli']
		data_dirs = ['MEDSTS', 'MEDNLI']
		task_names = [task_names[int(args.target_task_id)]]
		data_dirs = [data_dirs[int(args.target_task_id)]]
	if args.k_fold:
		target_data_dir = data_dirs[args.target_task_id]
		k_fold_data_dir = target_data_dir + '/k_fold_{}'.format(args.k)
		data_dirs[args.target_task_id] = k_fold_data_dir
	# if args.add_medsts_c:
	# 	assert args.k_fold==True
	# 	task_names.append('medsts_c')
	# 	data_dirs.append('MEDSTS_c')
	# 	k_fold_data_dir = data_dirs[-1] + '/k_fold_{}'.format(args.k)
	# 	data_dirs[-1] = k_fold_data_dir

	if task_names[0] not in processors:
		raise ValueError('Task not found: {}'.format(task_names[0]))

	processor_list = [processors[task_name]() for task_name in task_names]
	label_list = [processor.get_labels() for processor in processor_list]

	tokenizer = tokenization.FullTokenizer(
		vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

	train_examples = None
	num_train_steps = None
	num_tasks = len(task_names)

	if args.do_train:
		train_examples = [processor.get_train_examples(args.data_dir + data_dir) for processor, data_dir in zip(processor_list, data_dirs)]
		num_train_steps = int(len(train_examples[0]) / args.train_batch_size * args.num_train_epochs)
		if args.tasks =='all':
			total_tr = args.tr_factor * num_tasks * int(args.num_train_epochs)
		else:
			total_tr = int(0.5 * num_train_steps)

	if args.tasks == 'all':
		steps_per_epoch = args.gradient_accumulation_steps * args.tr_factor * num_tasks
	else:
		steps_per_epoch = int(num_train_steps / (2. * args.num_train_epochs))
	bert_config.num_tasks = num_tasks
	bert_config.hidden_size_aug = int(args.h_aug)

	model = BertForMultiTask(bert_config, [len(labels) for labels in label_list])

	if args.init_checkpoint is not None:
		if args.multi:
			load_checkpoint_mult(args.init_checkpoint, model, args.same, args.tasks)
		else:
			model.bert.load_state_dict(torch.load(args.init_checkpoint, map_location='cpu'))

	if args.freeze:
		for n, p in model.bert.named_parameters():
			if 'aug' in n or 'classifier' in n or 'mult' in n or 'gamma' in n or 'beta' in n:
				continue
			p.requires_grad = False

	model.to(device)
	if n_gpu > 1:
		model = DataParallelModel(model)

	group_size = 2

	optimizer_parameters = get_param_groups(model, args, group_size)
	optimizer = BERTAdam(optimizer_parameters, lr=args.learning_rate, warmup=args.warmup_proportion, t_total=total_tr)

	if args.do_eval:
		eval_loaders = []
		error_analysis_dicts = []
		for i, task in enumerate(task_names):
			eval_examples = processor_list[i].get_dev_examples(args.data_dir + data_dirs[i])
			eval_features = convert_examples_to_features(eval_examples, label_list[i], args.max_seq_length, tokenizer, output_modes[task])
			all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
			all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
			all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
			all_pids = [int(f.pid) for f in eval_examples]
			all_text_a = [f.text_a for f in eval_examples]
			all_text_b = [f.text_b for f in eval_examples]
			error_data = {'pids': all_pids, 'text_a': all_text_a, 'text_b': all_text_b}

			if output_modes[task] == 'classification':
				all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
			else:
				all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float32)

			eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
			eval_sampler = SequentialSampler(eval_data)
			eval_loaders.append(DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size, drop_last=True))
			error_analysis_dicts.append(error_data)
		
	global_step = 0
	if args.do_train:
		loaders = []
		logger.info(' Num tasks = {}'.format(len(train_examples)))
		for i, task in enumerate(task_names):
			train_features = convert_examples_to_features(train_examples[i], label_list[i], args.max_seq_length, tokenizer, output_modes[task])
			logger.info('********* Training data for {}'.format(task))
			logger.info('   Data size = {}'.format(len(train_features)))

			all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
			all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
			all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
			if output_modes[task] == 'classification':
				all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
			else:
				all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float32)
			train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
			train_sampler = RandomSampler(train_data)
			loaders.append(iter(DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size, drop_last=True)))
		total_params = sum(p.numel() for p in model.parameters())
		logger.info(' Num param = {}'.format(total_params))
		loaders = [cycle(it) for it in loaders]
		
		model.train()
		best_target_score = 0.
		task_id = 0
		all_ev_acc = []

		for epoch in trange(int(args.num_train_epochs), desc='Epoch'):
			if args.sample == 'anneal':
				probs = [len(dataset) for dataset in train_examples]
				probs = anneal(probs, epoch, args.num_train_epochs, anneal_factor=0.8, target_task_id=0, weight=5)

			tr_loss = [0. for i in range(num_tasks)]
			nb_tr_examples, nb_tr_steps = 0, 0

			## DEBUG
			# steps_per_epoch = 5

			for step in trange(steps_per_epoch, desc='Steps'):
				if step % args.gradient_accumulation_steps == 0:
					task_id = np.random.choice(len(probs), p=probs)
					output_mode = output_modes[task_names[task_id]]
				batch = next(loaders[task_id])
				batch = tuple(t.to(device) for t in batch)
				input_ids, input_mask, segment_ids, label_ids = batch
				logits = model(input_ids, segment_ids, input_mask, task_id, output_mode)

				if output_mode == 'classification':
					loss_fct = CrossEntropyLoss()
					loss_fct = DataParallelCriterion(loss_fct)
					logits = [logits[i].view(-1, logits[0].size(-1)) for i in range(len(logits))]
					loss = loss_fct(logits, label_ids.view(-1))
				else:
					loss_fct = MSELoss()
					loss_fct = DataParallelCriterion(loss_fct)
					logits = [logits[i].view(-1) for i in range(len(logits))]
					loss = loss_fct(logits, label_ids.view(-1))
				if n_gpu > 1:
					loss = loss.mean()
				if args.gradient_accumulation_steps > 1:
					loss = loss / args.gradient_accumulation_steps
				loss.backward()
				tr_loss[task_id] += loss.item()
				nb_tr_examples += input_ids.size(0)
				nb_tr_steps += 1
				if (step + 1) % args.gradient_accumulation_steps == 0:
					optimizer.step()
					model.zero_grad()
					global_step += 1

				#this is where you'd calculate training acc
	
			ev_acc = []
			for i, task in enumerate(task_names):
				acc = do_eval(model, logger, args.output_dir, device, tr_loss[i], nb_tr_steps, global_step, processor_list[i],
					label_list[i], tokenizer, eval_loaders[i], error_analysis_dicts[i], output_modes[task], i, task)
				ev_acc.append(acc)
			all_ev_acc.append(ev_acc)
			# logger.info('Average acc: {}'.format(np.mean(ev_acc)))
			if ev_acc[args.target_task_id] > best_target_score:
				best_target_score = ev_acc[args.target_task_id]
				model_to_save = model.module if hasattr(model, 'module') else model
				output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
				output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
				torch.save(model_to_save.state_dict(), output_model_file)
				bert_config.to_json_file(output_config_file)
				tokenizer.save_vocabulary(args.output_dir)

				##TODO: this is where you should add error analysis to get best version


			logger.info('Best target acc: {}'.format(best_target_score))
		
		output_eval_file = os.path.join(args.output_dir, 'eval_results.txt')
		with open(output_eval_file, 'w') as writer:
		    logger.info('******** Eval Results ********')
		    for n, acc in enumerate(all_ev_acc):
		        logger.info('   {} = {}\n'.format(n, acc))
		        writer.write('{} \t {}\n'.format(n, acc))

if __name__ == "__main__":
	main()





























































