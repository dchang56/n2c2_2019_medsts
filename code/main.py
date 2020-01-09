import numpy as np
import random
import argparse
import logging
from utils import *
from data_utils import *
import torch
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)
from tqdm import tqdm, trange
from pytorch_transformers import (WEIGHTS_NAME, BertConfig, BertForSequenceClassification, BertModel, BertTokenizer)
from pytorch_transformers import AdamW, WarmupLinearSchedule, WarmupCosineWithHardRestartsSchedule, WarmupCosineSchedule
from model import *

import warnings
warnings.filterwarnings("ignore")
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', level = logging.INFO)
logger = logging.getLogger(__name__)

def train(args, train_dataset, model, tokenizer):
	args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
	## DATALOADER
	train_sampler = SequentialSampler(train_dataset)
	train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
	graph_train_dataloader_a, graph_train_dataloader_b = load_graph_examples(args, reverse=True)
	t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
	assert len(train_dataset) == len(graph_train_dataloader_a.dataset)
	logger.info("***** Running training *****")
	logger.info("  Num examples = %d", len(train_dataset))
	logger.info("  Num Epochs = %d", args.num_train_epochs)
	logger.info("  Batch size per GPU = %d", args.per_gpu_train_batch_size)
	logger.info("  Total train batch size = %d", args.train_batch_size * args.gradient_accumulation_steps)
	logger.info("  Total optimization steps = %d", t_total)

	## OPTIMIZERS AND SCHEDULERS
	if not args.text_only:
		graph_optimizer = AdamW(model.graph_encoder.parameters(), lr=args.graph_lr, weight_decay=args.weight_decay)
	linear_optimizer = AdamW(model.classifier.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
	linear_c_optimizer = AdamW(model.classifier_c.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
	linear_c2_optimizer = AdamW(model.classifier_c2.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
	linear_type_optimizer = AdamW(model.classifier_type.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
	bert_optimizer_grouped_parameters = get_bert_param_groups(model.text_encoder, args)

	bert_optimizer = AdamW(bert_optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, weight_decay=args.weight_decay)
	bert_scheduler = WarmupLinearSchedule(bert_optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
	if not args.text_only:
		graph_scheduler = WarmupLinearSchedule(graph_optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
	linear_scheduler = WarmupLinearSchedule(linear_optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
	linear_c_scheduler = WarmupLinearSchedule(linear_c_optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
	linear_c2_scheduler = WarmupLinearSchedule(linear_c2_optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
	linear_type_scheduler = WarmupLinearSchedule(linear_type_optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

	## TRAIN
	global_step = 0
	best_val = 0.
	tr_loss, logging_loss = 0.0, 0.0
	model.zero_grad()
	set_seed(args)
	print('layernorm on: {}'.format(args.do_layernorm))
	for _ in trange(int(args.num_train_epochs), desc='Epoch'):
		for batch, data_a, data_b in tqdm(zip(train_dataloader, graph_train_dataloader_a, graph_train_dataloader_b), desc='Iteration', total=len(train_dataloader)):
			model.train()
			batch = tuple(t.to(args.device) for t in batch)
			data_a, data_b = data_a.to(args.device), data_b.to(args.device)

			loss_combined = 0.0
			if args.aux=='all':
				modes = ['medsts', 'medsts_c', 'medsts_c2', 'medsts_type']
			elif args.aux=='medsts_c':
				modes = ['medsts', 'medsts_c']
			elif args.aux=='medsts_c2':
				modes = ['medsts', 'medsts_c2']
			elif args.aux=='medsts_type':
				modes = ['medsts', 'medsts_type']
			else:
				modes = ['medsts']
			for mode in modes:
				# torch.cuda.empty_cache()
				logits = model(batch[0], batch[1], batch[2], data_a, data_b, mode, args.do_layernorm)
				if mode=='medsts':
					if args.kd:
						loss = F.mse_loss(logits, data_a.label) + args.loss_factor * bounded_kd_loss(logits, batch[5], data_a.label, margin=args.margin)
					else:
						loss = F.mse_loss(logits, data_a.label)
				elif mode=='medsts_c':
					loss = F.cross_entropy(logits, data_a.label_c)
				elif mode=='medsts_c2':
					loss = F.cross_entropy(logits, data_a.label_c2)
				elif mode=='medsts_type':
					loss = F.cross_entropy(logits, data_a.label_type)
				loss_combined += loss
			loss_combined.backward()
			
			torch.nn.utils.clip_grad_norm_(model.text_encoder.parameters(), args.max_grad_norm)
			
			tr_loss += loss.item()

			bert_scheduler.step()
			if not args.text_only:
				graph_scheduler.step()
			linear_scheduler.step()
			linear_c_scheduler.step()
			linear_c2_scheduler.step()
			linear_type_scheduler.step()

			bert_optimizer.step()
			if not args.text_only:
				graph_optimizer.step()
			linear_optimizer.step()
			linear_c_optimizer.step()
			linear_c2_optimizer.step()
			linear_type_optimizer.step()

			model.zero_grad()
			global_step += 1

			args.logging_steps = len(train_dataloader) // 4



			if global_step % args.logging_steps == 0:
				if args.do_eval:
					result, errors = evaluate(args, model, tokenizer)
					if result['pearson'] > best_val:
						best_val = result['pearson']
						output_dir = os.path.join(args.output_dir, 'checkpoint')
						if not os.path.exists(output_dir):
							os.makedirs(output_dir)
						logger.info("Saving checkpoint with acc {} at {}".format(best_val, output_dir))
						torch.save(args, os.path.join(output_dir, 'training_args.bin'))
						model_to_save = model.module if hasattr(model, 'module') else model
						torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'saved_model.pth'))
						tokenizer.save_pretrained(output_dir)
						#saving text encoder
						model.text_encoder.save_pretrained(output_dir)
						#error analysis
						errors.to_csv(os.path.join(output_dir, 'errors.csv'), index=False)
						#save results.txt
						best_eval_file = os.path.join(output_dir, 'eval_results.txt')
						with open(best_eval_file, 'a') as writer:
							logger.info("***** Eval results *****")
							for key in sorted(result.keys()):
								logger.info("  %s = %s", key, str(result[key]))
								writer.write("{} = {} \t step {} \n".format(key, str(result[key]), global_step))
				else:
					if global_step == t_total - 5:
						print('@@@@@@@@@@@@@@@@@@@@@@@@@ saving checkpoint @@@@@@@@@@@@@@@@@@@@@@')
						output_dir = os.path.join(args.output_dir, 'checkpoint_{}'.format(global_step))
						if not os.path.exists(output_dir):
							os.makedirs(output_dir)
						logger.info("Saving checkpoint at {}".format(output_dir))
						model.text_encoder.save_pretrained(output_dir)
						torch.save(args, os.path.join(output_dir, 'training_args.bin'))
						model_to_save = model.module if hasattr(model, 'module') else model
						torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'saved_model.pth'))
						tokenizer.save_pretrained(output_dir)


				logging_loss = tr_loss              
	return global_step, tr_loss / global_step

def evaluate(args, model, tokenizer, prefix=""):
	eval_task = args.task_name
	eval_output_dir = args.output_dir
	results = {}
	eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True, reverse=False)
	if not os.path.exists(eval_output_dir):
		os.makedirs(eval_output_dir)
	args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
	eval_sampler = SequentialSampler(eval_dataset)
	eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, drop_last=True)
	eval_graph_dataloader_a, eval_graph_dataloader_b = load_graph_examples(args, evaluate=True, reverse=False)

	## EVAL
	logger.info("***** Running evaluation {} *****".format(prefix))
	logger.info("  Num examples = %d", len(eval_dataset))
	logger.info("  Batch size = %d", args.eval_batch_size)
	eval_loss, nb_eval_steps = 0.0, 0
	preds = None
	out_label_ids = None
	pids = None

	for batch, data_a, data_b in tqdm(zip(eval_dataloader, eval_graph_dataloader_a, eval_graph_dataloader_b), desc='Evaluating', total=len(eval_dataloader)):
		model.eval()
		batch = tuple(t.to(args.device) for t in batch)
		data_a, data_b = data_a.to(args.device), data_b.to(args.device)
		with torch.no_grad():
			logits = model(batch[0], batch[1], batch[2], data_a, data_b, 'medsts', args.do_layernorm)
			tmp_eval_loss = F.mse_loss(logits, data_a.label)
			eval_loss += tmp_eval_loss.mean().item()
		nb_eval_steps += 1
		if preds is None:
			preds = logits.detach().cpu().numpy()
			out_label_ids = data_a.label.detach().cpu().numpy()
			pids = batch[4].detach().cpu().numpy()
		else:
			preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
			out_label_ids = np.append(out_label_ids, data_a.label.detach().cpu().numpy(), axis=0)
			pids = np.append(pids, batch[4].detach().cpu().numpy(), axis=0)
	eval_loss = eval_loss / nb_eval_steps
	if args.output_mode == 'classification':
		preds = np.argmax(preds, axis=1)
	elif args.output_mode == 'regression':
		preds = np.squeeze(preds)
	errors = generate_errors(args, preds, out_label_ids, pids)
	result = compute_metrics('regression', preds, out_label_ids)
	results.update(result)

	output_eval_file = os.path.join(eval_output_dir, 'eval_results.txt')
	with open(output_eval_file, 'a') as writer:
		# logger.info("***** Eval results {} *****".format(prefix))
		for key in sorted(result.keys()):
			# logger.info("  %s = %s", key, str(result[key]))
			writer.write("%s = %s\n" % (key, str(result[key])))
	return results, errors


def main():
	torch.cuda.empty_cache()
	parser = setup_parser()
	args = parser.parse_args()
	if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
		raise ValueError("Output directory already exists and is not empty.")
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	args.n_gpu = torch.cuda.device_count()
	args.device = device
	set_seed(args)
	args.task_name = args.task_name.lower()
	if args.task_name not in processors:
		raise ValueError("Task not found: {}".format(args.task_name))
	processor = processors[args.task_name]()
	args.output_mode = output_modes[args.task_name]
	label_list = processor.get_labels()

	##Load Models
	config = BertConfig.from_pretrained(args.config_name)
	tokenizer = BertTokenizer.from_pretrained(args.text_encoder_checkpoint, do_lower_case=args.do_lower_case)
	text_encoder = BertModel.from_pretrained(args.text_encoder_checkpoint, config=config)
	graph_encoder = GraphEncoder(args.n_hidden, args.min_score)

	medsts_classifier = PairClassifier(config.hidden_size+args.n_hidden, 1)
	medsts_c_classifier = PairClassifier(config.hidden_size+args.n_hidden, 5)
	medsts_c2_classifier = PairClassifier(config.hidden_size+args.n_hidden, 2)
	medsts_type_classifier = PairClassifier(config.hidden_size+args.n_hidden, 4)
	model = MedstsNet(text_encoder, graph_encoder, medsts_classifier, medsts_c_classifier, medsts_c2_classifier, medsts_type_classifier)
	if args.text_only:
		medsts_classifier = PairClassifier(config.hidden_size, 1)
		medsts_c_classifier = PairClassifier(config.hidden_size, 5)
		medsts_c2_classifier = PairClassifier(config.hidden_size, 2)
		medsts_type_classifier = PairClassifier(config.hidden_size, 4)
		model = MedstsNet_Textonly(text_encoder, medsts_classifier, medsts_c_classifier, medsts_c2_classifier, medsts_type_classifier)

	model.to(args.device)

	args.n_gpu = 1

	if args.do_train:
		train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False, reverse=True)
		global_step, tr_loss = train(args, train_dataset, model, tokenizer)
		logger.info('global step = {}, average loss = {}'.format(global_step, tr_loss))

if __name__ == "__main__":
	main()




























