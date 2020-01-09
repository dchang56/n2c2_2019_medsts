import numpy as np
import random
import argparse
import logging
from utils import *
from data_utils import *
import torch
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
from pytorch_transformers import (WEIGHTS_NAME, BertConfig, BertForSequenceClassification, BertModel, BertTokenizer)
from pytorch_transformers import AdamW, WarmupLinearSchedule, WarmupCosineWithHardRestartsSchedule
from model_ln import *

import warnings
warnings.filterwarnings("ignore")
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', level = logging.INFO)
logger = logging.getLogger(__name__)

def train(args, train_dataset, model, tokenizer):
	tb_writer = SummaryWriter()
	args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
	## DATALOADER
	train_sampler = SequentialSampler(train_dataset)
	train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
	graph_train_dataloader_a, graph_train_dataloader_b = load_graph_examples(args)
	args.logging_steps = len(train_dataloader)
	args.save_steps = len(train_dataloader)
	if args.max_steps > 0:
		t_total = args.max_steps
		args.num_total_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
	else:
		t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
	assert len(train_dataset) == len(graph_train_dataloader_a.dataset)
	logger.info("***** Running training *****")
	logger.info("  Num examples = %d", len(train_dataset))
	logger.info("  Num Epochs = %d", args.num_train_epochs)
	logger.info("  Batch size per GPU = %d", args.per_gpu_train_batch_size)
	logger.info("  Total train batch size = %d", args.train_batch_size * args.gradient_accumulation_steps)
	logger.info("  Total optimization steps = %d", t_total)



	graph_optimizer = AdamW(model.graph_encoder.parameters(), lr=args.graph_lr, weight_decay=args.weight_decay)

	linear_optimizer = AdamW(model.classifier.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
	linear_c_optimizer = AdamW(model.classifier_c.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
	linear_type_optimizer = AdamW(model.classifier_type.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

	# bert_optimizer_grouped_parameters = get_bert_param_groups(model, args)
	bert_optimizer_grouped_parameters = get_bert_param_groups(model.text_encoder, args)
	bert_optimizer = AdamW(bert_optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, weight_decay=args.weight_decay)
	if args.scheduler == 'linear':
		scheduler = WarmupLinearSchedule(bert_optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
		graph_scheduler = WarmupLinearSchedule(graph_optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
		linear_scheduler = WarmupLinearSchedule(linear_optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
		linear_c_scheduler = WarmupLinearSchedule(linear_c_optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
		linear_type_scheduler = WarmupLinearSchedule(linear_type_optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
	elif args.scheduler == 'cosine':
		scheduler = WarmupCosineWithHardRestartsSchedule(bert_optimizer, warmup_steps=args.warmup_steps, t_total=t_total, cycles=2.)
		graph_scheduler = WarmupCosineWithHardRestartsSchedule(graph_optimizer, warmup_steps=args.warmup_steps, t_total=t_total, cycles=2.)
		linear_scheduler = WarmupCosineWithHardRestartsSchedule(linear_optimizer, warmup_steps=args.warmup_steps, t_total=t_total, cycles=2.)
		linear_c_scheduler = WarmupCosineWithHardRestartsSchedule(linear_c_optimizer, warmup_steps=args.warmup_steps, t_total=t_total, cycles=2.)
		linear_type_scheduler = WarmupCosineWithHardRestartsSchedule(linear_type_optimizer, warmup_steps=args.warmup_steps, t_total=t_total, cycles=2.)

	## TRAIN
	global_step = 0
	tr_loss, logging_loss = 0.0, 0.0
	model.zero_grad()
	set_seed(args)
	for _ in trange(int(args.num_train_epochs), desc='Epoch'):
		for batch, data_a, data_b in tqdm(zip(train_dataloader, graph_train_dataloader_a, graph_train_dataloader_b), desc='Iteration', total=len(train_dataloader)):
			model.train()
			batch = tuple(t.to(args.device) for t in batch)
			data_a, data_b = data_a.to(args.device), data_b.to(args.device)

			loss_fcts = {'mse': F.mse_loss, 'smooth_l1': F.smooth_l1_loss, 'l1': F.l1_loss}
			loss_fct = loss_fcts[args.loss_fct]	
			loss_combined = 0.0
			for mode in ['medsts', 'medsts_c', 'medsts_type']:
				torch.cuda.empty_cache()
				logits = model(batch[0], batch[1], batch[2], data_a, data_b, mode=mode)
				if mode=='medsts':
					loss = loss_fct(logits, data_a.label)
				elif mode=='medsts_c':
					loss = F.cross_entropy(logits, data_a.label_c)
				elif mode=='medsts_type':
					loss = F.cross_entropy(logits, data_a.label_type)
				loss_combined += loss

			loss_combined.backward()
			if args.clip == 'all':
				torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
			else:
				torch.nn.utils.clip_grad_norm_(model.text_encoder.parameters(), args.max_grad_norm)
			

			tr_loss += loss.item()
			scheduler.step()
			bert_optimizer.step()

			graph_scheduler.step()
			linear_scheduler.step()
			graph_optimizer.step()

			# print('learning rate: {} \t graph optimizer lr: {}'.format(linear_optimizer.param_groups[0]['lr'], graph_optimizer.param_groups[0]['lr']))
			linear_optimizer.step()

			linear_c_scheduler.step()
			linear_type_scheduler.step()
			linear_c_optimizer.step()
			linear_type_optimizer.step()


			model.zero_grad()
			global_step += 1
			args.logging_steps = len(train_dataloader) // 4

			if args.logging_steps > 0 and global_step % args.logging_steps == 0:
				result = evaluate(args, model, tokenizer)
				tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
				tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
				logging_loss = tr_loss

			if args.save_steps > 0 and global_step % args.save_steps == 0:
				output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
				if not os.path.exists(output_dir):
					os.makedirs(output_dir)
				# model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
				# model_to_save.save_pretrained(output_dir)
				torch.save(args, os.path.join(output_dir, 'training_args.bin'))
				# logger.info("Saving model checkpoint to %s", output_dir)
				
		# result = evaluate(args, model, tokenizer)
	tb_writer.close()
	return global_step, tr_loss / global_step

def evaluate(args, model, tokenizer, prefix=""):
	eval_task = args.task_name
	eval_output_dir = args.output_dir
	results = {}
	eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)
	if not os.path.exists(eval_output_dir):
		os.makedirs(eval_output_dir)
	args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
	eval_sampler = SequentialSampler(eval_dataset)
	eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, drop_last=True)
	eval_graph_dataloader_a, eval_graph_dataloader_b = load_graph_examples(args, evaluate=True)

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
			logits = model(batch[0], batch[1], batch[2], data_a, data_b, mode='medsts')
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
	errors.to_csv(os.path.join(args.output_dir, 'errors.csv'), index=False)

	result = compute_metrics('regression', preds, out_label_ids)
	results.update(result)

	output_eval_file = os.path.join(eval_output_dir, 'eval_results.txt')
	with open(output_eval_file, 'a') as writer:
		logger.info("***** Eval results {} *****".format(prefix))
		for key in sorted(result.keys()):
			logger.info("  %s = %s", key, str(result[key]))
			writer.write("%s = %s\n" % (key, str(result[key])))
	return results


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
	num_labels = len(label_list)

	##Load Models
	config = BertConfig.from_pretrained(args.config_name)
	tokenizer = BertTokenizer.from_pretrained(args.text_encoder_checkpoint, do_lower_case=args.do_lower_case)
	text_encoder = BertModel.from_pretrained(args.text_encoder_checkpoint, config=config)
	graph_encoder = GraphEncoder(args.n_hidden, args.min_score)
	if args.graph_encoder_checkpoint:
		graph_encoder.gcnnet.load_state_dict(torch.load(args.graph_encoder_checkpoint))

	medsts_classifier = PairClassifier(config.hidden_size+args.n_hidden, 1)
	medsts_c_classifier = PairClassifier(config.hidden_size+args.n_hidden, 5)
	medsts_type_classifier = PairClassifier(config.hidden_size+args.n_hidden, 4)
	model = MedstsNet(text_encoder, graph_encoder, medsts_classifier, medsts_c_classifier, medsts_type_classifier)
	model.to(args.device)

	args.n_gpu = 1

	if args.do_train:
		train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
		global_step, tr_loss = train(args, train_dataset, model, tokenizer)
		logger.info('global step = {}, average loss = {}'.format(global_step, tr_loss))
		if not os.path.exists(args.output_dir):
			os.makedirs(args.output_dir)
		logger.info("saving model checkpoint to {}".format(args.output_dir))
		model_to_save = model.module if hasattr(model, 'module') else model
		# model_to_save.save_pretrained(args.output_dir)
		torch.save(model_to_save.state_dict(), os.path.join(args.output_dir, 'saved_model.pth'))
		tokenizer.save_pretrained(args.output_dir)
		torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

if __name__ == "__main__":
	main()





























