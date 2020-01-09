import argparse
import os
import numpy as np
import csv
import random
import torch
import json
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from tqdm import tqdm, trange
from scipy.stats import pearsonr
from torch.nn import CrossEntropyLoss, MSELoss
from parallel import DataParallelModel, DataParallelCriterion, gather
import pandas as pd
from data_utils import processors, output_modes
import logging
import pickle
from torch_geometric.data import DataLoader as GraphDataLoader
from utils_glue import convert_examples_to_features

from itertools import cycle




class InputExample(object):
    def __init__(self, pid, text_a, text_b=None, aug_a=None, aug_b=None, label=None, label_c=None, label_type=None):
        self.pid = pid
        self.text_a = text_a
        self.text_b = text_b
        self.aug_a = aug_a
        self.aug_b = aug_b
        self.label = label
        self.label_c = label_c
        self.label_type = label_type

class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id, pid):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.pid = pid

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
            examples.append(
                InputExample(pid=pid, text_a=text_a, text_b=text_b, aug_a=aug_a, aug_b=aug_b, label=label))
        return examples











def setup_parser():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--data_dir", type=str, required=True)
    # parser.add_argument("--model_name_or_path", default=None, type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--text_encoder_checkpoint", type=str, required=True)
    parser.add_argument("--graph_encoder_checkpoint", type=str, required=False)

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


    ## For Graph Encoder
    parser.add_argument('--n_hidden', type=int, default=128)
    parser.add_argument('--min_score', type=float, default=0.01)
    parser.add_argument('--graph_lr', type=float, default=0.0015)

    #testing params
    parser.add_argument('--clip_all', action='store_true')
    parser.add_argument('--loss_fct', type=str, default='mse')
    parser.add_argument('--scheduler', type=str, default='linear')


    return parser

def do_eval(model, logger, output_dir, device, tr_loss, nb_tr_steps, global_step, processor, label_list, tokenizer, eval_dataloader, error_analysis_dict, output_mode, i, task):

    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    preds = []
    all_label_ids = []
    all_input_ids = []
    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc='Evaluating'):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, i, output_mode)

            if output_mode == 'classification':
                loss_fct = CrossEntropyLoss()
                loss_fct = DataParallelCriterion(loss_fct)
                logits = [logits[i].view(-1, logits[0].size(1))
                          for i in range(len(logits))]
                tmp_eval_loss = loss_fct(logits, label_ids.view(-1))
            else:
                loss_fct = MSELoss()
                loss_fct = DataParallelCriterion(loss_fct)
                logits = [logits[i].view(-1) for i in range(len(logits))]
                tmp_eval_loss = loss_fct(logits, label_ids.view(-1))

            logits = gather(logits, target_device='cuda:0')
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(
                    preds[0], logits.detach().cpu().numpy(), axis=0)
            if len(all_label_ids) == 0:
                all_label_ids.append(label_ids.detach().cpu().numpy())
            else:
                all_label_ids[0] = np.append(
                    all_label_ids[0], label_ids.detach().cpu().numpy(), axis=0)
            if len(all_input_ids) == 0:
                all_input_ids.append(input_ids.detach().cpu().numpy())
            else:
                all_input_ids[0] = np.append(
                    all_input_ids[0], input_ids.detach().cpu().numpy(), axis=0)

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    preds = preds[0]
    all_label_ids = all_label_ids[0]
    all_input_ids = all_input_ids[0]
    all_pids = error_analysis_dict['pids'][:len(preds)]
    all_text_a = error_analysis_dict['text_a'][:len(preds)]
    all_text_b = error_analysis_dict['text_b'][:len(preds)]
    
    all_textpair_tokenized = [' '.join(tokenizer.convert_ids_to_tokens(ids)) for ids in all_input_ids]
    
    assert len(preds) == len(all_label_ids) == len(all_input_ids) == len(all_pids) == len(all_text_a) == len(all_text_b) == len(all_textpair_tokenized)
    
    all_textpair_tokenized = [tp.replace('[PAD]','').strip() for tp in all_textpair_tokenized]

    if output_mode == 'classification':
        preds = np.argmax(preds, axis=1)
        preds_rounded = preds
        eval_accuracy = accuracy(preds, all_label_ids)
    else:
        preds = np.squeeze(preds)
        preds_rounded = np.round(preds*4)/4
        eval_accuracy = pearsonr(preds, all_label_ids)[0]
    errors = generate_errors(preds, preds_rounded, all_label_ids, all_pids, all_text_a, all_text_b, all_textpair_tokenized)
    if i == 0:
        errors.to_csv(os.path.join(output_dir, 'error_table.csv'),sep=',',index=False)

    result = {'task name': task,
              'eval_loss': eval_loss,
              'eval_accuracy': eval_accuracy,
              'global_step': global_step,
              'loss': tr_loss / nb_tr_steps}

    output_eval_file = os.path.join(output_dir, 'eval_results.txt')
    with open(output_eval_file, 'w') as writer:
        logger.info('******** Eval Results *****')
        for key in sorted(result.keys()):
            logger.info('   %s = %s', key, str(result[key]))
            # writer.write('{} = {}\n'.format(key, str(result[key])))

    return eval_accuracy


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

def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    processor = processors[task]()
    output_mode = output_modes[task]
    cached_features_file = os.path.join(args.data_dir, 'cached_{}'.format('dev' if evaluate else 'train'))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        # logger.info('Loading features from cached file {}'.format(cached_features_file))
        features = torch.load(cached_features_file)
    else:
        # logger.info('Creating features from dataset file at {}'.format(args.data_dir))
        label_list = processor.get_labels()
        examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        
        ##this is where you double reverse
        if evaluate:
            examples_reverse = []
        else:
            examples_reverse = [InputExample(pid=example.pid, text_a=example.text_b, text_b=example.text_a, aug_a=example.aug_a, aug_b=example.aug_b, label=example.label) for example in examples]
        features = convert_examples_to_features(examples+examples_reverse, label_list, args.max_seq_length, tokenizer, output_mode, cls_token_at_end=True, cls_token=tokenizer.cls_token, 
            sep_token=tokenizer.sep_token, cls_token_segment_id=2, pad_on_left=True, pad_token_segment_id=4, augment=args.augment)
        # logger.info('Saving features into cached file {}'.format(cached_features_file))
        torch.save(features, cached_features_file)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
    all_pids = torch.tensor([int(f.pid) for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_pids)
    return dataset

def load_graph_examples(args, evaluate=False):
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
        dataloader_a = GraphDataLoader(dataset_a+dataset_b, batch_size=args.train_batch_size)
        dataloader_b = GraphDataLoader(dataset_b+dataset_a, batch_size=args.train_batch_size)
    return dataloader_a, dataloader_b

def load_graph_encoder(path):
    sd = torch.load(path)
    new_sd = {}
    for k, v in sd.items():
        new_sd['gcnnet.'+k] = v
    return new_sd

def generate_errors(args, all_preds, out_label_ids, pids):
    d = {'pred': all_preds, 'label': out_label_ids, 'pid': pids}
    df = pd.DataFrame.from_dict(d)
    df['error'] = df['label'] - df['pred']
    df['abs_error'] = abs(df['error'])
    df = df.sort_values(by='abs_error', ascending=False)

    dev_data = os.path.join(args.data_dir, 'dev.jsonl')
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
    return df

