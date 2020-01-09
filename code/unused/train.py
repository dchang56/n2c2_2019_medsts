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
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from parallel import DataParallelModel, DataParallelCriterion
import parallel

import pandas as pd

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    def __init__(self, pid, text_a, text_b=None, label=None):
        self.pid = pid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
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
    def get_test_examples(self, data_dir):
        raise NotImplementedError()
    def get_labels(self, output_mode):
        if output_mode == 'classification':
            return ['contradiction', 'entailment', 'neutral']
        elif output_mode == 'regression':
            return [None]
        else:
            raise KeyError(output_mode)
    @classmethod
    def _read_jsonl(cls, input_file):
        """Reads a jsonl file"""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for jsonline in f:
                line = json.loads(jsonline)
                lines.append(line)
            return lines

    @classmethod
    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            pid = line['pid']
            text_a = line['sentence1']
            text_b = line['sentence2']
            label = line['label']
            examples.append(
                InputExample(pid=pid, text_a=text_a, text_b=text_b, label=label))
        return examples


class StsbProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "train.jsonl")), "train")
    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "dev.jsonl")), "dev")
    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "test.jsonl")), "test")

class MednliProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "train.jsonl")), "train")
    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "dev.jsonl")), "dev")
    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "test.jsonl")), "test")

class MedstsProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, 'train.jsonl')), "train")
    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, 'dev.jsonl')), "dev")

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_mode):
    """Loads a data file into a list of InputBatch's"""
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    max_len = 0
    for (ex_index, example) in enumerate(examples):
        # if ex_index % 10000 == 0:
        #     logger.info("Writing example {} of {}".format(ex_index, len(examples)))
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            seq_len = len(tokens_a) + len(tokens_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            seq_len = len(tokens_a)
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
        if seq_len > max_len:
            max_len = seq_len
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        pid = example.pid

        if ex_index < 0:
            logger.info("*** Example ***")
            logger.info("pid: %s" % (example.pid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))
        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          pid=pid))
    print('Max sequence length: {}'.format(max_len))
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

def simple_accuracy(preds, labels):
    return (preds == labels).mean()
def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {"acc": acc, "f1": f1, "acc_and_f1": (acc + f1) / 2}
def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearman": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2}
def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == 'mnli':
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == 'stsb' or task_name == 'medsts':
        return pearson_and_spearman(preds, labels)
    elif task_name == 'mednli':
        return {'acc': simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)

def generate_errors(preds, all_label_ids, all_pids):
    d = {'y_h':preds,'y':all_label_ids, 'id':all_pids}
    df = pd.DataFrame.from_dict(d)
    df['e'] = df['y']-df['y_h']
    df['abs_e'] = abs(df['e'])
    df = df.sort_values(by='abs_e',ascending=False)
    return df

    ##################################################################################################         
    ############################################ PARSER ##############################################         
    ##################################################################################################
def setup_parser():
    parser = argparse.ArgumentParser()
    ##Required Parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese, biobert.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ##Other Parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--model_loc', type=str, default='', help="Specify the location of the bio or clinical bert model")
    parser.add_argument('--discriminative_finetuning', action='store_true', help='Whether to use discriminative fine-tuning')
    return parser



def main():
    parser = setup_parser()
    args = parser.parse_args()

    processors = {
    'stsb': StsbProcessor,
    'mednli': MednliProcessor,
    'medsts': MedstsProcessor}

    output_modes = {
    'mnli': 'classification',
    'stsb': 'regression',
    'mednli': 'classification',
    'medsts': 'regression'}

    bert_types = {
    'discharge': '/home/dc925/project/data/clinicalbert/biobert_pretrain_output_disch_100000',
    'all': '/home/dc925/project/data/clinicalbert/biobert_pretrain_output_all_notes_150000',
    'base_uncased': 'bert-base-uncased',
    'base_cased': 'bert-base-cased'}

    ##################################################################################################
    ################################### SETUP DATA, DEVICE, MODEL ####################################
    ##################################################################################################
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        n_gpu = 1
        #Initialize the distributed backend which will take care of synchronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: {}".format(task_name))

    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    label_list = processor.get_labels(output_mode)
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))
    model = BertForSequenceClassification.from_pretrained(args.bert_model,
        cache_dir=cache_dir,
        num_labels=num_labels)
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        model = DDP(model)
    elif n_gpu > 1:
        # model = torch.nn.DataParallel(model)
        model = DataParallelModel(model)
    

    ##################################################################################################
    ########################################### OPTIMIZER ############################################
    ##################################################################################################

    if args.do_train:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        if args.discriminative_finetuning:
            group1 = ['layer.0', 'layer.1.'] 
            group2 = ['layer.2', 'layer.3']
            group3 = ['layer.4', 'layer.5'] 
            group4 = ['layer.6', 'layer.7']
            group5 = ['layer.8', 'layer.9']
            group6 = ['layer.10', 'layer.11']
            group_all = ['layer.0', 'layer.1.', 'layer.2', 'layer.3', 'layer.4', 'layer.5', \
            'layer.6', 'layer.7', 'layer.8', 'layer.9', 'layer.10', 'layer.11']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)], \
                'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and any(nd in n for nd in group1)], \
                'weight_decay': 0.01, 'lr': args.learning_rate/2.6**5},
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and any(nd in n for nd in group2)], \
                'weight_decay': 0.01, 'lr': args.learning_rate/2.6**4},
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and any(nd in n for nd in group3)], \
                'weight_decay': 0.01, 'lr': args.learning_rate/2.6**3},
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and any(nd in n for nd in group4)], \
                'weight_decay': 0.01, 'lr': args.learning_rate/2.6**2},
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and any(nd in n for nd in group5)], \
                'weight_decay': 0.01, 'lr': args.learning_rate/2.6},
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and any(nd in n for nd in group6)], \
                'weight_decay': 0.01, 'lr': args.learning_rate},

                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)], \
                'weight_decay': 0.0},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and any(nd in n for nd in group1)], \
                'weight_decay': 0.0, 'lr': args.learning_rate/2.6**5},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and any(nd in n for nd in group2)], \
                'weight_decay': 0.0, 'lr': args.learning_rate/2.6**4},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and any(nd in n for nd in group3)], \
                'weight_decay': 0.0, 'lr': args.learning_rate/2.6**3},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and any(nd in n for nd in group4)], \
                'weight_decay': 0.0, 'lr': args.learning_rate/2.6**2},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and any(nd in n for nd in group5)], \
                'weight_decay': 0.0, 'lr': args.learning_rate/2.6},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and any(nd in n for nd in group6)], \
                'weight_decay': 0.0, 'lr': args.learning_rate},
            ]
        else:
            optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]            


        if args.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if args.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
            warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
                                                 t_total=num_train_optimization_steps)

        else:
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=args.learning_rate,
                                 warmup=args.warmup_proportion,
                                 t_total=num_train_optimization_steps)

    ##################################################################################################
    ############################################# TRAIN ##############################################
    ##################################################################################################
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, output_mode)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)

        all_pids = np.array([f.pid for f in eval_features])

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size, drop_last=True)

        model.train()
        epoch_metric = {}
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                # define a new function to compute loss values for both output_modes
                logits = model(input_ids, segment_ids, input_mask, labels=None)

                if output_mode == "classification":
                    loss_fct = CrossEntropyLoss()
                    loss_fct = DataParallelCriterion(loss_fct)
                    logits = [logits[i].view(-1, num_labels) for i in range(len(logits))]
                    loss = loss_fct(logits, label_ids.view(-1))
                elif output_mode == "regression":
                    loss_fct = MSELoss()
                    loss_fct = DataParallelCriterion(loss_fct)
                    logits = [logits[i].view(-1) for i in range(len(logits))]
                    loss = loss_fct(logits, label_ids.view(-1))
                if n_gpu > 1:
                    loss = loss.mean() #average on multi-gpu
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        #modify lr with special warm up BERT uses
                        #if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear.get_lr(global_step, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            with torch.no_grad():
                model.eval()
                eval_loss = 0
                nb_eval_steps = 0
                preds = []
                i = 0
            
                for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    label_ids = label_ids.to(device)

                    with torch.no_grad():
                        logits = model(input_ids, segment_ids, input_mask, labels=None)

                    if output_mode == 'classification':
                        # loss_fct = CrossEntropyLoss()
                        # tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
                        loss_fct = CrossEntropyLoss()
                        loss_fct = DataParallelCriterion(loss_fct)
                        logits = [logits[i].view(-1, num_labels) for i in range(len(logits))]
                        tmp_eval_loss = loss_fct(logits, label_ids.view(-1))
                    elif output_mode == 'regression':
                        # loss_fct = MSELoss()
                        # tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

                        loss_fct = MSELoss()
                        loss_fct = DataParallelCriterion(loss_fct)
                        logits = [logits[i].view(-1) for i in range(len(logits))]
                        tmp_eval_loss = loss_fct(logits, label_ids.view(-1))

                    eval_loss += tmp_eval_loss.mean().item()
                    nb_eval_steps += 1
                    logits = parallel.gather(logits, target_device='cuda:0')
                    if len(preds) == 0:
                        preds.append(logits.detach().cpu().numpy())
                    else:
                        preds[0] = np.append(
                            preds[0], logits.detach().cpu().numpy(), axis=0)
                eval_loss = eval_loss / nb_eval_steps
                preds = preds[0]
                if output_mode == 'classification':
                    preds = np.argmax(preds, axis=1)
                elif output_mode == 'regression':
                    preds = np.squeeze(preds)
                

                all_label_ids = all_label_ids[:preds.shape[0]]
                all_pids = all_pids[:preds.shape[0]]
                errors = generate_errors(preds, all_label_ids.numpy(), all_pids)

                result = compute_metrics(task_name, preds, all_label_ids.numpy())
                


                loss = tr_loss / global_step if args.do_train else None

                result['eval_loss'] = eval_loss
                result['global_step'] = global_step
                result['loss'] = loss
                logger.info('***** Eval Results *****')
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))

                epoch_metric[_] = result['pearson'] if output_mode=='regression' else result['acc']



        output_eval_file = os.path.join(args.output_dir, 'eval_results.txt')
        with open(output_eval_file, 'w') as writer:
            logger.info('***** Eval Results *****')
            # for key in sorted(result.keys()):
            #     logger.info("  %s = %s", key, str(result[key]))
            #     writer.write("%s = %s\n" % (key, str(result[key])))
            # writer.write("{}     {}\n".format("epoch","pearson"))
            for key in sorted(epoch_metric.keys()):
                writer.write("{}\t{}\t{}\t{}\n".format(key, str(epoch_metric[key]), args.learning_rate, args.train_batch_size))

        errors.to_csv('errors.txt',sep='\t',index=False)


    ##################################################################################################
    ########################################## SAVE & RELOAD #########################################
    ##################################################################################################
    if args.do_train:
        #Save a trained model, config, and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model #only save the model itself
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(args.output_dir)
        model = BertForSequenceClassification.from_pretrained(args.output_dir, num_labels=num_labels)
        tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    else:
        model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=num_labels)
    model.to(device)

    ##################################################################################################
    ########################################### EVALUATION ###########################################
    ##################################################################################################
    # if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    #     eval_examples = processor.get_dev_examples(args.data_dir)
    #     eval_features = convert_examples_to_features(
    #         eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
    #     logger.info("***** Running evaluation *****")
    #     logger.info("  Num examples = %d", len(eval_examples))
    #     logger.info("  Batch size = %d", args.eval_batch_size)
    #     all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    #     all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    #     all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

    #     if output_mode == "classification":
    #         all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    #     elif output_mode == "regression":
    #         all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)

    #     eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    #     eval_sampler = SequentialSampler(eval_data)
    #     eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)


        # this is the part to put in training loop
        # model.eval()
        # eval_loss = 0
        # nb_eval_steps = 0
        # preds = []
        # for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
        #     input_ids = input_ids.to(device)
        #     input_mask = input_mask.to(device)
        #     segment_ids = segment_ids.to(device)
        #     label_ids = label_ids.to(device)
        #     with torch.no_grad():
        #         logits = model(input_ids, segment_ids, input_mask, labels=None)
        #     if output_mode == 'classification':
        #         loss_fct = CrossEntropyLoss()
        #         tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        #     elif output_mode == 'regression':
        #         loss_fct = MSELoss()
        #         tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

        #     eval_loss += tmp_eval_loss.mean().item()
        #     nb_eval_steps += 1
        #     if len(preds) == 0:
        #         preds.append(logits.detach().cpu().numpy())
        #     else:
        #         preds[0] = np.append(
        #             preds[0], logits.detach().cpu().numpy(), axis=0)
        # eval_loss = eval_loss / nb_eval_steps
        # preds = preds[0]
        # if output_mode == 'classification':
        #     preds = np.argmax(preds, axis=1)
        # elif output_mode == 'regression':
        #     preds = np.squeeze(preds)
        # result = compute_metrics(task_name, preds, all_label_ids.numpy())
        # loss = tr_loss / global_step if args.do_train else None

        # result['eval_loss'] = eval_loss
        # result['global_step'] = global_step
        # result['loss'] = loss



        # output_eval_file = os.path.join(args.output_dir, 'eval_results.txt')
        # with open(output_eval_file, 'w') as writer:
        #     logger.info('***** Eval Results *****')
        #     for key in sorted(result.keys()):
        #         logger.info("  %s = %s", key, str(result[key]))
        #         writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
    main()
