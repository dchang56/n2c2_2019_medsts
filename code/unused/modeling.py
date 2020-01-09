import copy
import json
import math
import six
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.autograd import Variable
from torch.nn.parameter import Parameter

def gelu(x):
	"""Implementation of the gelu activation function.
		For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
		0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
	"""
	return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BertConfig(object):
	"""Configuration class to store the configuration of a `BertModel`.
	"""
	def __init__(self,
				vocab_size,
				hidden_size=768,
				num_hidden_layers=12,
				num_attention_heads=12,
				intermediate_size=3072,
				hidden_act="gelu",
				hidden_dropout_prob=0.1,
				attention_probs_dropout_prob=0.1,
				max_position_embeddings=512,
				type_vocab_size=2,
				initializer_range=0.02,
				pals=False,
				mult=False,
				top=False,
				houlsby=False,
				bert_lay_top=False,
				num_tasks=1,
				hidden_size_aug=204):
		"""Constructs BertConfig.
		Args:
			vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
			hidden_size: Size of the encoder layers and the pooler layer.
			num_hidden_layers: Number of hidden layers in the Transformer encoder.
			num_attention_heads: Number of attention heads for each attention layer in
				the Transformer encoder.
			intermediate_size: The size of the "intermediate" (i.e., feed-forward)
				layer in the Transformer encoder.
			hidden_act: The non-linear activation function (function or string) in the
				encoder and pooler.
			hidden_dropout_prob: The dropout probabilitiy for all fully connected
				layers in the embeddings, encoder, and pooler.
			attention_probs_dropout_prob: The dropout ratio for the attention
				probabilities.
			max_position_embeddings: The maximum sequence length that this model might
				ever be used with. Typically set this to something large just in case
				(e.g., 512 or 1024 or 2048).
			type_vocab_size: The vocabulary size of the `token_type_ids` passed into
				`BertModel`.
			initializer_range: The sttdev of the truncated_normal_initializer for
				initializing all weight matrices.
		"""
		self.vocab_size = vocab_size
		self.hidden_size = hidden_size
		self.num_hidden_layers = num_hidden_layers
		self.num_attention_heads = num_attention_heads
		self.hidden_act = hidden_act
		self.intermediate_size = intermediate_size
		self.hidden_dropout_prob = hidden_dropout_prob
		self.attention_probs_dropout_prob = attention_probs_dropout_prob
		self.max_position_embeddings = max_position_embeddings
		self.type_vocab_size = type_vocab_size
		self.initializer_range = initializer_range
		self.hidden_size_aug = hidden_size_aug
		self.pals = pals
		self.houlsby = houlsby
		self.mult = mult
		self.top = top
		self.bert_lay_top = bert_lay_top
		self.num_tasks=num_tasks

	@classmethod
	def from_dict(cls, json_object):
		"""Constructs a `BertConfig` from a Python dictionary of parameters."""
		config = BertConfig(vocab_size=None)
		for (key, value) in six.iteritems(json_object):
			config.__dict__[key] = value
		return config

	@classmethod
	def from_json_file(cls, json_file):
		"""Constructs a `BertConfig` from a json file of parameters."""
		with open(json_file, "r") as reader:
			text = reader.read()
		return cls.from_dict(json.loads(text))

	def to_dict(self):
		"""Serializes this instance to a Python dictionary."""
		output = copy.deepcopy(self.__dict__)
		return output

	def to_json_string(self):
		"""Serializes this instance to a JSON string."""
		return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

	def to_json_file(self, json_file_path):
		with open(json_file_path, 'w', encoding='utf-8') as writer:
			writer.write(self.to_json_string())


class BERTLayerNorm(nn.Module):
	def __init__(self, config, multi_params=None, variance_epsilon=1e-12):
		"""Construct a layernorm module in the TF style (epsilon inside the square root).
		"""
		super(BERTLayerNorm, self).__init__()
		if multi_params is not None:
			self.gamma = nn.Parameter(torch.ones(config.hidden_size_aug))
			self.beta = nn.Parameter(torch.zeros(config.hidden_size_aug))
		else:
			self.gamma = nn.Parameter(torch.ones(config.hidden_size))
			self.beta = nn.Parameter(torch.zeros(config.hidden_size))
		self.variance_epsilon = variance_epsilon

	def forward(self, x):
		u = x.mean(-1, keepdim=True)
		s = (x - u).pow(2).mean(-1, keepdim=True)
		x = (x - u) / torch.sqrt(s + self.variance_epsilon)
		return self.gamma * x + self.beta

class BERTEmbeddings(nn.Module):
	def __init__(self, config):
		super(BERTEmbeddings, self).__init__()
		self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
		self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
		self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
		self.LayerNorm = BERTLayerNorm(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
	def forward(self, input_ids, token_type_ids=None):
		seq_length = input_ids.size(1)
		position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
		position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
		if token_type_ids is None:
			token_type_ids = torch.zeros_like(input_ids)
		words_embeddings = self.word_embeddings(input_ids)
		position_embeddings = self.position_embeddings(position_ids)
		token_type_embeddings = self.token_type_embeddings(token_type_ids)

		embeddings = words_embeddings + position_embeddings + token_type_embeddings
		embeddings = self.LayerNorm(embeddings)
		embeddings = self.dropout(embeddings)

		return embeddings

class BERTSelfAttention(nn.Module):
	def __init__(self, config, multi_params=None):
		super(BERTSelfAttention, self).__init__()
		if config.hidden_size % config.num_attention_heads != 0:
			raise ValueError('hidden size is not a multiple of num heads')
		if multi_params is not None:
			self.num_attention_heads = multi_params
			self.attention_head_size = int(config.hidden_size_aug / self.num_attention_heads)
			self.all_head_size = self.num_attention_heads * self.attention_head_size
			hidden_size = config.hidden_size_aug
		else:
			self.num_attention_heads = config.num_attention_heads
			self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
			self.all_head_size = self.num_attention_heads * self.attention_head_size
			hidden_size = config.hidden_size

		self.query = nn.Linear(hidden_size, self.all_head_size)
		self.key = nn.Linear(hidden_size, self.all_head_size)
		self.value = nn.Linear(hidden_size, self.all_head_size)

		self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

	def transpose_for_scores(self, x):
		new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
		x = x.view(*new_x_shape)
		return x.permute(0, 2, 1, 3)

	def forward(self, hidden_states, attention_mask):
		mixed_query_layer = self.query(hidden_states)
		mixed_key_layer = self.key(hidden_states)
		mixed_value_layer = self.value(hidden_states)

		query_layer = self.transpose_for_scores(mixed_query_layer)
		key_layer = self.transpose_for_scores(mixed_key_layer)
		value_layer = self.transpose_for_scores(mixed_value_layer)

		attention_scores = torch.matmul(query_layer, key_layer.transpose(-1,-2))
		attention_scores = attention_scores / math.sqrt(self.attention_head_size)
		attention_scores = attention_scores + attention_mask

		attention_probs = nn.Softmax(dim=-1)(attention_scores)
		attention_probs = self.dropout(attention_probs)

		context_layer = torch.matmul(attention_probs, value_layer)
		context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
		new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
		context_layer = context_layer.view(*new_context_layer_shape)
		return context_layer

class BERTSelfOutput(nn.Module):
	def __init__(self, config, multi_params=None, houlsby=False):
		super(BERTSelfOutput, self).__init__()
		if houlsby:
			multi = BERTLowRank(config)
			self.multi_layers = nn.ModuleList([copy.deepcopy(multi) for _ in range(config.num_tasks)])
		if multi_params is not None:
			self.dense = nn.Linear(config.hidden_size_aug, config.hidden_size_aug)
		else:
			self.dense = nn.Linear(config.hidden_size, config.hidden_size)
		self.LayerNorm = BERTLayerNorm(config, multi_params)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.houlsby = houlsby
	def forward(self, hidden_states, input_tensor, attention_mask=None, i=0):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.dropout(hidden_states)
		if self.houlsby:
			hidden_states = hidden_states + self.multi_layers[i](hidden_states, attention_mask)
		hidden_states = self.LayerNorm(hidden_states + input_tensor)
		return hidden_states

class BERTAttention(nn.Module):
	def __init__(self, config, multi_params=None, houlsby=False):
		super(BERTAttention, self).__init__()
		self.self = BERTSelfAttention(config, multi_params)
		self.output = BERTSelfOutput(config, multi_params, houlsby)
	def forward(self, input_tensor, attention_mask, i=0):
		self_output = self.self(input_tensor, attention_mask)
		attention_output = self.output(self_output, input_tensor, attention_mask, i=i)
		return attention_output

class BERTPals(nn.Module):
	def __init__(self, config):
		super(BERTPals, self).__init__()
		self.aug_dense = nn.Linear(config.hidden_size, config.hidden_size_aug)
		self.aug_dense2 = nn.Linear(config.hidden_size_aug, config.hidden_size)
		self.attn = BERTSelfAttention(config, 6)
		self.config = config
		self.hidden_act_fn = gelu
	def forward(self, hidden_states, attention_mask=None):
		hidden_states_aug = self.aug_dense(hidden_states)
		hidden_states_aug = self.attn(hidden_states_aug, attention_mask)
		hidden_states = self.aug_dense2(hidden_states_aug)
		hidden_states = self.hidden_act_fn(hidden_states)
		return hidden_states

class BERTLowRank(nn.Module):
	def __init__(self, config, extra_dim=None):
		super(BERTLowRank, self).__init__()
		if config.extra_dim:
			self.aug_dense = nn.Linear(config.hidden_size, config.extra_dim)
			self.aug_dense2 = nn.Linear(config.extra_dim, config.hidden_size)
		else:
			self.aug_dense = nn.Linear(config.hidden_size, config.hidden_size_aug)
			self.aug_dense2 = nn.Linear(config.hidden_size_aug, config.hidden_size)
		self.config = config
		self.hidden_act_fn = gelu
	def forward(self, hidden_states, attention_mask=None):
		hidden_states_aug = self.aug_dense(hidden_states)
		hidden_states_aug = self.hidden_act_fn(hidden_states_aug)
		hidden_states = self.aug_dense2(hidden_states_aug)
		return hidden_states

class BERTIntermediate(nn.Module):
	def __init__(self, config):
		super(BERTIntermediate, self).__init__()
		self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
		self.config = config
		self.intermediate_act_fn = gelu
	def forward(self, hidden_states):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.intermediate_act_fn(hidden_states)
		return hidden_states

class BERTOutput(nn.Module):
	def __init__(self, config, houlsby=False):
		super(BERTOutput, self).__init__()
		self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
		self.LayerNorm = BERTLayerNorm(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		if houlsby:
			if config.pals:
				multi = BERTPals(config)
			else:
				multi = BERTLowRank(config)
			self.multi_layers = nn.ModuleList([copy.deepcopy(multi) for _ in range(config.num_tasks)])
		self.houlsby = houlsby
	def forward(self, hidden_states, input_tensor, attention_mask=None, i=0):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.dropout(hidden_states)
		if self.houlsby:
			hidden_states = hidden_states + self.multi_layers[i](input_tensor, attention_mask)
		hidden_states = self.LayerNorm(hidden_states + input_tensor)
		return hidden_states

class BERTLayer(nn.Module):
	def __init__(self, config, mult=False, houlsby=False):
		super(BERTLayer, self).__init__()
		self.attention = BERTAttention(config, houlsby=houlsby)
		self.intermediate = BERTIntermediate(config)
		self.output = BERTOutput(config, houlsby=houlsby)
		if mult:
			if config.pals:
				multi = BERTPals(config)
			else:
				multi = BERTLowRank(config)
			self.multi_layers = nn.ModuleList([copy.deepcopy(multi) for _ in range(config.num_tasks)])
		self.mult = mult
		self.houlsby = houlsby
	def forward(self, hidden_states, attention_mask, i=0):
		attention_output = self.attention(hidden_states, attention_mask, i)
		intermediate_output = self.intermediate(attention_output)
		if self.mult:
			extra = self.multi_layers[i](hidden_states, attention_mask)
			layer_output = self.output(intermediate_output, attention_output + extra)
		elif self.houlsby:
			layer_output = self.output(intermediate_output, attention_output, attention_mask, i)
		else:
			layer_output = self.output(intermediate_output, attention_output)
		return layer_output

class BERTEncoder(nn.Module):
	def __init__(self, config):
		super(BERTEncoder, self).__init__()
		self.config = config
		if config.houlsby:
			self.multis = [True if i < 20 else False for i in range(config.num_hidden_layers)]
			self.layer = nn.ModuleList([BERTLayer(config, houlsby=mult) for mult in self.multis])
		elif config.mult:
			self.multis = [True if i < 20 else False for i in range(config.num_hidden_layers)]
			self.layer = nn.ModuleList([BERTLayer(config, mult=mult) for mult in self.multis])
		else:
			layer = BERTLayer(config)
			self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

		if config.top:
			if config.bert_lay_top:
				multi = BERTLayer(config)
			else:
				#projection and attention for top
				mult_dense = nn.Linear(config.hidden_size, config.hidden_size_aug)
				self.mult_dense = nn.ModuleList([copy.deepcopy(mult_dense) for _ in range(config.num_tasks)])
				mult_dense2 = nn.Linear(config.hidden_size_aug, config.hidden_size)
				self.mult_dense2 = nn.ModuleList([copy.deepcopy(mult_dense2) for _ in range(config.num_tasks)])
				multi = nn.ModuleList([copy.deepcopy(BERTAttention(config, 12)) for _ in range(6)])
			self.multi_layers = nn.ModuleList([copy.deepcopy(multi) for _ in range(config.num_tasks)])
			self.gelu = gelu

		if config.mult and config.pals:
			dense = nn.Linear(config.hidden_size, config.hidden_size_aug)
			#shared encoder and decoder across layers
			self.mult_aug_dense = nn.ModuleList([copy.deepcopy(dense) for _ in range(config.num_tasks)])
			dense2 = nn.Linear(config.hidden_size_aug, config.hidden_size)
			self.mult_aug_dense2 = nn.ModuleList([copy.deepcopy(dense2) for _ in range(config.num_tasks)])
			for l, layer in enumerate(self.layer):
				if self.multis[l]:
					for i, lay in enumerate(layer.multi_layers):
						#for each BERTPals
						lay.aug_dense = self.mult_aug_dense[i]
						lay.aug_dense2 = self.mult_aug_dense2[i]

	def forward(self, hidden_states, attention_mask, i=0):
		all_encoder_layers = []
		for layer_module in self.layer:
			hidden_states = layer_module(hidden_states, attention_mask, i)
			all_encoder_layers.append(hidden_states)
		if self.config.top:
			if self.config.bert_lay_top:
				all_encoder_layers[-1] = self.multi_layers[i](hidden_states, attention_mask)
			else:
				hidden_states = self.mult_dense[i](hidden_states)
				for lay in self.multi_layers[i]:
					hidden_states = lay(hidden_states, attention_mask)
				all_encoder_layers[-1] = self.mult_dense2[i](hidden_states)
		return all_encoder_layers

class BERTPooler(nn.Module):
	def __init__(self, config):
		super(BERTPooler, self).__init__()
		dense = nn.Linear(config.hidden_size, config.hidden_size)
		self.activation = nn.Tanh()
		self.pool = False
		if self.pool:
			self.mult_dense_layers = nn.ModuleList([copy.deepcopy(dense) for _ in range(config.num_tasks)])
		else:
			self.dense = dense
		self.mult = config.mult
		self.top = config.top
	def forward(self, hidden_states, i=0):
		first_token_tensor = hidden_states[:, 0]
		if (self.mult or self.top) and self.pool:
			pooled_output = self.mult_dense_layers[i](first_token_tensor)
		else:
			pooled_output = self.dense(first_token_tensor)
		pooled_output = self.activation(pooled_output)
		return pooled_output

class BertModel(nn.Module):
	def __init__(self, config: BertConfig):
		super(BertModel, self).__init__()
		self.embeddings = BERTEmbeddings(config)
		self.encoder = BERTEncoder(config)
		self.pooler = BERTPooler(config)
	def forward(self, input_ids, token_type_ids=None, attention_mask=None, i=0):
		if attention_mask is None:
			attention_mask = torch.ones_like(input_ids)
		if token_type_ids is None:
			token_type_ids = torch.zeros_like(input_ids)

		#create a 3D attention mask from a 2D tensor mask
		#sizes [bs, 1, 1, from_seq_length]
		#broadcast to [bs, num_heads, to_seq_length, from_seq_length]
		extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
		#attention_mask is 1 for positions we want to attend and 0 for masked,
		#this operation creates a tensor which is 0 for positions we want to attend
		#and -10000 for masked positions
		#since we add it to the raw scres before softmax, this is effectively same as removing them entirely
		extended_attention_mask = extended_attention_mask.float()
		extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

		embedding_output = self.embeddings(input_ids, token_type_ids)
		all_encoder_layers = self.encoder(embedding_output, extended_attention_mask, i)
		sequence_output = all_encoder_layers[-1]
		pooled_output = self.pooler(sequence_output, i)
		return all_encoder_layers, pooled_output

class BertForMultiTask(nn.Module):
	def __init__(self, config, tasks_n_label):
		super(BertForMultiTask, self).__init__()
		self.bert = BertModel(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.classifier = nn.ModuleList([nn.Linear(config.hidden_size, num_labels) for i, num_labels in enumerate(tasks_n_label)])
		
		def init_weights(module):
			if isinstance(module, (nn.Linear, nn.Embedding)):
				module.weight.data.normal_(mean=0.0, std=config.initializer_range)
			elif isinstance(module, BERTLayerNorm):
				module.beta.data.normal_(mean=0.0, std=config.initializer_range)
				module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
			if isinstance(module, nn.Linear):
				if module.bias is not None:
					module.bias.data.zero_()
		self.apply(init_weights)

	def forward(self, input_ids, token_type_ids, attention_mask, task_id, output_mode, labels=None):
		_, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, task_id)
		pooled_output = self.dropout(pooled_output)
		logits = self.classifier[task_id](pooled_output)

		if labels is not None and output_mode == 'classification':
			loss_fct = CrossEntropyLoss()
			loss = loss_fct(logits, labels)
			return loss, logits
		elif labels is not None and output_mode == 'regression':
			loss_fct = MSELoss()
			loss = loss_fct(logits, labels.unsqueeze(1))
			return loss, logits
		else:
			return logits









































































































































































