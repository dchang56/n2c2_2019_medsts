import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Sequential, Linear, ELU, ReLU
from pytorch_transformers.modeling_bert import gelu
from torch_geometric.nn import GCNConv, SAGPooling, global_mean_pool, global_add_pool
from pytorch_transformers.modeling_bert import gelu


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class GCNNet(nn.Module):
	def __init__(self, num_features, n_hidden, min_score):
		super(GCNNet, self).__init__()
		self.conv1 = GCNConv(num_features, n_hidden)
		self.conv2 = GCNConv(n_hidden, n_hidden)
		self.conv3 = GCNConv(n_hidden, n_hidden // 4) 
		self.pool = SAGPooling(n_hidden, min_score=min_score, GNN=GCNConv)      
		self.activation = gelu
		self.final_pooling = global_add_pool
		self.layer_norm = LayerNorm(hidden_size=n_hidden)

	def forward(self, data):
		x, edge_index, batch = data.x, data.edge_index, data.batch
		x = self.activation(self.conv1(x, edge_index))
		x = F.dropout(x, p=0.5, training=self.training)

		x2 = self.activation(self.conv2(x, edge_index))
		x2 = F.dropout(x2, p=0.5, training=self.training)

		#layernorm + skip connection

		x = self.layer_norm(x2 + x)

		
		x, edge_index, _, batch, perm, score = self.pool(x, edge_index, None, batch)

		x = self.activation(self.conv3(x, edge_index))
		x = self.final_pooling(x, batch)
		return x

class Pooler(nn.Module):
	def __init__(self, n_hidden, n_out):
		super(Pooler, self).__init__()
		self.dense = Linear(n_hidden, n_out)
		self.activation = nn.Tanh()
	def forward(self, x):
		pooled_output = self.dense(x)
		pooled_output = self.activation(pooled_output)
		return pooled_output

class GraphEncoder(nn.Module):
	def __init__(self, n_hidden, min_score):
		super(GraphEncoder, self).__init__()
		self.gcnnet = GCNNet(500, n_hidden, min_score)
		self.pooler = Pooler(n_hidden, n_hidden)
	def forward(self, data_a, data_b):
		x1 = self.gcnnet(data_a)
		x2 = self.gcnnet(data_b)
		x = torch.cat([x1, x2, x1-x2, x1*x2], dim=-1)
		x = self.pooler(x)

		return x

class PairClassifier(nn.Module):
	def __init__(self, dim_model, num_classes):
		super(PairClassifier, self).__init__()
		self.linear = Linear(dim_model, num_classes)
	def forward(self, x):
		return self.linear(x)

class MedstsNet(nn.Module):
	def __init__(self, bert, graph_encoder, classifier, classifier_c, classifier_type):
		super(MedstsNet, self).__init__()
		self.text_encoder = bert
		self.graph_encoder = graph_encoder
		self.dropout = nn.Dropout(p=0.5)
		self.classifier = classifier
		self.classifier_c = classifier_c
		self.classifier_type = classifier_type
	def forward(self, input_ids, attention_mask, token_type_ids, data_a, data_b, mode='medsts'):
		outputs = self.text_encoder(input_ids, token_type_ids, attention_mask)
		pooled_output = outputs[1]
		graph_output = self.graph_encoder(data_a, data_b)

		out = torch.cat([pooled_output, graph_output], dim=-1)
		out = self.dropout(out)

		if mode=='medsts':
			logits = self.classifier(out).squeeze()
		elif mode=='medsts_c':
			logits = self.classifier_c(out)
		elif mode=='medsts_type':
			logits = self.classifier_type(out)

		return logits























