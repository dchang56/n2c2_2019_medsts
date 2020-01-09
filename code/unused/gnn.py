import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGPooling, global_mean_pool, global_add_pool
from torch_geometric.data import DataLoader
from torch.nn import Sequential, Linear, ELU, ReLU
import pickle
import argparse
import os
import random
import numpy as np
from scipy.stats import pearsonr

class GCNNet(nn.Module):
	def __init__(self, num_features, n_hidden, min_score):
		super(GCNNet, self).__init__()
		self.conv1 = GCNConv(num_features, n_hidden)
		self.conv2 = GCNConv(n_hidden, n_hidden)
		self.conv3 = GCNConv(n_hidden, n_hidden // 4) 
		self.pool = SAGPooling(n_hidden, min_score=min_score, GNN=GCNConv)      
		self.activation = ELU()
		self.final_pooling = global_add_pool

	def forward(self, data):
		x, edge_index, batch = data.x, data.edge_index, data.batch
		x = self.activation(self.conv1(x, edge_index))
		x = F.dropout(x, p=0.5, training=self.training)
		x = self.activation(self.conv2(x, edge_index))
		x = F.dropout(x, p=0.5, training=self.training)
		
		x, edge_index, _, batch, perm, score = self.pool(x, edge_index, None, batch)

		x = self.activation(self.conv3(x, edge_index))
		x = self.final_pooling(x, batch)
		return x

class PairClassifier(nn.Module):
	def __init__(self, dim_model, dim_hidden, num_classes):
		super(PairClassifier, self).__init__()
		self.classifier = Sequential(Linear(dim_model, dim_hidden),
							 ReLU(),
							 nn.Dropout(0.5),
							 Linear(dim_hidden, dim_hidden),
							 ReLU(),
							 nn.Dropout(0.5),
							 Linear(dim_hidden, num_classes))
	def forward(self, x):
		return self.classifier(x)
	
class Net(nn.Module):
	def __init__(self, n_hidden, min_score):
		super(Net, self).__init__()
		self.gcnnet = GCNNet(500, n_hidden, min_score)
		self.medsts_c = PairClassifier(n_hidden, 64, 5)
		self.medsts_type = PairClassifier(n_hidden, 64, 4)
		self.medsts = PairClassifier(n_hidden, 64, 1)
	def forward(self, data_a, data_b, mode='medsts'):
		x1 = self.gcnnet(data_a)
		x2 = self.gcnnet(data_b)
		x = torch.cat([x1, x2, x1-x2, x1*x2], dim=-1)
		if mode=='medsts':
			out = self.medsts(x).squeeze()
		elif mode=='medsts_c':
			out = self.medsts_c(x)
		elif mode=='medsts_type':
			out = self.medsts_type(x)
		else:
			return x
		return out


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir',
		required=True,
		type=str)
	parser.add_argument('--output_dir',
		default='',
		type=str)
	parser.add_argument('--num_epochs',
		default=400,
		type=int)
	parser.add_argument('--n_hidden',
		default=128,
		type=int)
	parser.add_argument('--min_score',
		default=0.015,
		type=float)
	parser.add_argument('--lr',
		default=0.001,
		type=float)
	parser.add_argument('--weight_decay',
		default=5e-4)
	parser.add_argument('--seed',
		default=42,
		type=int)
	parser.add_argument('--batch_size',
		default=32,
		type=int)

	args = parser.parse_args()

	n_gpu = torch.cuda.device_count()
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if n_gpu > 0:
		torch.cuda.manual_seed_all(args.seed)

	data_dir = args.data_dir
	with open(os.path.join(data_dir, 'dev_dataset_a.p'), 'rb') as f:
		dev_dataset_a = pickle.load(f)
	with open(os.path.join(data_dir, 'dev_dataset_b.p'), 'rb') as f:
		dev_dataset_b = pickle.load(f)
	with open(os.path.join(data_dir, 'train_dataset_a.p'), 'rb') as f:
		train_dataset_a = pickle.load(f)
	with open(os.path.join(data_dir, 'train_dataset_b.p'), 'rb') as f:
		train_dataset_b = pickle.load(f)
	train_a_dataloader = DataLoader(train_dataset_a+train_dataset_b, batch_size=args.batch_size)
	train_b_dataloader = DataLoader(train_dataset_b+train_dataset_a, batch_size=args.batch_size)
	dev_a_dataloader = DataLoader(dev_dataset_a+dev_dataset_b, batch_size=len(dev_dataset_a)+len(dev_dataset_b))
	dev_b_dataloader = DataLoader(dev_dataset_b+dev_dataset_a, batch_size=len(dev_dataset_b)+len(dev_dataset_a))

	n_hidden = args.n_hidden
	device = 'cuda'

	model = Net(n_hidden=n_hidden, min_score=args.min_score).to(device)
	optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr, weight_decay=args.lr)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', min_lr=0.00002, factor=0.9, patience=25)

	def train(model, loader_a, loader_b):
		model.train()
		total_loss = 0
		for data_a, data_b in zip(loader_a, loader_b):
			data_a, data_b = data_a.to(device), data_b.to(device)
			optimizer.zero_grad()

			out = model(data_a, data_b, mode='medsts')
			out_c = model(data_a, data_b, mode='medsts_c')
			out_type = model(data_a, data_b, mode='medsts_type')

			loss = F.mse_loss(out, data_a.label)
			loss_c = F.cross_entropy(out_c, data_a.label_c)
			loss_type = F.cross_entropy(out_type, data_a.label_type)

			loss_combined = loss + loss_c + loss_type
			loss_combined.backward()

			total_loss = data_a.num_graphs * loss_combined.item()
			optimizer.step()
		return total_loss / len(loader_a.dataset)
	def test(model, loader_a, loader_b):
		model.eval()
		running_corr = 0
		for data_a, data_b in zip(loader_a, loader_b):
			data_a, data_b = data_a.to(device), data_b.to(device)
			pred = model(data_a, data_b, mode='medsts')
			corr = pearsonr(pred.detach().cpu().numpy(), data_a.label.detach().cpu().numpy())[0]
			running_corr += corr
		return running_corr / (len(loader_a))
	

	output_dir = args.output_dir
	output_file = os.path.join(output_dir, 'results.txt')
	with open(output_file, 'w') as out:
		for epoch in range(args.num_epochs+1):
			loss = train(model, train_a_dataloader, train_b_dataloader)
			train_acc = test(model, train_a_dataloader, train_b_dataloader)
			test_acc = test(model, dev_a_dataloader, dev_b_dataloader)
			scheduler.step(test_acc)
			if epoch % 10 == 0:
				out.write('epoch {} \t loss: {} \t train acc: {} \t val acc: {}\n'.format(
					epoch, loss, train_acc, test_acc))
		
			if epoch % 20 == 0:
				print('epoch {} \t loss {} \t train acc: {} \t val acc: {}'.format(epoch, loss, train_acc, test_acc))

if __name__ == '__main__':
	main()













