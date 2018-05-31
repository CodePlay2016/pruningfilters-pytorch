import torch
from torch.autograd import Variable
from torchvision import models
import cv2
import sys, os
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dataset
from prune import *
import argparse
from operator import itemgetter
from heapq import nsmallest
import time
import matplotlib.pyplot as plt


class ModifiedVGG16Model(torch.nn.Module):
	def __init__(self):
		super(ModifiedVGG16Model, self).__init__()

		model = models.vgg16(pretrained=True)
		self.features = model.features

		for param in self.features.parameters():
			param.requires_grad = False

		self.classifier = nn.Sequential(
		    nn.Dropout(),
		    nn.Linear(25088, 4096),
		    nn.ReLU(inplace=True),
		    nn.Dropout(),
		    nn.Linear(4096, 4096),
		    nn.ReLU(inplace=True),
		    nn.Linear(4096, 2))

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x

class FilterPrunner:
	def __init__(self, model):
		self.model = model
		self.reset()
	
	def reset(self):
		# self.activations = []
		# self.gradients = []
		# self.grad_index = 0
		# self.activation_to_layer = {}
		self.filter_ranks = {}

	def forward(self, x):
		self.activations = []
		self.gradients = []
		self.grad_index = 0
		self.activation_to_layer = {}

		activation_index = 0
		for layer, (name, module) in enumerate(self.model.features._modules.items()):
		    x = module(x)
		    if isinstance(module, torch.nn.modules.conv.Conv2d):
                        x.register_hook(self.compute_rank)
                        self.activations.append(x)
                        self.activation_to_layer[activation_index] = layer
                        activation_index += 1

		return self.model.classifier(x.view(x.size(0), -1))

	def compute_rank(self, grad):
		activation_index = len(self.activations) - self.grad_index - 1
		activation = self.activations[activation_index]
		values = \
			torch.sum((activation * grad), dim = 0).\
				sum(dim=2).sum(dim=3)[0, :, 0, 0].data
		
		# Normalize the rank by the filter dimensions
		values = \
			values / (activation.size(0) * activation.size(2) * activation.size(3))

		if activation_index not in self.filter_ranks:
			self.filter_ranks[activation_index] = \
				torch.FloatTensor(activation.size(1)).zero_().cuda()

		self.filter_ranks[activation_index] += values
		self.grad_index += 1

	def lowest_ranking_filters(self, num):
		data = []
		for i in sorted(self.filter_ranks.keys()):
			for j in range(self.filter_ranks[i].size(0)):
				data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))

		return nsmallest(num, data, itemgetter(2))

	def normalize_ranks_per_layer(self):
		for i in self.filter_ranks:
			v = torch.abs(self.filter_ranks[i])
			v = v / np.sqrt(torch.sum(v * v))
			self.filter_ranks[i] = v.cpu()

	def get_prunning_plan(self, num_filters_to_prune):
		filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)

		# After each of the k filters are prunned,
		# the filter index of the next filters change since the model is smaller.
		filters_to_prune_per_layer = {}
		for (l, f, _) in filters_to_prune:
			if l not in filters_to_prune_per_layer:
				filters_to_prune_per_layer[l] = []
			filters_to_prune_per_layer[l].append(f)

		for l in filters_to_prune_per_layer:
			filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
			for i in range(len(filters_to_prune_per_layer[l])):
				filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i

		filters_to_prune = []
		for l in filters_to_prune_per_layer:
			for i in filters_to_prune_per_layer[l]:
				filters_to_prune.append((l, i))

		return filters_to_prune				

class PrunningFineTuner_VGG16:
	def __init__(self, train_path, test_path, model, log_dir=None):
		self.train_data_loader, self.valid_data_loader, self.test_data_loader\
			 = dataset.train_valid_test_loader(train_path)
		self.test_path = test_path
		self.model = model
		self.criterion = torch.nn.CrossEntropyLoss()
		self.prunner = FilterPrunner(self.model) 
		self.model.train()
		self.log_dir = log_dir
		self.p = Printer(log_dir)

	def test(self):
		self.model.eval()
		correct = 0
		total = 0

		for i, (batch, label) in enumerate(self.test_data_loader):
			batch = batch.cuda()
			output = model(Variable(batch))
			pred = output.data.max(1)[1]
	 		correct += pred.cpu().eq(label).sum()
	 		total += label.size(0)
	 	
	 	self.p.log("Test Accuracy :%.4f"% (float(correct) / total))
	 	
	 	self.model.train()

	def eval_train(self):
		self.model.eval()
		correct = 0
		total = 0
		for i, (batch, label) in enumerate(self.train_data_loader):
			if i and i%100 == 0: print i,
			batch = batch.cuda()
			output = model(Variable(batch))
			pred = output.data.max(1)[1]
	 		correct += pred.cpu().eq(label).sum()
	 		total += label.size(0)
	 	train_acc = float(correct) / total
	 	self.p.log("\ntrain accuracy :%.4f"% train_acc)
	 	
	 	self.model.train()

	def valid_train(self):
		self.model.eval()
		correct = 0
		total = 0
		for i, (batch, label) in enumerate(self.valid_data_loader):
			if i and i%10 == 0: print i,
			batch = batch.cuda()
			output = model(Variable(batch))
			pred = output.data.max(1)[1]
	 		correct += pred.cpu().eq(label).sum()
	 		total += label.size(0)
	 	valid_acc = float(correct) / total
	 	self.p.log("\nvalid Accuracy :%.4f"%valid_acc)
	 	
	 	self.model.train()
		return valid_acc

	def train(self, optimizer = None, epoches = 10):
		if optimizer is None:
			optimizer = \
				optim.Adam(model.classifier.parameters(), 
					lr=0.0001)

		best_acc = 0

		for i in range(epoches):
			self.p.log("Epoch: %d"%i)
			start = time.time()
			self.train_epoch(optimizer)
			train_time = time.time() - start
			self.eval_train()
			train_eval_time = time.time() - start - train_time
			valid_acc = self.valid_train()
			if best_acc < valid_acc:
				best_acc = valid_acc
				torch.save(self.model, os.path.join(self.log_dir,"model"))
				self.p.log("model resaved...")
			self.p.log("train step time elaps: %.2f, train_acc eval time elaps: %.2f, total time elaps: %.2f"%(
				train_time, train_eval_time, time.time()-start))
		self.test()
		self.p.log("Finished fine tuning. best valid acc is %.4f"%best_acc)
		
	
	def train_batch(self, optimizer, batch, label, rank_filters):
		self.model.zero_grad()
		input = Variable(batch)

		if rank_filters:
			output = self.prunner.forward(input)
			self.criterion(output, Variable(label)).backward()
		else:
			self.criterion(self.model(input), Variable(label)).backward()
			optimizer.step()

	def train_epoch(self, optimizer = None, rank_filters = False):
		for batch, label in self.train_data_loader:
			self.train_batch(optimizer, batch.cuda(), label.cuda(), rank_filters)

	def get_candidates_to_prune(self, num_filters_to_prune):
		self.prunner.reset()

		self.train_epoch(rank_filters = True)
		
		self.prunner.normalize_ranks_per_layer()

		return self.prunner.get_prunning_plan(num_filters_to_prune)
		
	def total_num_filters(self):
		filters = 0
		for name, module in self.model.features._modules.items():
			if isinstance(module, torch.nn.modules.conv.Conv2d):
				filters = filters + module.out_channels
		return filters

	def prune(self):
		#Get the accuracy before prunning
		self.test()

		self.model.train()

		#Make sure all the layers are trainable
		for param in self.model.features.parameters():
			param.requires_grad = True

		number_of_filters = self.total_num_filters()
		num_filters_to_prune_per_iteration = 512
		iterations = int(float(number_of_filters) / num_filters_to_prune_per_iteration)

		iterations = int(iterations * 2.0 / 3)

		self.p.log(r"Number of prunning iterations to reduce 67% filters "+str(iterations))

		for _ in range(iterations):
			self.p.log("Ranking filters.. ")
			prune_targets = self.get_candidates_to_prune(num_filters_to_prune_per_iteration)
			layers_prunned = {}
			for layer_index, filter_index in prune_targets:
				if layer_index not in layers_prunned:
					layers_prunned[layer_index] = 0
				layers_prunned[layer_index] = layers_prunned[layer_index] + 1 

			self.p.log("Layers that will be prunned :"+str(layers_prunned))
			self.p.log("Prunning filters.. ")
			model = self.model.cpu()
			for layer_index, filter_index in prune_targets:
				model = prune_vgg16_conv_layer(model, layer_index, filter_index)

			self.model = model.cuda()

			message = str(100*float(self.total_num_filters()) / number_of_filters) + "%"
			self.p.log("Filters prunned"+str(message))
			self.test()
			self.p.log("Fine tuning to recover from prunning iteration.")
			optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
			self.train(optimizer, epoches = 10)


		self.p.log("Finished. Going to fine tune the model a bit more")
		self.train(optimizer, epoches = 15)
		torch.save(model, "model_prunned")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--prune", dest="prune", action="store_true")
    parser.add_argument("--train_path", type = str, default = "train")
    parser.add_argument("--test_path", type = str, default = "test")
    parser.add_argument("--model_path", type = str, default = "model")
    parser.set_defaults(train=False)
    parser.set_defaults(prune=False)
    args = parser.parse_args()
    return args

class Printer():
	def __init__(self, log_dir):
		self.log_dir = log_dir
	def log(self, mstr):
		print(mstr)
		with open(os.path.join(self.log_dir,"log.txt"),"a") as f:
			f.write(mstr+"\n")

if __name__ == '__main__':
	args = get_args()

	time_info = time.strftime('%Y-%m-%d_%H%M%S',time.localtime(time.time()))
	if args.train:
		model = ModifiedVGG16Model().cuda()
		log_dir = os.path.abspath("./log/train-"+time_info+"/")
	elif args.prune:
		model = torch.load(args.model_path).cuda()
		log_dir = os.path.abspath("./log/prune-"+time_info+"/")
	os.system("mkdir "+log_dir)
	os.system("touch "+os.path.join(log_dir,"log.txt"))
	p = Printer(log_dir)
	p.log(str(model))

	p.log("time is :"+time_info)
	fine_tuner = PrunningFineTuner_VGG16(args.train_path, args.test_path, model, log_dir)
	if args.train:
		p.log("begin training...")
		fine_tuner.train(epoches = 20)
		os.system("cp finetune.py "+log_dir)
		# torch.save(model, log_dir+"model")

	elif args.prune:
		fine_tuner.prune()