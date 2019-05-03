import os
import time
import numpy as np
import argparse
import torch
import torch.nn as nn
from vnetg_data_load import *

def arg_parser():
	parser = argparse.ArgumentParser(description='EBP')
	parser.add_argument('--data_path', type=str, default=None, help='data path')
	parser.add_argument('--gpu_id', type=int, default=0)
	parser.add_argument('--slices', type=int, default=5, help='neighboring slices')
	parser.add_argument('--organ_id', type=int, default=1)
	parser.add_argument('-t', '--timestamp', type=str, default=None)
	parser.add_argument('--folds', type=int, default=1)
	parser.add_argument('-f', '--current_fold', type=int, default=0)
	parser.add_argument('-b', '--batch', type=int, default=32, help='input batch size for training')
	parser.add_argument('-e', '--epoch', type=int, default=1, help='number of epochs to train')
	parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
	return parser.parse_args()

if __name__ == '__main__':
	args = arg_parser()

	# HyperParameter
	epoch = args.epoch
	batch_size = args.batch
	lr = args.lr
	os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu_id)
	np.set_printoptions(precision=3, suppress=True)

	# build pytorch dataset
	training_set = EBP(train=True, data_path=args.data_path, folds=args.folds, current_fold=args.current_fold, organ_id=args.organ_id, slices=args.slices)
	testing_set = EBP(train=False, data_path=args.data_path, folds=args.folds, current_fold=args.current_fold, organ_id=args.organ_id, slices=args.slices)
	trainloader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=12)
	testloader = torch.utils.data.DataLoader(testing_set, batch_size=batch_size, shuffle=False, num_workers=12)

class DownTransition(nn.Module):

	def __init__(self,inchan,outchan,layer,dilation_=1):
		super(DownTransition, self).__init__()
		self.dilation_ = dilation_
		self.outchan = outchan
		self.layer = layer
		self.down = nn.Conv2d(in_channels=inchan,out_channels=self.outchan,kernel_size=3,padding=1,stride=2, groups=2) # /2
		self.bn = nn.BatchNorm2d(num_features=self.outchan,affine=True)
		self.conv = self.make_layers()
		self.relu = nn.ELU(inplace=True)

	def make_layers(self):
		layers = []
		for i in range(self.layer):
			layers.append(nn.ELU(inplace=True))
			layers.append(nn.Conv2d(self.outchan,self.outchan,kernel_size=3,padding=self.dilation_,stride=1,dilation=self.dilation_,groups=2))
			layers.append(nn.BatchNorm2d(num_features=self.outchan,affine=True))

		return nn.Sequential(*layers)

	def forward(self,x):
		out1 = self.down(x)
		out2 = self.conv(self.bn(out1))
		out2 = self.relu(torch.add(out1,out2))
		return out2

class UpTransition(nn.Module):

	def __init__(self,inchan,outchan,layer,last=False):
		super(UpTransition, self).__init__()
		self.last = last
		self.outchan = outchan
		self.layer = layer
		self.up = nn.ConvTranspose2d(in_channels=inchan,out_channels=self.outchan,kernel_size=4,padding=1,stride=2) # *2
		self.bn = nn.BatchNorm2d(num_features=self.outchan,affine=True)
		self.conv = self.make_layers()
		self.relu = nn.ELU(inplace=True)
		if self.last is True:
			self.conv1 = nn.Conv2d(self.outchan,1,kernel_size=1) # 1*1 conv. one channel

	def make_layers(self):
		layers = []
		for i in range(self.layer):
			layers.append(nn.ELU(inplace=True))
			layers.append(nn.Conv2d(self.outchan,self.outchan,kernel_size=3,padding=1,stride=1,groups=2))
			layers.append(nn.BatchNorm2d(num_features=self.outchan,affine=True))

		return nn.Sequential(*layers)

	def forward(self,x):
		out1 = self.up(x)
		out = self.conv(self.bn(out1))
		out = self.relu(torch.add(out1,out))
		if self.last is True:
			out = self.conv1(out) # NCHW, C=1
			out = torch.clamp(out, min=-Y_SCALE, max=Y_SCALE)
		return out

class Vnet(nn.Module):
	def __init__(self, slices, inchans, outchans, down_layers, up_layers, dilations):
		super(Vnet,self).__init__()

		self.layer0 = nn.Sequential(
				nn.Conv2d(16, 16, kernel_size=7, stride=1, padding=3, groups=2, bias=False),
				nn.BatchNorm2d(16,affine=True),
				nn.ELU(inplace=True)
			)

		self.block_num = len(inchans)
		self.down = nn.ModuleList()
		self.up = nn.ModuleList()

		for i in range(self.block_num):
			self.down.append(DownTransition(inchan=inchans[i], outchan=outchans[i], layer=down_layers[i], dilation_=dilations[i]))
			if i==0 :
				self.up.append(UpTransition(inchan=outchans[i], outchan=inchans[i], layer=up_layers[i], last=True))
			else:
				self.up.append(UpTransition(inchan=outchans[i], outchan=inchans[i], layer=up_layers[i]))

		for m in self.modules():
			if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
				nn.init.kaiming_normal_(m.weight.data)
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def forward(self,x):
		x = self.layer0(x)
		out_down = []
		out_down.append(self.down[0](x))

		for i in range(1,self.block_num):
			out_down.append(self.down[i](out_down[i-1]))

		out_up = self.up[self.block_num-1](out_down[self.block_num-1])
		for i in reversed(range(self.block_num-1)):
			out_up = self.up[i](torch.add(out_up,out_down[i]))

		return out_up

class OHEM(nn.Module):
	def __init__(self, top_k=0.7):
		super(OHEM, self).__init__()
		self.criterion = nn.MSELoss(reduce=False)
		self.top_k = top_k
	def forward(self, input, target):
		batch = input.shape[0]
		loss = self.criterion(input.view(batch, -1), target.view(batch, -1))
		values, _ = torch.topk(torch.mean(loss, dim=1), int(self.top_k * batch))
		return torch.mean(values)

def train():
	for e in range(epoch):
		model.train()
		total_loss = np.zeros((4, ITER_TH))
		period_loss = np.zeros((4, ITER_TH))
		total_pivot_kind = np.zeros((4, ITER_TH), dtype=np.int32)
		period_pivot_kind = np.zeros((4, ITER_TH), dtype=np.int32)
		start = time.time()

		for index, (image, target, pivot_kind, iter_kind) in enumerate(trainloader):
			batch = image.shape[0]
			image, target = image.cuda().float(), target.cuda().float()
			optimizer.zero_grad()
			output = model(image)
			loss = valid_criterion(output.view(batch, -1), target.view(batch, -1))
			loss = torch.mean(loss, dim=1)
			for p in range(batch):
				total_pivot_kind[pivot_kind[p].item(), iter_kind[p].item()] += 1
				period_pivot_kind[pivot_kind[p].item(), iter_kind[p].item()] += 1
				total_loss[pivot_kind[p].item(), iter_kind[p].item()] += loss[p].item()
				period_loss[pivot_kind[p].item(), iter_kind[p].item()] += loss[p].item()
			OHEM_loss = train_criterion(output, target)
			OHEM_loss.backward()
			optimizer.step()
			if index % period == (period - 1):
				print ("CNN Train Epoch[%d/%d], Iter[%05d], Time elapsed %ds" %(e+1, epoch, index, time.time()-start))
				print ('avg loss:', period_loss / period_pivot_kind)
				period_loss = np.zeros((4, ITER_TH))
				period_pivot_kind = np.zeros((4, ITER_TH), dtype=np.int32)

		print ('#'*10, "CNN Train TOTAL Epoch[%d/%d], Time elapsed %ds" %(e+1, epoch, time.time()-start))
		print ('#'*10, 'avg loss:', total_loss / total_pivot_kind)
		for param_group in optimizer.param_groups:
			param_group['lr'] *= 0.5
		print('#'*10, 'lr decay')

		with torch.no_grad():
			model.eval()
			total_loss = np.zeros((4, ITER_TH))
			period_loss = np.zeros((4, ITER_TH))
			total_pivot_kind = np.zeros((4, ITER_TH), dtype=np.int32)
			period_pivot_kind = np.zeros((4, ITER_TH), dtype=np.int32)
			start = time.time()
			
			for index, (image, target, pivot_kind, iter_kind) in enumerate(testloader):
				batch = image.shape[0]
				image, target = image.cuda().float(), target.cuda().float()
				output = model(image)
				loss = valid_criterion(output.view(batch, -1), target.view(batch, -1))
				loss = torch.mean(loss, dim=1)
				for p in range(batch):
					total_pivot_kind[pivot_kind[p].item(), iter_kind[p].item()] += 1
					period_pivot_kind[pivot_kind[p].item(), iter_kind[p].item()] += 1
					total_loss[pivot_kind[p].item(), iter_kind[p].item()] += loss[p].item()
					period_loss[pivot_kind[p].item(), iter_kind[p].item()] += loss[p].item()
				if index % period == (period - 1):
					print ("CNN Valid Epoch[%d/%d], Iter[%05d], Time elapsed %ds" %(e+1, epoch, index, time.time()-start))
					print ('avg loss:', period_loss / period_pivot_kind)
					period_loss = np.zeros((4, ITER_TH))
					period_pivot_kind = np.zeros((4, ITER_TH), dtype=np.int32)

			print ('*'*10, "CNN Valid TOTAL Epoch[%d/%d], Time elapsed %ds" %(e+1, epoch, time.time()-start))
			print ('*'*10, 'avg loss:', total_loss / total_pivot_kind)
		torch.save(model.state_dict(), os.path.join(args.data_path, 'models', 'dataset_organ' + str(args.organ_id), \
			'vnetg_e' + str(e) + 'S' + str(args.slices) + 'FD' + str(args.folds) + str(args.current_fold) + '_' + args.timestamp + '.pkl'))
	print('#' * 10 , 'end of training stage!')

if __name__ == '__main__':
	model = Vnet(slices=args.slices, inchans=[16,64,128], outchans=[64,128,256], down_layers=[3,3,3], up_layers=[3,3,3], dilations=[2,2,2])
	model = model.cuda()
	train_criterion = OHEM()
	valid_criterion = nn.MSELoss(reduce=False)
	optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)

	model_parameters = filter(lambda p: p.requires_grad, model.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])
	print('model parameters:', params, 'training batches', len(trainloader), 'valid batches', len(testloader))
	period = 20

	train()
