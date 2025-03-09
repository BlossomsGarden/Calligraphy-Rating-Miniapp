#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import h5py
import random
import csv

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

os.chdir('D:\Python\Projects\calligraphy-ratings\protonet\protonets')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def eucli_tensor(x,y):	#计算两个tensor的欧氏距离，用于loss的计算
	return -1*torch.sqrt(torch.sum((x-y)*(x-y))).view(1)


class CNNnet(torch.nn.Module):
	def __init__(self, input_shape, outDim):
		super(CNNnet, self).__init__()
		self.conv1 = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=input_shape[0],
							out_channels=16,
							kernel_size=3,
							stride=1,
							padding=1),
			torch.nn.BatchNorm2d(16),
			torch.nn.MaxPool2d(2),
			torch.nn.ReLU()
		)
		self.conv2 = torch.nn.Sequential(
			torch.nn.Conv2d(16, 32, 3, 1, 1),
			torch.nn.BatchNorm2d(32),
			nn.MaxPool2d(2),
			torch.nn.ReLU()
		)
		self.conv3 = torch.nn.Sequential(
			torch.nn.Conv2d(32, 64, 3, 1, 1),
			torch.nn.BatchNorm2d(64),
			nn.MaxPool2d(2),
			torch.nn.ReLU()
		)
		self.conv4 = torch.nn.Sequential(
			torch.nn.Conv2d(64, 64, 3, 1, 1),
			torch.nn.BatchNorm2d(64),
			# nn.MaxPool2d(2)
			torch.nn.ReLU()
		)
		self.conv5 = torch.nn.Sequential(
			torch.nn.Conv2d(64, 64, 3, 1, 1),
			torch.nn.BatchNorm2d(64),
			# nn.MaxPool2d(2)
			torch.nn.ReLU()
		)
		self.mlp1 = torch.nn.Linear(10816, 125)  # '''此处修改torch.nn.Linear(x,125)中的x位置'''
		self.mlp2 = torch.nn.Linear(125, outDim)

	def forward(self, x):  # '''根据__init__做相应修改'''
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = self.conv5(x)
		x = self.mlp1(x.view(x.size(0), -1))
		x = self.mlp2(x)
		return x


class Protonets(object):
	def __init__(self, input_shape, outDim, Ns, Nq, Nc, log_data, step, trainval=True):
		# Ns:支持集数量,Nq：查询集数量,Nc：每次迭代所选类数,log_data：模型和类对应的中心所要储存的位置,step:若trainval==True则读取已训练的第step步的模型和中心,trainval：是否从新开始训练模型
		self.input_shape = input_shape
		self.outDim = outDim
		self.batchSize = 1
		self.Ns = Ns
		self.Nq = Nq
		self.Nc = Nc
		if trainval == False:
			# 若训练一个新的模型，初始化CNN和中心点
			self.center = {}
			self.model = CNNnet(input_shape, outDim).cuda()
		else:
			# 否则加载CNN模型和中心点
			self.center = {}
			self.model = torch.load(log_data + 'model_net_' + str(step) + '.pkl')  # '''修改,存储模型的文件名'''
			self.load_center(log_data + 'model_center_' + str(step) + '.csv')  # '''修改,存储中心的文件名'''

	def compute_center(self, data_set):  # data_set是一个numpy对象，是某一个支持集，计算支持集对应的中心的点
		center = 0
		for i in range(self.Ns):
			data = np.reshape(data_set[i], [1, self.input_shape[0], self.input_shape[1], self.input_shape[2]])
			data = Variable(torch.from_numpy(data)).cuda()
			data = self.model(data)[0]  # 将查询点嵌入另一个空间
			if i == 0:
				center = data
			else:
				center += data
		center /= self.Ns
		return center

	def train(self, labels_data, class_number):  # 网络的训练
		# Select class indices for episode
		class_index = list(range(class_number))
		random.shuffle(class_index)
		choss_class_index = class_index[:self.Nc]
		sample = {'xc': [], 'xq': []}
		for label in choss_class_index:
			D_set = labels_data[label]
			# 从D_set随机取支持集和查询集
			support_set, query_set = self.randomSample(D_set)
			# 计算中心点
			self.center[label] = self.compute_center(support_set)
			# 将支持集和查询集存储在list中
			sample['xc'].append(self.center[label])  # list
			sample['xq'].append(query_set)
		# 优化器
		optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
		optimizer.zero_grad()
		protonets_loss = self.loss(sample)
		protonets_loss.backward()
		optimizer.step()

	def loss(self, sample):  # 自定义loss
		loss_1 = autograd.Variable(torch.FloatTensor([0])).cuda()
		for i in range(self.Nc):
			query_dataSet = sample['xq'][i]
			for n in range(self.Nq):
				data = np.reshape(query_dataSet[n], [1, self.input_shape[0], self.input_shape[1], self.input_shape[2]])
				data = Variable(torch.from_numpy(data)).cuda()
				data = self.model(data)[0]  # 将查询点嵌入另一个空间
				# 查询点与每个中心点逐个计算欧氏距离
				predict = 0
				for j in range(self.Nc):
					center_j = sample['xc'][j]
					if j == 0:
						predict = eucli_tensor(data, center_j)
					else:
						predict = torch.cat((predict, eucli_tensor(data, center_j)), 0)
				# 为loss叠加
				loss_1 += -1 * F.log_softmax(predict, dim=0)[i]
		loss_1 /= self.Nq * self.Nc
		return loss_1

	def randomSample(self, D_set):  # 从D_set随机取支持集和查询集
		index_list = list(range(D_set.shape[0]))
		random.shuffle(index_list)
		support_data_index = index_list[:self.Ns]
		query_data_index = index_list[self.Ns:self.Ns + self.Nq]
		support_set = []
		query_set = []
		for i in support_data_index:
			support_set.append(D_set[i])
		for i in query_data_index:
			query_set.append(D_set[i])
		return support_set, query_set


	def load_center(self, path):
		csvReader = csv.reader(open(path))
		for line in csvReader:
			label = int(line[0])
			center = [float(line[i]) for i in range(1, len(line))]
			center = np.array(center)
			center = Variable(torch.from_numpy(center)).cuda()
			self.center[label] = center

	def evaluation_model(self, labels_testData, labels_trainData):
		train_accury = []
		for y in self.center.keys():
			for data in labels_trainData[y]:
				data = np.reshape(data, [1, self.input_shape[0], self.input_shape[1], self.input_shape[2]])
				data = Variable(torch.from_numpy(data)).cuda()
				data = self.model(data)[0]  # 将test.data嵌入另一个空间
				predict = 0
				predict1 = 0
				predict2 = 0
				j2label_c = {}
				j = 0  # 第j个中心
				for label_c in self.center.keys():  # 计算data到每个中心点得距离
					center_j = self.center[label_c]
					j2label_c[j] = label_c

					if j == 0:
						predict = eucli_tensor(data, center_j)
					else:
						predict = torch.cat((predict, eucli_tensor(data, center_j)), 0)
					j += 1
				predict1 = predict
				predict2 = F.log_softmax(predict, dim=0)
				x = j2label_c
				y_pre_j = int(torch.argmax(F.log_softmax(predict, dim=0)))  # 离第j个中心最近
				y_pre = j2label_c[y_pre_j]  # 第j个中心对应得标签是y_pre
				train_accury.append(1 if y_pre == y else 0)

		test_accury = []
		for data in labels_testData[0]:
			data = np.reshape(data, [1, self.input_shape[0], self.input_shape[1], self.input_shape[2]])
			data = Variable(torch.from_numpy(data)).cuda()
			data = self.model(data)[0]  # 将test.data嵌入另一个空间
			predict = 0
			j2label_c = {}
			j = 0  # 第j个中心
			for label_c in self.center.keys():  # 计算data到每个中心点得距离
				center_j = self.center[label_c]
				j2label_c[j] = label_c

				if j == 0:
					predict = eucli_tensor(data, center_j)
				else:
					predict = torch.cat((predict, eucli_tensor(data, center_j)), 0)
				j += 1
			predict3 = predict
			predict4 = F.log_softmax(predict, dim=0)
			y_pre_j = int(torch.argmax(F.log_softmax(predict, dim=0)))  # 离第j个中心最近
			y_pre = j2label_c[y_pre_j]  # 第j个中心对应得标签是y_pre

			an = 0
			ary = predict3.tolist()
			for i in range(0, 4):
				an = an + ary[i]
			fimi = 100 - (max(ary) / an) * 100
			if y_pre == 1:
				# print("最像汉仪尚巍手书，相似度：%.2f%%"%fimi)
				return "汉仪尚巍手书", "HYShangWSHJ", "%.2f%%" % fimi
			elif y_pre == 2:
				# print("最像郑文公碑楷书，相似度：%.2f%%"%fimi)
				return "郑文公碑楷书", "FZZhengWGBKSJW", "%.2f%%" % fimi
			elif y_pre == 0:
				# print("最像钟齐陈伟勋行楷，相似度：%.2f%%"%fimi)
				return "钟齐陈伟勋行楷", "ZhongQiChenWeixun", "%.2f%%" % fimi
			elif y_pre == 3:
				# print("最像褚遂良楷书，相似度：%.2f%%"%fimi)
				return "褚遂良楷书", "FZChuSLKSJW", "%.2f%%" % fimi
			elif y_pre == 4:
				# print("最像柳公权楷书，相似度：%.2f%%"%fimi)
				return "柳公权楷书", "FZLiugongquan", "%.2f%%" % fimi
