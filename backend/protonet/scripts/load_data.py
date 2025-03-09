#!/usr/bin/env python
# coding: utf-8
import os 
import matplotlib.image as mpimg
import numpy as np
import csv
#将图片数据转化为numpy，每一个类得数据被为训练集和测试集，并存储在字典中

os.chdir('D:\Python\Projects\calligraphy-ratings\protonet/scripts')
def load_data():
	labels_trainData = {}
	label = 0
	for file in os.listdir('D:\Python\Projects\calligraphy-ratings\protonet/styletrain'):
		for dir in os.listdir('D:\Python\Projects\calligraphy-ratings\protonet/styletrain/' + file):
			labels_trainData[label] = []
			data = []
			for png in os.listdir('D:\Python\Projects\calligraphy-ratings\protonet/styletrain/' + file +'/' + dir):
				image_np = mpimg.imread('D:\Python\Projects\calligraphy-ratings\protonet/styletrain/' + file +'/' + dir+'/' +png)
				image_np.resize(105,105)
				image_np.astype(np.float64)
				data.append(image_np)
			labels_trainData[label] = np.array(data)
			label += 1
	labels_testData = {}
	label = 0
	for file in os.listdir('D:\Python\Projects\calligraphy-ratings\protonet/styletest'):
		for dir in os.listdir('D:\Python\Projects\calligraphy-ratings\protonet/styletest/' + file):
			labels_testData[label] = []
			data = []
			for png in os.listdir('D:\Python\Projects\calligraphy-ratings\protonet/styletest/' + file +'/' + dir):
				image_np = mpimg.imread('D:\Python\Projects\calligraphy-ratings\protonet/styletest/' + file +'/' + dir+'/' +png)
				image_np.resize(105,105)
				image_np.astype(np.float64)
				data.append(image_np)
			labels_testData[label] = np.array(data)
			label += 1            
	return labels_trainData ,labels_testData
