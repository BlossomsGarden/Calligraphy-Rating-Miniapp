#!/usr/bin/env python
# coding: utf-8
import sys
import cv2 as cv

sys.path.append('D:\Python\Projects\calligraphy-ratings\protonet\protonets')
from protonets_net import Protonets

import os 
os.chdir('D:\Python\Projects\calligraphy-ratings\protonet\scripts')
from protonet.scripts.load_data import load_data

import numpy as np
from utils import del_files


def train_protonet(img):
	# 写入图片
	del_files('protonet/styletest/1/rec')
	cv.imwrite('protonet/styletest/1/rec/png.png', img)

	# 载入数据
	labels_trainData ,labels_testData = load_data()
	class_number = max(list(labels_trainData.keys()))
	wide = labels_trainData[0][0].shape[0]
	length = labels_trainData[0][0].shape[1]
	for label in labels_trainData.keys():
		labels_trainData[label] = np.reshape(labels_trainData[label], [-1, 1, wide, length])
	for label in labels_testData.keys():
		labels_testData[label] = np.reshape(labels_testData[label], [-1, 1, wide, length])


	# 根据需求修改类的初始化参数，参数含义见protonets_net.py
	protonets = Protonets((1,wide,length),5,7,7,3,'protonet/log/',500)
	return protonets.evaluation_model(labels_testData,labels_trainData)


