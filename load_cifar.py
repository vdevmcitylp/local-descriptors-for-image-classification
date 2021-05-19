""" Load CIFAR-10 dataset"""

import os.path as osp
import pickle

import numpy as np

def unpickle(file):
	
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding = 'bytes')
	return dict

def train():
	"""
	Returns x_train and y_train for the CIFAR-10 dataset
	"""
	
	x_train = []
	y_train = []

	cifar10_root = osp.join("dataset", "cifar-10-batches-py")

	for batch_id in range(1, 6):

		batch = unpickle(osp.join(
			cifar10_root, "data_batch_{}".format(batch_id)))
		
		x_train.append(batch[b"data"])
		y_train.append(batch[b"labels"])

	x_train = np.concatenate(x_train, axis = 0)
	y_train = np.concatenate(y_train, axis = 0)

	return x_train, y_train

def test():
	"""
	Returns x_test and y_test for the CIFR-10 dataset
	"""	
	cifar10_root = osp.join("dataset", "cifar-10-batches-py")

	test_data = unpickle(osp.join(cifar10_root, "test_batch"))
	
	x_test = test_data[b"data"]
	y_test = np.array(test_data[b"labels"])

	return x_test, y_test