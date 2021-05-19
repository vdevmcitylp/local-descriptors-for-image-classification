""" Load dataset"""

import pickle

def unpickle(file):
	
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding = 'bytes')
	return dict

def train():
	"""
	Returns x_train and y_train for the CIFR-10 dataset
	"""
	
	x_train = []
	y_train = []

	for batch_id in range(1, 6):

		batch = unpickle("dataset/data_batch_{}".format(batch_id))
		x_train.append(batch["data"])
		y_train.append(batch["labels"])

	return x_train, y_train

def test():
	"""
	Returns x_test and y_test for the CIFR-10 dataset
	"""	
	
	test_data = unpickle('test_batch')
	
	x_test = test_data["data"]
	y_test = test_data["labels"]

	return x_test, y_test