"""
Helper function for loading, preprocessing and saving the preprocessed images to disk for the CIFAR-10 dataset

Write your own function for your own dataset.
- Convert images to grayscale
- Save the converted images to disk
"""

import os
import os.path as osp

import numpy as np

import load_cifar
import preprocess

def data_helper(num_channels, img_height, img_width):

	# Load train data
	x_train, y_train = load_cifar.train()	

	# Load test data
	x_test, y_test = load_cifar.test()
		
	# Preprocess: Convert to grayscale

	grayscale_root = osp.join("grayscale-images")
	if not osp.exists(grayscale_root):
		os.makedirs(grayscale_root)

	for mode in ["train", "test"]:

		save_file = osp.join(grayscale_root, "{}_images.npy".format(mode))
		
		print("Converting {} to grayscale ...".format(mode))
		
		if mode == "train":
			grayscale_images = preprocess.colour2grayscale(x_train, 
				num_channels, img_height, img_width)
			labels = y_train

		elif mode == "test":
			grayscale_images = preprocess.colour2grayscale(x_test, 
				num_channels, img_height, img_width)
			labels = y_test

		print("Converted!")

		with open(save_file, "wb") as handle:
			np.save(handle, grayscale_images)

		print("Saved grayscale images at {}.".format(save_file))

		with open(osp.join("dataset", "{}_labels.npy".format(mode)), "wb") as handle:
			np.save(handle, labels)

		print("Saved labels at dataset/{}_labels.npy".format(mode))

if __name__ == '__main__':

	# For CIFAR-10 images
	num_channels, img_width, img_height = 3, 32, 32

	data_helper(num_channels, img_height, img_width)