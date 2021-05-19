import os
import os.path as osp

import numpy as np

def data(grayscale_root):

	with open(osp.join(grayscale_root, "train_images.npy"), "rb") as handle:
		x_train = np.load(handle)

	with open(osp.join(grayscale_root, "test_images.npy"), "rb") as handle:
		x_test = np.load(handle)

	return x_train, x_test