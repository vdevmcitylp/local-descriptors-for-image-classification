
import os
import os.path as osp
import argparse

import numpy as np

from operators import cslbp, csldp, csldmp, cslmp, csltp, xcslbp, xcslmp, xcsltp
import load_grayscale

def get_texture_operator(operator):

	texture_operator_dict = {
		"cslbp": cslbp, 
		# "csldp": csldp, 
		# "csldmp": csldmp, 
		# "cslmp": cslmp, 
		# "csltp": csltp, 
		# "xcslbp": xcslbp, 
		# "xcslmp": xcslmp, 
		# "xcsltp": xcsltp
	}

	return texture_operator_dict[operator]

def compute_features(args):

	# Load grayscale images
	x_train, x_test = load_grayscale.data("grayscale-images")
	
	print("Loaded grayscale images.\n")

	# Get corresponding texture operator
	texture_operator = get_texture_operator(args.operator)

	print("Using {} texture operator".format(args.operator))
	print("Computing features.\n")

	# This takes about 30-35 minutes for CIFAR-10, hence saving to disk.

	x_train = texture_operator.get_features(x_train, args.img_height, args.img_width)
	x_test = texture_operator.get_features(x_test, args.img_height, args.img_width)

	if not osp.exists("features"):
		os.makedirs("features")

	with open(osp.join("features", "{}_train_features.npy".format(args.operator)), "wb") as handle:
		np.save(handle, x_train)

	with open(osp.join("features", "{}_test_features.npy".format(args.operator)), "wb") as handle:
		np.save(handle, x_test)

	print("Computed features and saved to disk in 'features' directory.")

def main(args):

	compute_features(args)

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()

	parser.add_argument("--operator", choices = ["cslbp", "csldp", "csldmp", "cslmp", "csltp", "xcslbp", "xcslmp", "xcsltp"], default = "cslbp")

	parser.add_argument("--img_height", default = 32, type = int)
	parser.add_argument("--img_width", default = 32, type = int)

	args = parser.parse_args()

	main(args)