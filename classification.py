
import os
import os.path as osp
import argparse
import numpy as np

from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

def run(x_train, x_test, y_train, y_test):

	# Split train into train-validation set

	model = XGBClassifier(n_estimators = 800)
	model.fit(x_train, y_train)

	# make predictions for test data
	y_pred = model.predict(x_test)
	predictions = [round(value) for value in y_pred]

	# Evaluate predictions
	accuracy = accuracy_score(y_test, predictions)
	print("Test accuracy: {}".format(accuracy))

if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument("--operator", choices = ["cslbp", "csldp", "csldmp", "cslmp", "csltp", "xcslbp", "xcslmp", "xcsltp"], default = "cslbp")

	args = parser.parse_args()

	with open(osp.join("features", "{}_train_features.npy".format(args.operator)), "rb") as handle:
		x_train = np.load(handle)

	with open(osp.join("features", "{}_test_features.npy".format(args.operator)), "rb") as handle:
		x_test = np.load(handle)

	with open(osp.join("dataset", "train_labels.npy"), "rb") as handle:
		y_train = np.load(handle)

	with open(osp.join("dataset", "test_labels.npy"), "rb") as handle:
		y_test = np.load(handle)

	run(x_train, x_test, y_train, y_test)
