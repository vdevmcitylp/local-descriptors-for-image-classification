import pickle

import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.externals import joblib
from sklearn.metrics import precision_recall_curve


def heaviside(x):
	return x > 0

def csLBP():

	# Get image dimensions
	

	zeroHorizontal = np.zeros(34).reshape(1, 34)
	zeroVertical = np.zeros(32).reshape(32, 1)

	trainSet = []

	with open('../Grayscale/gray', 'rb') as f:
		grayscaleImgList = cPickle.load(f)

	grayscaleImgList = np.array(grayscaleImgList)	

	for img in grayscaleImgList:

		img = np.concatenate((img, zeroVertical), axis=1)
		img = np.concatenate((zeroVertical, img), axis=1)
		img = np.concatenate((zeroHorizontal, img), axis=0)
		img = np.concatenate((img, zeroHorizontal), axis=0)

		cslbpImg = np.zeros((33, 33))
		for x in range(1, 33):
			for y in range(1, 33):
				
				s1 = heaviside(img[x-1, y-1] - img[x+1, y+1])
				s2 = heaviside(img[x-1, y] - img[x+1, y])*2 
				s3 = heaviside(img[x-1, y+1] - img[x+1, y-1])*4 
				s4 = heaviside(img[x, y+1] - img[x, y-1])*8

				s = s1 + s2 + s3 + s4

				cslbpImg[x, y] = s

		cslbpImg = cslbpImg[1:33, 1:33].astype(int) 	
		plt.imshow(cslbpImg, cmap='gray')
		plt.show()
		hist = np.zeros(16).astype(int)

		cslbpImg = cslbpImg.flatten()
		for i in cslbpImg:
			hist[i] = hist[i] + 1

		trainSet.append(hist)

		# plt.hist(cslbpImg, bins=np.arange(256))
		# plt.show()

	trainSet = np.array(trainSet)

	fo = open("feature_cslbp", "wb")
	cPickle.dump(trainSet, fo)

csLBP()
# csLBPTest()
