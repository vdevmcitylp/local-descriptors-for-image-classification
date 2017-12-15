import cPickle
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.externals import joblib
from sklearn.metrics import precision_recall_curve

def unpickle(file):
	
	fo = open(file, 'rb')
	dict = cPickle.load(fo)
	fo.close()
	return dict

def colour2grayscale(R, grayscaleImgList):

	# grayscaleImgList = []

	images = R['data']
	
	for i in xrange(10000):
		
		img = images[i, ]
		
		imRed = img[0:1024].reshape(32, 32)
		imGreen = img[1024:2048].reshape(32, 32)
		imBlue = img[2048:3072].reshape(32, 32)

		grayscaleImg = (imRed*0.3 + imGreen*0.59 + imBlue*0.11).astype(int)

		grayscaleImgList.append(grayscaleImg)

	#grayscaleImgList = np.array(grayscaleImgList)	
	
	return grayscaleImgList	
	# fo = open('gray1', 'wb')
	# cPickle.dump(grayscaleImgList, fo)

def heavyside(x):
	return x >= 0

def xcsLBP():

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

		xcslbpImg = np.zeros((33, 33))
		for x in range(1, 33):
			for y in range(1, 33):
				xcslbpImg[x, y] = heavyside((img[x-1, y-1]-img[x+1, y+1]+img[x, y])+(img[x-1, y-1]-img[x, y])*(img[x+1, y+1]-img[x, y])) + heavyside((img[x-1, y]-img[x+1, y]+img[x, y])+(img[x-1, y]-img[x, y])*(img[x+1, y]-img[x, y]))*2 + heavyside((img[x-1, y+1]-img[x+1, y-1]+img[x, y])+((img[x-1, y+1]-img[x, y])*(img[x+1, y-1]-img[x, y])))*4 + heavyside((img[x, y+1]-img[x, y-1]+img[x, y])+((img[x, y+1]-img[x, y])*(img[x, y-1]-img[x, y])))*8

		xcslbpImg = xcslbpImg[1:33, 1:33].astype(int) 	
		plt.imshow(img, cmap = 'gray')
		plt.show()
		plt.imshow(xcslbpImg, cmap = 'gray')
		plt.show()	
		hist = np.zeros(16).astype(int)

		xcslbpImg = xcslbpImg.flatten()
		for i in xcslbpImg:
			hist[i] = hist[i] + 1

		trainSet.append(hist)

		# plt.hist(cslbpImg, bins=np.arange(256))
		# plt.show()

	trainSet = np.array(trainSet)

	fo = open("feature_xcslbp", "wb")
	cPickle.dump(trainSet, fo)

def xcsLBPTest():

	zeroHorizontal = np.zeros(34).reshape(1, 34)
	zeroVertical = np.zeros(32).reshape(32, 1)

	trainSet = []

	with open('gray_test', 'rb') as f:
		grayscaleImgList = cPickle.load(f)

	grayscaleImgList = np.array(grayscaleImgList)	

	for img in grayscaleImgList:

		img = np.concatenate((img, zeroVertical), axis=1)
		img = np.concatenate((zeroVertical, img), axis=1)
		img = np.concatenate((zeroHorizontal, img), axis=0)
		img = np.concatenate((img, zeroHorizontal), axis=0)

		xcslbpImg = np.zeros((33, 33))
		for x in range(1, 33):
			for y in range(1, 33):
				xcslbpImg[x, y] = heavyside((img[x-1, y-1]-img[x+1, y+1]+img[x, y])+(img[x-1, y-1]-img[x, y])*(img[x+1, y+1]-img[x, y])) + heavyside((img[x-1, y]-img[x+1, y]+img[x, y])+(img[x-1, y]-img[x, y])*(img[x+1, y]-img[x, y]))*2 + heavyside((img[x-1, y+1]-img[x+1, y-1]+img[x, y])+((img[x-1, y+1]-img[x, y])*(img[x+1, y-1]-img[x, y])))*4 + heavyside((img[x, y+1]-img[x, y-1]+img[x, y])+((img[x, y+1]-img[x, y])*(img[x, y-1]-img[x, y])))*8

		xcslbpImg = xcslbpImg[1:33, 1:33].astype(int) 	

		hist = np.zeros(16).astype(int)

		xcslbpImg = xcslbpImg.flatten()
		for i in xcslbpImg:
			hist[i] = hist[i] + 1

		trainSet.append(hist)

		# plt.hist(cslbpImg, bins=np.arange(256))
		# plt.show()

	trainSet = np.array(trainSet)

	fo = open("feature_test_xcslbp", "wb")
	cPickle.dump(trainSet, fo)

def classification(labels, testLabels):

	with open("feature_xcslbp", 'rb') as f:
		X_train = cPickle.load(f)

	with open("feature_test_xcslbp", "rb") as f:
		X_test = cPickle.load(f)

	y_train = labels
	y_test = testLabels
	# split data into train and test sets
	# seed = 7
	# test_size = 0.33
	# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

	# model = XGBClassifier(n_estimators=400)
	# model.fit(X_train, y_train)

	#joblib.dump(model, "xcslbpmodel")
	model = joblib.load("xcslbpmodel")

	# make predictions for test data
	y_pred = model.predict(X_test)
	predictions = [round(value) for value in y_pred]

	# evaluate predictions
	accuracy = accuracy_score(y_test, predictions)
	print "Accuracy: %.2f%%" % (accuracy * 100.0)

	Y_train = label_binarize(y_train, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
	Y_test = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
	n_classes = Y_train.shape[1]

	Predictions = label_binarize(predictions, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

	print roc_auc_score(Y_test, Predictions)

	precision = dict()
	recall = dict()
	average_precision = dict()
	for i in range(10):
		precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i], Predictions[:, i])
		average_precision[i] = average_precision_score(Y_test[:, i], Predictions[:, i])

	# A "micro-average": quantifying score on all classes jointly
	precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(), Predictions.ravel())
	average_precision["micro"] = average_precision_score(Y_test, Predictions, average="micro")
	print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))

# R1 = unpickle('data_batch_1')
# R2 = unpickle('data_batch_2')
# R3 = unpickle('data_batch_3')
# R4 = unpickle('data_batch_4')
# R5 = unpickle('data_batch_5')

# RTest = unpickle('test_batch')

# labels = R1['labels'] + R2['labels'] + R3['labels'] + R4['labels'] + R5['labels']
# testLabels = RTest['labels']

# testX = colour2grayscale(RTest, [])
# train = colour2grayscale(R1, [])
# train = colour2grayscale(R2, train)	
# train = colour2grayscale(R3, train)
# train = colour2grayscale(R4, train)
# train = colour2grayscale(R5, train)

# fo = open('gray', 'wb')
# cPickle.dump(train, fo)

with open("../Labels/train_labels", "rb") as f:
	labels = cPickle.load(f)

with open("../Labels/test_labels", "rb") as f:
	testLabels = cPickle.load(f)

xcsLBP()
# xcsLBPTest()

#classification(labels, testLabels)	
