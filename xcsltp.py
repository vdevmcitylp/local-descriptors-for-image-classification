import cPickle
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

def heavyside(x, y):
	
	if x > 3 and y > 3:
		return 2
	elif x < -3 and y < -3:
		return 1
	else:
		return 0

def xcsLTP():

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

				s1 = heavyside(img[x-1, y-1]-img[x, y], img[x, y]-img[x+1, y+1]) 
				#s2 = heavyside((img[x-1, y]-img[x+1, y]+img[x, y])+(img[x-1, y]-img[x, y])*(img[x+1, y]-img[x, y]))*3 
				s3 = heavyside(img[x-1, y+1]-img[x, y], img[x, y]-img[x+1, y-1])*3
				#s4 = heavyside((img[x, y+1]-img[x, y-1]+img[x, y])+((img[x, y+1]-img[x, y])*(img[x, y-1]-img[x, y])))*27

				#s = s1 + s2 + s3 + s4
				s = s1 + s3

				xcslbpImg[x, y] = s

		xcslbpImg = xcslbpImg[1:33, 1:33].astype(int) 	
		plt.imshow(img, cmap = 'gray')
		plt.show()
		plt.imshow(xcslbpImg, cmap = 'gray')
		plt.show()
		hist = np.zeros(9).astype(int)

		xcslbpImg = xcslbpImg.flatten()
		for i in xcslbpImg:
			hist[i] = hist[i] + 1

		trainSet.append(hist)

		# plt.hist(cslbpImg, bins=np.arange(256))
		# plt.show()

	trainSet = np.array(trainSet)

	fo = open("feature_xcsltp", "wb")
	cPickle.dump(trainSet, fo)

def xcsLTPTest():

	zeroHorizontal = np.zeros(34).reshape(1, 34)
	zeroVertical = np.zeros(32).reshape(32, 1)

	trainSet = []

	with open('../Grayscale/gray_test', 'rb') as f:
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
				
				s1 = heavyside(img[x-1, y-1]-img[x, y], img[x, y]-img[x+1, y+1]) 
				#s2 = heavyside((img[x-1, y]-img[x+1, y]+img[x, y])+(img[x-1, y]-img[x, y])*(img[x+1, y]-img[x, y]))*3 
				s3 = heavyside(img[x-1, y+1]-img[x, y], img[x, y]-img[x+1, y-1])*3
				#s4 = heavyside((img[x, y+1]-img[x, y-1]+img[x, y])+((img[x, y+1]-img[x, y])*(img[x, y-1]-img[x, y])))*27

				#s = s1 + s2 + s3 + s4
				s = s1 + s3

				xcslbpImg[x, y] = s

		xcslbpImg = xcslbpImg[1:33, 1:33].astype(int) 	

		hist = np.zeros(9).astype(int)

		xcslbpImg = xcslbpImg.flatten()
		for i in xcslbpImg:
			hist[i] = hist[i] + 1

		trainSet.append(hist)

		# plt.hist(cslbpImg, bins=np.arange(256))
		# plt.show()

	trainSet = np.array(trainSet)

	fo = open("feature_test_xcsltp", "wb")
	cPickle.dump(trainSet, fo)

def classification(labels, testLabels):

	with open("feature_xcsltp", 'rb') as f:
		X_train = cPickle.load(f)

	with open("feature_test_xcsltp", "rb") as f:
		X_test = cPickle.load(f)

	y_train = labels
	y_test = testLabels
	# split data into train and test sets
	# seed = 7
	# test_size = 0.33
	# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

	model = XGBClassifier(n_estimators=800)
	model.fit(X_train, y_train)

	# make predictions for test data
	y_pred = model.predict(X_test)
	predictions = [round(value) for value in y_pred]

	# evaluate predictions
	accuracy = accuracy_score(y_test, predictions)
	print "Accuracy: %.2f%%" % (accuracy * 100.0)

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

xcsLTP()
#xcsLTPTest()

#classification(labels, testLabels)	
