

def train():
	pass

def validate():
	pass

def test():
	pass


def classification(labels, testLabels):

	# Load features

	with open("feature_cslbp", 'rb') as f:
		X_train = cPickle.load(f)

	with open("feature_test_cslbp", "rb") as f:
		X_test = cPickle.load(f)

	y_train = np.array(labels)
	y_test = np.array(testLabels)
	
	# Split train into train-validation set




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
	# print "Accuracy: " + accuracy
	print(accuracy)


with open("../Labels/train_labels", "rb") as f:
	labels = pickle.load(f)

with open("../Labels/test_labels", "rb") as f:
	testLabels = pickle.load(f)

#classification(labels, testLabels)	
