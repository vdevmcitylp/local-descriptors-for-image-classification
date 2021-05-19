
import load_data


def main():

	# Load train data
	x_train, y_train = load_data.train()

	# Load test data
	x_test, y_test = load_data.test()
	
	# Preprocess: Convert to grayscale

	x_train = preprocess.convert2grayscale(x_train)
	x_test = preprocess.convert2grayscale(x_test)

	# Save grayscale images to disk
	fo = open('gray', 'wb')
	pickle.dump(train, fo)


if __name__ == '__main__':
	main()