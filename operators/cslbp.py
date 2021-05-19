""" Center Symmetric LBP """

from tqdm import tqdm
import numpy as np

def threshold(x):
	return x > 0

def get_features(images, img_height, img_width):
	
	zeroHorizontal = np.zeros(img_width + 2).reshape(1, img_width + 2)
	zeroVertical = np.zeros(img_height).reshape(img_height, 1)

	features = []

	for img in tqdm(images):

		img = np.concatenate((img, zeroVertical), axis = 1)
		img = np.concatenate((zeroVertical, img), axis = 1)
		img = np.concatenate((zeroHorizontal, img), axis = 0)
		img = np.concatenate((img, zeroHorizontal), axis = 0)

		pattern_img = np.zeros((img_height + 1, img_width + 1))
		
		for x in range(1, img_height + 1):
			for y in range(1, img_width + 1):
				
				s1 = threshold(img[x-1, y-1] - img[x+1, y+1])
				s2 = threshold(img[x-1, y] - img[x+1, y])*2 
				s3 = threshold(img[x-1, y+1] - img[x+1, y-1])*4 
				s4 = threshold(img[x, y+1] - img[x, y-1])*8

				s = s1 + s2 + s3 + s4

				pattern_img[x, y] = s

		pattern_img = pattern_img[1:(img_height+1), 1:(img_width+1)].astype(int) 	
		histogram = np.histogram(pattern_img, bins = np.arange(17))[0]
		histogram = histogram.reshape(1, -1)

		features.append(histogram)

	features = np.concatenate(features, axis = 0)

	return features
	