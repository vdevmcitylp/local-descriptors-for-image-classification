""" 
Center Symmetric Local Ternary Pattern 
Code might contain bugs
"""

from tqdm import tqdm
import numpy as np

def sigmoid(x):

	if x > 3:
		return 2
	elif x < -3:
		return 1
	else:
		return 0 

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
				
				s1 = sigmoid(img[x-2, y-2] - img[x+2, y+2])
				s3 = sigmoid(img[x-2, y+2] - img[x+2, y-2])*3
		
				s = s1 + s3
			
				pattern_img[x, y] = s

		pattern_img = pattern_img[1:(img_height+1), 1:(img_width+1)].astype(int) 	
		histogram = np.histogram(pattern_img, bins = np.arange(17))[0]
		histogram = histogram.reshape(1, -1)

		features.append(histogram)

	features = np.concatenate(features, axis = 0)

	return features
	