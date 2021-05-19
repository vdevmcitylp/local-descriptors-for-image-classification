""" 
I recommend that you use OpenCV functions to convert to grayscale instead of this for a custom dataset.
This will work for CIFAR-10.
"""

import numpy as np

def colour2grayscale(images: np.array, 
	num_channels: int,
	img_width: int,
	img_height: int,
	mode: str = "train"):
	"""
	Accepts a batch of RBG colour images and converts them to grayscale
	Shape: num_images x (num_pixels x num_channels)
	Will probably only work for CIFAR-10

	The *colour2grayscale* function implements the conversion formula as defined in http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0029740

	For example, 10 images of shape 32x32x3 => 10 x (32 x 32 x 3)
	
	Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
	"""

	grayscale_images = []

	num_images, num_pixels = images.shape # num_images x 3072
	num_pixels_per_channel = num_pixels // num_channels # 1024

	for (i, img) in enumerate(images):
			
		imRed = img[:num_pixels_per_channel].reshape(img_height, img_width)
		imGreen = img[num_pixels_per_channel : 2*num_pixels_per_channel].reshape(img_height, img_width)
		imBlue = img[2*num_pixels_per_channel:].reshape(img_height, img_width)

		grayscale_img = (imRed*0.3 + imGreen*0.59 + imBlue*0.11).astype(int)

		grayscale_images.append(grayscale_img.reshape(1, img_height, img_width))

	# import pdb
	# pdb.set_trace()

	grayscale_images = np.concatenate(grayscale_images, axis = 0)

	return grayscale_images


