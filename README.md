# local-descriptors-for-image-classification

Each file implements one of the variants of the [Local Binary Pattern (LBP)](http://jultika.oulu.fi/files/isbn9514270762.pdf).

1. [Center Symmetric LBP](http://www.ee.oulu.fi/mvg/files/pdf/pdf_750.pdf)
2. [Center Symmetric Local Derivative Pattern](https://ieeexplore.ieee.org/document/6011859/) 
3. Center Symmetric Local Derivative Mapped Pattern
4. [Center Symmetric Local Mapped Pattern](https://dl.acm.org/citation.cfm?id=2554895)
5. [Center Symmetric Local Ternary Pattern](https://www.computer.org/csdl/proceedings/cvpr/2010/6984/00/05540195-abs.html)
6. [Extended Center Symmetric Local Binary Pattern](https://hal.archives-ouvertes.fr/hal-01227955/document)
7. Extended Center Symmetric Local Mapped Pattern
8. [Extended Center Symmetric Local Ternary Pattern](https://link.springer.com/chapter/10.1007/978-3-642-23321-0_56)

In collaboration with [Bhargav Parsi](https://bhargav265.github.io/bhargavparsi/), I've proposed two new local descriptors and the paper is under review, so fingers crossed! I'll release the code for them post acceptance.

All these files have the same underlying structure with the only difference being in the algorithm being implemented.
All algorithms are trained and tested on the CIFAR-10 dataset.
I'll go through the structure of each file now.

The first few lines are the necessary imports.

### Reading Input File

    def unpickle(file):
	
        fo = open(file, 'rb')
        dict = cPickle.load(fo)
        fo.close()
        return dict
        
This function reads in a CIFAR-10 pickle file and stores the data in a dictionary.

### Converting to Grayscale

These local descriptors require the input to be in grayscale. The *colour2grayscale* function implements the conversion formula as defined [here](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0029740).

G<sub>luminance</sub> = 0.3R + 0.59G + 0.11B

    grayscaleImg = (imRed*0.3 + imGreen*0.59 + imBlue*0.11).astype(int)


### Threshold Function

This varies from operator to operator and is defined in the *heaviside* function.

### The Algorithm

First, I pad the image with zeros before running the algorithm. 

    img = np.concatenate((img, zeroVertical), axis=1)
		img = np.concatenate((zeroVertical, img), axis=1)
		img = np.concatenate((zeroHorizontal, img), axis=0)
		img = np.concatenate((img, zeroHorizontal), axis=0)

The function then goes on to implement the respective algorithm (CS-LBP in this case). 

    cslbpImg = np.zeros((33, 33))
		for x in range(1, 33):
			for y in range(1, 33):
				
				s1 = heavyside(img[x-1, y-1] - img[x+1, y+1])
				s2 = heavyside(img[x-1, y] - img[x+1, y])*2 
				s3 = heavyside(img[x-1, y+1] - img[x+1, y-1])*4 
				s4 = heavyside(img[x, y+1] - img[x, y-1])*8

				s = s1 + s2 + s3 + s4

				cslbpImg[x, y] = s

We then compute the histogram of the resultant image to get the feature vector.

    hist = np.zeros(16).astype(int)

		cslbpImg = cslbpImg.flatten()
		for i in cslbpImg:
			hist[i] = hist[i] + 1

### Classification

I'm using the [XGBoost](https://xgboost.readthedocs.io/en/latest/) Classifier for classification.

    model = XGBClassifier(n_estimators=800)
	  model.fit(X_train, y_train)

You have to play with the *n_estimators* to get the best accuracy.

And that's about it! Feel free to open an issue if required.
