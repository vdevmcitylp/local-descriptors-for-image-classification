# local-descriptors-for-image-classification

UPDATE: My paper has been accepted! Check it out [here](https://www.igi-global.com/article/center-symmetric-local-descriptors-for-image-classification/217023).

Each file implements one of the variants of the [Local Binary Pattern (LBP)](http://jultika.oulu.fi/files/isbn9514270762.pdf).

1. [Center Symmetric LBP](http://www.ee.oulu.fi/mvg/files/pdf/pdf_750.pdf)
2. [Center Symmetric Local Derivative Pattern](https://ieeexplore.ieee.org/document/6011859/) 
3. Center Symmetric Local Derivative Mapped Pattern
4. [Center Symmetric Local Mapped Pattern](https://dl.acm.org/citation.cfm?id=2554895)
5. [Center Symmetric Local Ternary Pattern](https://www.computer.org/csdl/proceedings/cvpr/2010/6984/00/05540195-abs.html)
6. [Extended Center Symmetric Local Binary Pattern](https://hal.archives-ouvertes.fr/hal-01227955/document)
7. Extended Center Symmetric Local Mapped Pattern
8. [Extended Center Symmetric Local Ternary Pattern](https://link.springer.com/chapter/10.1007/978-3-642-23321-0_56)


## Setup

### Environment 
```
- Python 3.7.6
- Ubuntu 16.04
```

Preferably, create a new environment using [virtualenv](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

### Requirements

After activating the virtual environment, run the following command to install dependencies.

```
pip install -r requirements.txt
```

### Datset Setup

Download the CIFAR-10 dataset from [here]() and extract in the `dataset` directory.

For CIFAR-10, run this command to convert the images to grayscale.

```
python data_helper.py
```

This is a helper function for loading, preprocessing and saving the preprocessed images to disk for the CIFAR-10 dataset

Please write your own function for a custom dataset which,
- Converts images to grayscale (you can use OpenCV functions)
- Saves the converted images to disk

<hr>

All the files in ```operators/``` have the same underlying structure with the only difference being in the algorithm being implemented.
All algorithms are trained and tested on the CIFAR-10 dataset.

## To compute features

    python main.py --operator cslbp --img_height 32 --img_width 32

The above command will compute CS-LBP features, check the `operator` argument for more choices.

## Brief Explanation

### Converting to Grayscale

These local descriptors require the input to be in grayscale.

    grayscaleImg = (imRed*0.3 + imGreen*0.59 + imBlue*0.11).astype(int)

### The Algorithm

First, I pad the image with zeros before running the algorithm. 

    img = np.concatenate((img, zeroVertical), axis=1)
    img = np.concatenate((zeroVertical, img), axis=1)
    img = np.concatenate((zeroHorizontal, img), axis=0)
    img = np.concatenate((img, zeroHorizontal), axis=0)

The function then goes on to implement the respective algorithm (CS-LBP in this case). 

    pattern_img = np.zeros((img_height + 1, img_width + 1))
        
        for x in range(1, img_height + 1):
            for y in range(1, img_width + 1):
                
                s1 = threshold(img[x-1, y-1] - img[x+1, y+1])
                s2 = threshold(img[x-1, y] - img[x+1, y])*2 
                s3 = threshold(img[x-1, y+1] - img[x+1, y-1])*4 
                s4 = threshold(img[x, y+1] - img[x, y-1])*8

                s = s1 + s2 + s3 + s4

                pattern_img[x, y] = s

We then compute the histogram of the resultant image to get the feature vector.

    histogram = np.histogram(pattern_img, bins = np.arange(17))[0]

### Classification

    python classification.py --operator cslbp

I'm using the [XGBoost](https://xgboost.readthedocs.io/en/latest/) Classifier for classification.

    model = XGBClassifier(n_estimators=800)
    model.fit(X_train, y_train)

You have to play with *n_estimators* to get the best accuracy.

And that's about it! Feel free to open an issue if required.
