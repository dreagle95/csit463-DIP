# import the necessary packages
from skimage.measure import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

home_path = os.getcwd()
images_path = os.path.join(home_path, 'stopClassification2')
images_path = os.path.join(home_path, 'warnClassification2')


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def compare_images(imageA, imageB):
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB)

    # setup the figure
    fig = plt.figure()
    plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))

    # show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap=plt.cm.gray)
    plt.axis("off")

    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap=plt.cm.gray)
    plt.axis("off")

    # show the images
    plt.show()
def knn(pot_sign):
    # load the images
     stop_path = os.path.join(home_path, 'stopClassification2')
     warn_path = os.path.join(home_path, 'warnClassification2')

     for im in os.path.join(stop_path, os.listdir(stop_path)):
        stop_sign = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        compare_images(pot_sign, stop_sign)

     for im in os.path.join(warn_path, os.listdir(warn_path)):
        warn_sign = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        compare_images(pot_sign, warn_sign)
    # convert the images to grayscale