import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join
import scipy as pi, scipy.misc
from segmentation import segment_sign


def main():
    # initialize an empty array
    ims = []
    # define some path variables to attain the pictures
    images_path = os.path.join(os.getcwd(), 'sign-dataset')
    stop_path = os.path.join(images_path, 'stop-signs')
    speed_path = os.path.join(images_path, 'speed-signs')
    yield_path = os.path.join(images_path, 'yield-signs')
    warning_path = os.path.join(images_path, 'warning-signs')

    # add 40 images, 10 from each sign, into the array
    for i in range(1, 44):
        # ims.append(cv2.imread(join(speed_path, os.listdir(speed_path)[i])))
        # ims.append(cv2.imread(join(stop_path, os.listdir(stop_path)[i])))
        # ims.append(cv2.imread(join(yield_path, os.listdir(yield_path)[i])))
        ims.append(cv2.imread(join(warning_path, os.listdir(warning_path)[i])))

    for im in ims:
        potential_signs = segment_sign(im)





if __name__ == '__main__':
    main()


