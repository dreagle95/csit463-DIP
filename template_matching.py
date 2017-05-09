import cv2
import numpy as np
import os
from os.path import join


stop_temp_path = os.path.join(os.getcwd(), 'stopTemplate')
warn_temp_path = os.path.join(os.getcwd(), 'warnTemplate')
stopTemplate = cv2.imread(join(stop_temp_path, 'template.jpg'))
warnTemplate = cv2.imread(join(warn_temp_path, 'template.jpg'))
templates = [stopTemplate, warnTemplate]

#Mean Squared Error between 2 images
def mse(imageA, imageB):
    cv2.cvtColor(imageA, cv2.COLOR_BGR2HSV)
    cv2.cvtColor(imageB, cv2.COLOR_BGR2HSV)
    hueerr = 0
    saterr = 0
    valerr = 0
    for j in range (0,28):
        for k in range (0,28):
            hueerr += np.sum((imageA[j][k][0].astype("float") - imageB[j][k][0].astype("float")) ** 2)
            saterr += np.sum((imageA[j][k][1].astype("float") - imageB[j][k][1].astype("float")) ** 2)
            valerr += np.sum((imageA[j][k][2].astype("float") - imageB[j][k][2].astype("float")) ** 2)

    hueerr /= float(imageA.shape[0] * imageA.shape[1])
    saterr /= float(imageA.shape[0] * imageA.shape[1])
    valerr /= float(imageA.shape[0] * imageA.shape[1])
    err = [hueerr, saterr, valerr]
    return err

def minimum(list):
    if list[0] < list[1]:
        return 0
    else:
        return 1

def temp_match(image):
    MAX_AVE = 5061
    # copy = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2GRAY)
    copy = image.copy()
    aveError = []
    classification = ""
    for template in templates:
        aveError.append(sum((mse(template, copy)))/3)

    # returns the index position of the smaller error yielding potential
    highestPotential = minimum(aveError)

    if highestPotential == 0:
        if aveError[highestPotential] < MAX_AVE:
            # its a stopsign
            classification = "StopSign"
        else:
            # its a false positive
            classification = "False-positive"
    if highestPotential == 1:
        if aveError[highestPotential] < MAX_AVE:
            # its a warning sign
            classification = "WarningSign"
        else:
            # its a false positive
            classification = "False-positive"
    # print(classification)
    return classification
