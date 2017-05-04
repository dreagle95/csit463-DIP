import cv2
import numpy as np
import os

stop_temp_path = os.path.join(os.getcwd(), 'stopTemplate')
warn_temp_path = os.path.join(os.getcwd(), 'warnTemplate')
#stopTemplate = cv2.cvtColor(cv2.imread(stop_temp_path + '\\template.jpg'), cv2.COLOR_RGB2GRAY)
#warnTemplate = cv2.cvtColor(cv2.imread(warn_temp_path + '\\template.jpg'), cv2.COLOR_RGB2GRAY)
stopTemplate = cv2.imread(stop_temp_path + '\\template.jpg')
warnTemplate = cv2.imread(warn_temp_path + '\\template.jpg')
templates = [stopTemplate, warnTemplate]

#Mean Squared Error between 2 images
def mse(imageA, imageB):
    cv2.cvtColor(imageA, cv2.COLOR_RGB2HSV)
    cv2.cvtColor(imageB, cv2.COLOR_RGB2HSV)
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
    #copy = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2GRAY)
    copy = image.copy()
    aveError = []
    classification = ""
    for template in templates:
        aveError.append(sum((mse(template, copy)))/3)

    #returns the index position of the smaller error yielding potential
    highestPotential = minimum(aveError)

    if highestPotential == 0:
        if aveError[highestPotential] < MAX_AVE:
            #its a stopsign
            classification = "StopSign"
        else:
            #its a false positive
            classification = "False-positive"
    if highestPotential == 1:
        if aveError[highestPotential] < MAX_AVE:
            #its a warning sign
            classification = "WarningSign"
        else:
            #its a false positive
            classification = "False-positive"
    #print(classification)
    return classification








    """ Below code was for testing, probably will be ignored"""




        # cv2.imshow("copy", copy)
        # entry = cv2.waitKey(0)
        # if template is stopTemplate:
        #     print("stopsign", error, chr(entry))
        #     entry = ""
        # if template is warnTemplate:
        #     print("warning sign", error, chr(entry))
        #     entry = ""
        # cv2.destroyAllWindows()
    # w, h = template.shape[::-1]
    # # apply template matching
    # # starting to go through the template dictionary to run over
    # # the image, with convolution, searching for a match of the template
    # res = cv2.matchTemplate(copy, template, cv2.TM_SQDIFF_NORMED)
    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # top_left = min_loc
    # bottom_right = (top_left[0] + w, top_left[1] + h)
    # cv2.rectangle(copy, top_left, bottom_right, 255, 2)
    # #cv2.imshow("res", res)
    # # cv2.imshow("res", copy)
    # #cv2.waitKey(0)
    # # cv2.destroyAllWindows()