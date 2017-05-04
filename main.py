import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join
import scipy as pi, scipy.misc
from segmentation import segment_sign
from normalize import normalize
from template_matching import temp_match
#
# def get_hogs(pots):
#     hog_desc = cv2.HOGDescriptor()
#
#     hogs = []
#     for pot in pots:
#         hog = hog_desc.computeGradient(pot)
#         hogs.append(hog_desc.compute(pot))
#         # print(len(hog))

def main():
    # initialize an empty array
    ims = []
    # define some path variables to attain the pictures
    images_path = os.path.join(os.getcwd(), 'temp')
    stop_temp_path = os.path.join(os.getcwd(), 'stopTemplate')
    warn_temp_path = os.path.join(os.getcwd(), 'warnTemplate')
    #loading in the templates
    stopTemplate = cv2.cvtColor(cv2.imread(stop_temp_path + '\\template.jpg'), cv2.COLOR_RGB2GRAY)
    warnTemplate = cv2.cvtColor(cv2.imread(warn_temp_path + '\\template.jpg'), cv2.COLOR_RGB2GRAY)

    templates =[stopTemplate, warnTemplate]

    #reading all the images from the good data set and saving to a list
    for i in os.listdir(images_path):
        image = cv2.imread(os.path.join(images_path, i))
        if image is not None:
            ims.append(image)


    #iterating through that list to apply segmentation to the images and send
    #what was segmented to the classifiers
    for im in ims:
        potential_signs = segment_sign(im)
        potential_signs = normalize(potential_signs)
        for sign in potential_signs:
            classification = temp_match(sign)
            cv2.imshow("sign", sign)
            print(classification)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # hog_features = get_hogs(potential_signs)

if __name__ == '__main__':
    main()