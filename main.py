import cv2
from classify_cnn import classify_DNN
import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join
import scipy as pi, scipy.misc
from segmentation import segment_sign
from normalize import normalize
from template_matching import temp_match

def main():
    # initialize an empty array
    ims = []
    # define some path variables to attain the pictures
    images_path = join(os.getcwd(), 'good-data')
    stop_temp_path = join(os.getcwd(), 'stopTemplate')
    warn_temp_path = join(os.getcwd(), 'warnTemplate')

    # loading in the templates
    stopTemplate = cv2.cvtColor(cv2.imread(join(stop_temp_path, 'template.jpg')), cv2.COLOR_BGR2GRAY)
    warnTemplate = cv2.cvtColor(cv2.imread(join(warn_temp_path, 'template.jpg')), cv2.COLOR_BGR2GRAY)

    templates =[stopTemplate, warnTemplate]

    # reading all the images from the good data set and saving to a list
    for i in os.listdir(images_path):
        image = cv2.imread(join(images_path, i))
        if image is not None:
            # iterating through that list to apply segmentation to the images and send
            # what was segmented to the classifiers
            potential_signs = segment_sign(image)

            # because we developed these individually,
            # we normalized to different size images
            template_signs = normalize(potential_signs, (28,28))
            nn_signs = normalize(potential_signs, (64,64))

            # perform template matching
            for sign in template_signs:
                classification = temp_match(sign)
                if classification == "StopSign" or classification == "WarningSign":
                    cv2.imshow("sign", sign)
                    print(classification)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

            # predict with the neural network
            predictions = classify_DNN(nn_signs)
            print(predictions)
            for i, prediction in enumerate(predictions):
                cv2.imshow("Sign:", potential_signs[i])
                print("Stop Sign" if prediction == 0 else
                      "Warning Sign" if prediction == 1  else "False Positive")
                cv2.waitKey(0)



if __name__ == '__main__':
    main()