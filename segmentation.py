import cv2
import matplotlib as mpl
import numpy as np
import scipy as pi


def segment_sign(im):
    # list of boundaries
    lower_warning = np.array([51, 93.3, 29.4], dtype="uint8")
    upper_warning = np.array([43, 71.7, 94.3], dtype="uint8")
    lower_stop = np.array([0, 100, 19.2], dtype="uint8")
    upper_stop = np.array([359, 78, 100], dtype="uint8")

    boundaries = [
        # Yellow boundaries for warning signs
        ([5, 65, 75], [68, 190, 240]),
        # red boundaries for stop signs
        ([0,0,29], [60,56,255])
    ]

    # hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    # find colors within the specified boundaries
    # and apply the mask
    mask = np.zeros((len(im), len(im[0])), dtype=np.uint8)
    for (upper, lower) in boundaries:
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        temp_mask = cv2.inRange(im, lower, upper)
        mask = mask + temp_mask
        print(mask.shape, temp_mask.shape)

    output = cv2.bitwise_and(im, im, mask=mask)
    imgray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("imgray", imgray)
    ret, thresh = cv2.threshold(imgray, 20, 255, 0)

    # show the images
    cv2.imshow("images", np.hstack([im, output]))
    # cv2.imshow("thresh", thresh)
    cv2.waitKey(0)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))

    cv2.drawContours(im, contours, -1, (255, 0, 0), 3)
    cv2.imshow("image", im)
    cv2.waitKey(0)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("turd", im)
    cv2.waitKey(0)