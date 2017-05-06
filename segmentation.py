import cv2
import matplotlib as mpl
import numpy as np
import scipy as pi
from kmeans_segmentation import kMeansSegmentation

"""
    Thought that if we brightened the image we might
    be able to get the signs with shadows better
    I think I was wrong
"""
def brighten(im):
    im = np.float32(im)
    maxIntensity = 255.0
    x = np.arange(maxIntensity)
    phi = 1
    theta = 1
    out = (maxIntensity/phi)*(im/(maxIntensity/theta))*0.75
    out= np.uint8(out)
    return out

"""
    This also does not work, often darkens the signs but increases
    contrast everywhere else
"""
def global_contrast_normalization(im, s, lmbda, epsilon):
    X = np.array(im)
    X_av = np.mean(X)
    X = X - X_av

    contrast = np.sqrt(lmbda + np.mean(X**2))
    X = s * X / max(contrast, epsilon)

    return np.uint8(im)

def segment_sign(image):
    im = image.copy()
    im = brighten(im)
    """Tried to remove noise, didn't help either"""


    """
        Tried to use CLAHE to adjust contrast in image
        For some reason, it is distorting heavily
    """
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    lab[:,:,0] = cv2.equalizeHist(lab[:,:,0])
    clahe1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6,6))
    clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(65,65))
    lab[:,:,0] = clahe2.apply(lab[:,:,0])
    im = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    # im = global_contrast_normalization(im, 1, 10, 0.0000000001)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    hsv[:,:,1] = cv2.equalizeHist(hsv[:,:,1])
    hsv[:,:,1] = clahe1.apply(hsv[:,:,1])
    # hsv[:,:,2] = clahe1.apply(hsv[:,:,2])
    # hsv = cv2.fastNlMeansDenoisingColored(hsv, None, 10, 10, 7, 7)
    im = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    hsv_colors = {
        "yellow": ((np.array([18, 165, 25]), np.array([29, 255, 255]))),
        "red": ((np.array([0, 135, 0]), np.array([10, 255, 255])),
                (np.array([169, 150, 0]), np.array([180, 255, 255])))
    }


    potential_signs1 = []
    for key, colors in hsv_colors.items():
        if key == "red":
            mask0 = cv2.inRange(hsv, colors[0][0], colors[0][1])
            mask1 = cv2.inRange(hsv, colors[1][0], colors[1][1])
            mask = mask0 + mask1
        else:
            mask = cv2.inRange(hsv, colors[0], colors[1])

        out = cv2.bitwise_and(im, im, mask=mask)
        blur = cv2.blur(out, (5, 5), 0)
        imgray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_OTSU)
        # cv2.imshow("out", out)
        # cv2.imshow("thresh", thresh)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        #
        # cv2.imshow("images", np.hstack([im, out]))
        # cv2.waitKey(0)

        heir, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        copy = im.copy()
        # print(len(contours))
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # print(area)

            if int(area) > 0:
                epsilon = 0.000000000000000001* cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                cv2.drawContours(copy, [approx], -1, (255, 0, 0), 3)

                if int(area) > 150:
                    x, y, w, h = cv2.boundingRect(cnt)
                    x -= 10
                    y -= 10
                    if x < 0: x = 0
                    if y < 0: y = 0
                    # cv2.rectangle(copy, (x, y), (x + w+20, y + h+20), (0, 255, 0), 2)
                    # cv2.imshow("copy", copy)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    # potential_signs.append(np.array([x, y, w, h]))
                    potential_signs1.append(image[y:(y + h + 20), x:(x + w + 20)])
    #
    # for i in potential_signs1:
    #     cv2.imshow("a;lskdfj", i)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #print(len(potential_signs1))
    potential_signs = kMeansSegmentation(potential_signs1)
    #print(len(potential_signs))
    return potential_signs