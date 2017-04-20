import cv2
import matplotlib as mpl
import numpy as np
import scipy as pi

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


def normalize(potentials):
    norm_ims = []
    dim = (16, 16)

    for sign in potentials:
        print(sign.shape)
        resized = cv2.resize(sign, dim)



        norm_ims.append(resized)
    return norm_ims

def segment_sign(im):
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

    # dark_yellow = np.uint8([[[5, 65, 75]]])
    # hsv_dyellow = cv2.cvtColor(dark_yellow, cv2.COLOR_BGR2HSV)
    # light_yellow = np.uint8([[[68, 190, 240]]])
    # hsv_lyellow = cv2.cvtColor(light_yellow, cv2.COLOR_BGR2HSV)
    # dark_red = np.uint8([[[0, 0, 29]]])
    # light_red = np.uint8([[[60, 56, 255]]])
    # hsv_dred = cv2.cvtColor(dark_red, cv2.COLOR_BGR2HSV)
    # hsv_lred = cv2.cvtColor(light_red, cv2.COLOR_BGR2HSV)
    # print(hsv_dred, hsv_lred, hsv_lyellow, hsv_dyellow)

    # hsv[:,:,1] = cv2.equalizeHist(hsv[:,:,1])
    hsv[:,:,1] = clahe1.apply(hsv[:,:,1])
    # hsv[:,:,2] = clahe1.apply(hsv[:,:,2])
    # hsv = cv2.fastNlMeansDenoisingColored(hsv, None, 10, 10, 7, 7)
    im = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # cv2.imshow("im", im)
    # cv2.waitKey(0)
    # return

    # hsv_color_pairs = (
    #     # (np.array([18, 200, 50]), np.array([27, 255, 255])),
    #     (np.array([0, 100, 0]), np.array([9, 255, 255])),
    # )

    potential_signs = []

    hsv_colors = {
        "yellow": ((np.array([18, 175, 50]), np.array([29, 255, 255]))),
        "red": ((np.array([0, 140, 0]), np.array([10, 255, 255])),
                (np.array([169, 150, 0]), np.array([180, 255, 255])))
    }

    for key, colors in hsv_colors.items():
        potential_signs = []

        if key == "red":
            mask0 = cv2.inRange(hsv, colors[0][0], colors[0][1])
            mask1 = cv2.inRange(hsv, colors[1][0], colors[1][1])
            # cv2.imshow("mask0", mask0)
            # cv2.imshow("mask1", mask1)
            # cv2.waitKey(0)
            mask = mask0 + mask1
        else:
            mask = cv2.inRange(hsv, colors[0], colors[1])

        out = cv2.bitwise_and(im, im, mask=mask)
        blur = cv2.blur(out, (5, 5), 0)
        imgray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_OTSU)

        cv2.imshow("images", np.hstack([im, out]))
        # cv2.imshow("thresh", thresh)
        # cv2.imshow("out", out)
        # cv2.imshow("blur", blur)
        cv2.waitKey(0)

        heir, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # cv2.imshow("image", image)
        # cv2.waitKey(0)

        copy = im.copy()
        # print(len(contours))

        signs = []
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            # print(area)

            if int(area) > 0:
                epsilon = 0.000000000000000001* cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                cv2.drawContours(copy, [approx], -1, (255, 0, 0), 3)

                if 150 < area < 3000:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # potential_signs.append(np.array([x, y, w, h]))
                    potential_signs.append(copy[y:(y + h + 20), x:(x + w + 20)])

        cv2.imshow("potential_signs", copy)
        cv2.waitKey(0)
        # # cv2.imshow("Signs", potential_signs[0])
        # for i in range(0,len(potential_signs)):
        #     cv2.imshow("potential_signs", potential_signs[i])
        #     cv2.waitKey(0)
        #     print(potential_signs[i].shape)
        #     print(len(potential_signs))
        # cv2.waitKey(0)

    # print(len(potential_signs))
    # regions of interest? Can we do that with photos or is that only for videos?
    # now we need to get these boxes into an image to pass to the CNN

    # potential_signs = normalize(potential_signs)
    return potential_signs
