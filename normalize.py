import cv2
import numpy as np


class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        # if the shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            shape = "triangle"

        # if the shape has 4 vertices, it is either a square or
        # a rectangle
        elif len(approx) == 4:
            # compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)

            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

        # if the shape is a pentagon, it will have 5 vertices
        elif len(approx) == 5:
            shape = "pentagon"

        # otherwise, we assume the shape is a circle
        else:
            shape = "circle"

        # return the name of the shape
        return shape


def shapeDetection(potential_signs):
    # load the image and resize it to a smaller factor so that
    # the shapes can be approximated better
    for image in potential_signs:
        resized = cv2.resize(image, (300, 300))
        cv2.imshow("resized", image)
        cv2.waitKey(0)
        ratio = image.shape[0] / float(resized.shape[0])

        # convert the resized image to grayscale, blur it slightly,
        # and threshold it
        copy = image.copy()
        gray = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blur, 1, 255)

        cv2.imshow("edges", edges)
        cv2.waitKey(0)
        # edges = cv2.convertScaleAbs(edges)
        # find contours in the thresholded image and initialize the
        # shape detector
        heir, cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        sd = ShapeDetector()

        # loop over the contours
        print("# of cnts: ",len(cnts))
        for c in cnts:
            area = cv2.contourArea(c)
            if area > 5.0:
                # compute the center of the contour, then detect the name of the
                # shape using only the contour
                M = cv2.moments(c)
                if M["m00"] != 0 and M["m01"] != 0 and M["m10"] != 0:
                    cX = int((M["m10"] / M["m00"]) * ratio)
                    cY = int((M["m01"] / M["m00"]) * ratio)
                    shape = sd.detect(c)
                    # multiply the contour (x, y)-coordinates by the resize ratio,
                    # then draw the contours and the name of the shape on the image
                    if shape != "unidentified":
                        c = c.astype("float")
                        c *= ratio
                        c = c.astype("int")
                        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
                        print(shape)
                    # cv2.imshow("hi", image)
                    # cv2.waitKey(0)
                    # show the output image
        cv2.imshow("im", image)
        cv2.waitKey(0)

def normalize(potentials):
    # potentials = shapeDetection(potentials)
    norm_ims = []
    dim = (128,128)

    for sign in potentials:
        resized = cv2.resize(sign, dim)
        norm_ims.append(resized)
    return norm_ims