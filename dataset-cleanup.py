import cv2
import numpy as np
import os
from os.path import join
"""
    This is a one-time use script used to clean up our data
    set by filtering out images with poor performance.
    If an image has more than 7 bounding boxes, it puts it into
    a directory titled "bad-data", if it has less than 7
    bounding boxes, the image is shown with the contours to the
    user to be manually decided if it should remain as good
    data.  These images will be placed in a directory titled
    "good-data" lolz
"""
home_path = os.getcwd()
images_path = os.path.join(home_path, 'all_eligible_signs')

rejected = False

def brighten(im):
    im = np.float32(im)
    maxIntensity = 255.0
    x = np.arange(maxIntensity)
    phi = 1
    theta = 1
    out = (maxIntensity/phi)*(im/(maxIntensity/theta))*0.85
    out= np.uint8(out)
    return out

index = 0
for i in os.listdir(images_path):
    im = cv2.imread(os.path.join(images_path, i))
    if im is not None:
        image = im.copy()
        index += 1
        #run segmentation code
        image = brighten(image)

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
        clahe1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
        clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(65, 65))
        lab[:, :, 0] = clahe2.apply(lab[:, :, 0])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        # im = global_contrast_normalization(im, 1, 10, 0.0000000001)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        hsv[:,:,1] = cv2.equalizeHist(hsv[:,:,1])
        hsv[:, :, 1] = clahe1.apply(hsv[:,:,1])
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        hsv_colors = {
            "yellow": ((np.array([18, 165, 25]), np.array([29, 255, 255]))),
            "red": ((np.array([0, 135, 0]), np.array([10, 255, 255])),
                    (np.array([169, 125, 0]), np.array([180, 255, 255])))
        }

        potential_signs = []
        for key, colors in hsv_colors.items():


            if key == "red":
                mask0 = cv2.inRange(hsv, colors[0][0], colors[0][1])
                mask1 = cv2.inRange(hsv, colors[1][0], colors[1][1])
                mask = mask0 + mask1
            else:
                mask = cv2.inRange(hsv, colors[0], colors[1])

            out = cv2.bitwise_and(image, image, mask=mask)
            blur = cv2.blur(out, (5, 5), 0)
            imgray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_OTSU)

            heir, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



            for i, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)

                if int(area) > 0:
                    epsilon = 0.000000000000000001 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    # if key == "red":
                    #     cv2.drawContours(copy, [approx], -1, (0, 0, 255), 2)
                    # else:
                    #     cv2.drawContours(copy, [approx], -1, (255,255,0), 2)
                    if 150 < area < 3000:
                        x, y, w, h = cv2.boundingRect(cnt)
                        if key == "red":
                            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        else:
                            cv2.rectangle(image, (x, y), (x + w, y + h), (255,255,0), 2)
                        # potential_signs.append(np.array([x, y, w, h]))
                        potential_signs.append(image[y:(y + h), x:(x + w)])

            if len(potential_signs) > 30 or len(potential_signs) == 0:
                cv2.imwrite(home_path + '\\bad-data\\' + str(index) + ".jpg", im)
                cv2.destroyAllWindows()
                rejected = True
                break

        if rejected == False:
            cv2.imshow("image", image)
            selection = cv2.waitKey(0)

            # selection = input("press 'g' to put this image in good-data or 'b' to"
            #                   " put this image in bad-data")
            cv2.destroyAllWindows()

            if selection == ord('g'):
                cv2.imwrite(home_path + '\\good-data\\' + str(index) + ".jpg", im)
                cv2.destroyAllWindows()
                selection = ""

            elif selection == ord('b'):
                cv2.imwrite(home_path + '\\bad-data\\' + str(index) + ".jpg", im)
                cv2.destroyAllWindows()
                selection = ""

        rejected = False
