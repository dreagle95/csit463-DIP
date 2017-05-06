import cv2
import os
import numpy as np

home_path = os.getcwd()
images_path = os.path.join(home_path, 'good-data')

def brighten(im):
    im = np.float32(im)
    maxIntensity = 255.0
    x = np.arange(maxIntensity)
    phi = 1
    theta = 1
    out = (maxIntensity/phi)*(im/(maxIntensity/theta))*0.75
    out= np.uint8(out)
    return out

images = []
for i in os.listdir(images_path):
    image = cv2.imread(os.path.join(images_path, i))
    if image is not None:
        images.append(image)

potential_signs = []
for image in images:
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

    hsv[:, :, 1] = clahe1.apply(hsv[:, :, 1])
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    hsv_colors = {
        "yellow": ((np.array([18, 175, 50]), np.array([29, 255, 255]))),
        "red": ((np.array([0, 140, 0]), np.array([10, 255, 255])),
                (np.array([169, 150, 0]), np.array([180, 255, 255])))
    }

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

        copy = image.copy()

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            # print(area)

            if int(area) > 0:
                epsilon = 0.000000000000000001 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                cv2.drawContours(copy, [approx], -1, (255, 0, 0), 3)

                if 150 < area < 3000:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # potential_signs.append(np.array([x, y, w, h]))
                    potential_signs.append(copy[y:(y + h + 20), x:(x + w + 20)])
index = 0
for index, sign in enumerate(potential_signs):
    index+=1
    cv2.imshow("Potential sign", sign)
    classification = cv2.waitKey(0)

    if classification == ord('s'):
        cv2.imwrite(home_path + '\\stopClassification\\' + str(classification) + str(index)
                    +'.jpg', sign)
        cv2.destroyAllWindows()
        classification = ""

    elif classification == ord('w'):
        cv2.imwrite(home_path + '\\warnClassification\\' + str(classification) + str(index)
                    +'.jpg', sign)
        cv2.destroyAllWindows()
        classification = ""

    if(classification == ord('f')):
        print("false-positive")
        cv2.destroyAllWindows()
        classification = ""