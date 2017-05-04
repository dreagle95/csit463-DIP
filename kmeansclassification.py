import cv2
import os
import numpy as np
from random import randint

home_path = os.getcwd()
images_path = os.path.join(home_path, 'warnClassification')

images = []
signs = []
for i in os.listdir(images_path):
    image = cv2.imread(os.path.join(images_path, i))
    if image is not None:
        images.append(image)

for im in images:
    copy = im.copy()
    #reshape to all pixels by 3
    Z = copy.reshape((-1,3))

    #convert to np.float32
    Z = np.float32(Z)

    #define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 5

    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    #Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((copy.shape))
    hsv = cv2.cvtColor(copy, cv2.COLOR_BGR2HSV)
    # cv2.imshow("res2", res2)
    # cv2.waitKey(0)

    yellow_boundary = [([20, 90, 90], [30, 255, 255])]
    # red_boundary = {
    #     "red": ((np.array([0,140,0]), np.array([8,255,255])),
    #             (np.array([145,150,0]), np.array([180,255,255])))
    # }
    #
    # for key, colors in red_boundary.items():
    #     if key == "red":
    #         mask0 = cv2.inRange(hsv, colors[0][0], colors[0][1])
    #         mask1 = cv2.inRange(hsv, colors[1][0], colors[1][1])
    #         mask = mask0 + mask1

    for (lower, upper) in yellow_boundary:
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")
        mask = cv2.inRange(hsv, lower, upper)
        output = cv2.bitwise_and(res2, res2, mask = mask)
        blur = cv2.blur(output, (5,5), 0)
        imgray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_OTSU)

        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # below line is used to connect lines in thresh
        # cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21)), iterations=10)
        # calc_Contours = []
        # for i, cnt in enumerate(contours):
        #     M = cv2.moments(cnt)
        #     cx = int(M['m10']/M['m00'])
        #     cy = int(M['m01']/M['m00'])
        #     calc_Contours[i] = (cx,cy)

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if int(area) > 0: #and hierarchy[0,i,3] == -1 <-- possible and condition
                epsilon = 0.0000000000000000000000001 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                #print (approx)
                cv2.drawContours(res2, [approx], -1, (255, 0, 0), 2)

            if area > 150:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
                signs.append(im[y:(y+h), x:(x+w)])
                cv2.imshow("copy", copy)
                cv2.imshow("thresh", thresh)
                cv2.imshow("res2", res2)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                #print(len(signs))
        #print(len(contours))

# index = 0
# for (index, sign) in enumerate(signs):
#     index+=1
#     cv2.imshow("Sign", sign)
#     classification = cv2.waitKey(0)
#
#     if classification == ord('g'):
#         cv2.imwrite(home_path + '\\stopClassification2\\' + str(classification)
#                     + str(index) +'.jpg', sign)
#         cv2.destroyAllWindows()
#         classification = ""
#     else:
#         cv2.destroyAllWindows()