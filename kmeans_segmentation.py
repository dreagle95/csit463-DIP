import cv2
import numpy as np

def kMeansSegmentation(images):
    return_signs = []
    for im in images:
        copy = im.copy()

        #reshape to all pixels by 3
        Z = copy.reshape((-1,3))

        #convert to np.float32
        Z = np.float32(Z)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        # if key == "red":
        #     K = 5
        #     ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        #     # Now convert back into uint8, and make original image
        #     center = np.uint8(center)
        #     res = center[label.flatten()]
        #     res2 = res.reshape((copy.shape))
        #     hsv = cv2.cvtColor(copy, cv2.COLOR_BGR2HSV)
        #
        #     # lower red mask(0-8)
        #     lower_red = np.array([0, 140, 0])
        #     upper_red = np.array([10, 255, 255])
        #     mask0 = cv2.inRange(hsv, lower_red, upper_red)
        #
        #     # upper red mask (145-180)
        #     lower_red = np.array([145, 150, 0])
        #     upper_red = np.array([180, 255, 255])
        #     mask1 = cv2.inRange(hsv, lower_red, upper_red)
        #
        #     mask = mask0 + mask1
        #
        # elif key == "yellow":
        #     K = 4
        #     ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        #     # Now convert back into uint8, and make original image
        #     center = np.uint8(center)
        #     res = center[label.flatten()]
        #     res2 = res.reshape((copy.shape))
        #     hsv = cv2.cvtColor(copy, cv2.COLOR_BGR2HSV)
        #     #yellow boundaries
        #     lower_yellow = (np.array([20, 90, 90]))
        #     upper_yellow = (np.array([30, 255, 255]))
        #     mask = cv2.inRange(hsv, lower_yellow , upper_yellow )

        K = 5
        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        #Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((copy.shape))
        hsv = cv2.cvtColor(copy, cv2.COLOR_BGR2HSV)

        hsv_colors = {
            "yellow": ((np.array([20, 90, 90]), np.array([30, 255, 255]))),
            "red": ((np.array([0, 140, 0]), np.array([8, 255, 255])),
                    (np.array([145, 150, 0]), np.array([180, 255, 255])))
        }

        for key, colors in hsv_colors.items():
            if key == "red":
                mask0 = cv2.inRange(hsv, colors[0][0], colors[0][1])
                mask1 = cv2.inRange(hsv, colors[1][0], colors[1][1])
                mask = mask0 + mask1
            else:
                mask = cv2.inRange(hsv, colors[0], colors[1])

            output = cv2.bitwise_and(res2, res2, mask=mask)
            blur = cv2.blur(output, (21, 21), 0)
            imgray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_OTSU)
            # cv2.imshow("output", output)
            # cv2.imshow("thresh", thresh)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                sign = None
                if int(area) > 0:
                    epsilon = 0.0000000000000000000000001 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    #print (approx)
                    #cv2.drawContours(res2, [approx], -1, (255, 0, 0), 2)

                    if area > 150:
                        x, y, w, h = cv2.boundingRect(cnt)
                        cv2.rectangle(copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        sign = im[y:(y+h), x:(x+w)]
                        # cv2.imshow("sldkfj", copy)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()

                        if sign is not None:
                            return_signs.append(sign)

    # for i in return_signs:
    #     cv2.imshow("asl;dkfj", i)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    return return_signs