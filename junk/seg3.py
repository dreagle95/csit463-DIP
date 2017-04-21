import numpy as np
import cv2

image = cv2.imread("example.png")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
hsv[:,:,1] = cv2.equalizeHist(hsv[:,:,1])

h, s, v = cv2.split(hsv)

mask = cv2.inRange(h, 20, 38)
out = cv2.bitwise_and(image, image, mask=mask)
blur = cv2.blur(out, (5, 5), 0)
imgray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_OTSU)

cv2.imshow("images", np.hstack([image, out]))
cv2.imshow("thresh", thresh)
cv2.imshow("out", out)
cv2.imshow("blur", blur)
cv2.waitKey(0)

heir, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    area = cv2.contourArea(cnt)
    # print(area)

    if int(area) > 0:
        epsilon = 0.0000000001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.drawContours(image, [approx], -1, (255, 0, 0), 3)

        if area > 175 and area < 2200:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # potential_signs.append(np.array([x, y, w, h]))
            # potential_signs.append(copy[y:(y + h + 20), x:(x + w + 20)])
cv2.imshow("potential_signs", image)
cv2.waitKey(0)