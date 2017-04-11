import numpy as np
import cv2


image = cv2.imread("example.png")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
hsv[:, 0, :] = cv2.equalizeHist(hsv[:, 0, :])

# dark_yellow = np.uint8([[[5, 65, 75]]])
# hsv_dyellow = cv2.cvtColor(dark_yellow, cv2.COLOR_BGR2HSV)
# light_yellow = np.uint8([[[68, 190, 240]]])
# hsv_lyellow = cv2.cvtColor(light_yellow, cv2.COLOR_BGR2HSV)
# dark_red = np.uint8([[[0, 0, 29]]])
# light_red = np.uint8([[[60, 56, 255]]])
# hsv_dred = cv2.cvtColor(dark_red, cv2.COLOR_BGR2HSV)
# hsv_lred = cv2.cvtColor(light_red, cv2.COLOR_BGR2HSV)
# print(hsv_dred, hsv_lred)

hsv_color_pairs = (
    (np.array([16, 100, 1]), np.array([26, 255, 255])),
    (np.array([0, 75, 0]), np.array([9, 255, 255]))
)

potential_signs = []

for colors in hsv_color_pairs:
    mask = cv2.inRange(hsv, colors[0], colors[1])
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

    # cv2.imshow("image", image)
    # cv2.waitKey(0)

    copy = image.copy()
    # print(len(contours))

    signs = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        #print(area)

        if int(area) > 0:
            epsilon = 0.00000001 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            cv2.drawContours(image, [approx], -1, (255, 0, 0), 3)

            if area > 175 and area < 2200:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
                #potential_signs.append(np.array([x, y, w, h]))
                potential_signs.append(copy[y:(y+h+20), x:(x+w+20)])
                #print("x "+str(x+w))
                #print("w "+str(w))
                #print("y " +str(y+h))
                #print(copy.shape)

    cv2.imshow("potential_signs", copy)
    cv2.waitKey(0)
    #cv2.imshow("Signs", potential_signs[0])
    for i in range(0,len(potential_signs)):
        cv2.imshow("potential_signs", potential_signs[i])
        cv2.waitKey(0)
        print(potential_signs[i].shape)
        print(len(potential_signs))
    cv2.waitKey(0)

#print(len(potential_signs))
# regions of interest? Can we do that with photos or is that only for videos?
# now we need to get these boxes into an image to pass to the CNN
