import numpy as np
import cv2

image = cv2.imread("example2.png")

#list of boundaries
lower_warning = np.array([51,93.3,29.4])
upper_warning = np.array([43,71.7,94.3])


boundaries = [
    #Yellow boundaries for warning signs
    ([5,65,75], [68,190,240]),
   # ([0,0,29], [60,56,255])
]

#looping over the boundaries
for(lower, upper) in boundaries:
    lower = np.array(lower, dtype = "uint8")
    upper = np.array(upper, dtype = "uint8")

    #find colors within the specified boundaries
    #and apply the mask
    for i in range(1,3):
        mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask = mask)
    imgray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("imgray", imgray)
    ret, thresh = cv2.threshold(imgray, 70, 255, 0)

    #show the images
    cv2.imshow("images", np.hstack([image, output]))
   # cv2.imshow("thresh", thresh)
    cv2.waitKey(0)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print (len(contours))

    cv2.drawContours(image, contours, -1, (255,0,0), 3)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(image,(x,y),(x+w, y+h),(0,255,0),2)
    cv2.imshow("turd", image)
    cv2.waitKey(0)