import numpy as np
import cv2

def find_if_close(cnt1,cnt2):
    row1,row2 = cnt1.shape[0],cnt2.shape[0]
    areas = (cv2.contourArea(cnt1), cv2.contourArea(cnt2))
    if sum(areas) != 0:
        for i in range(row1):
            for j in range(row2):
                dist = np.linalg.norm(cnt1[i]-cnt2[j])
                if abs(dist) < 5 :
                    return True
                elif i==row1-1 and j==row2-1:
                    return False
    else:
        return False

image = cv2.imread("example3.png")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
hsv[:,0,:] = cv2.equalizeHist(hsv[:,0,:])

dark_yellow = np.uint8([[[5,65,75]]])
hsv_dyellow = cv2.cvtColor(dark_yellow, cv2.COLOR_BGR2HSV)
light_yellow = np.uint8([[[68,190,240]]])
hsv_lyellow = cv2.cvtColor(light_yellow, cv2.COLOR_BGR2HSV)
dark_red = np.uint8([[[0,0,29]]])
light_red = np.uint8([[[60,56,255]]])
hsv_dred = cv2.cvtColor(dark_red, cv2.COLOR_BGR2HSV)
hsv_lred = cv2.cvtColor(light_red, cv2.COLOR_BGR2HSV)
print(hsv_dred, hsv_lred)

mask = np.zeros((len(image), len(image[0])), dtype=np.uint8)
temp_mask = cv2.inRange(hsv, np.array([16, 100, 1]), np.array([26, 255, 255]))
mask += temp_mask
temp_mask = cv2.inRange(hsv, np.array([0, 75, 0]), np.array([9, 255, 255]))
mask += temp_mask


out = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("out", out)
blur = cv2.blur(out, (5,5),0)
cv2.imshow("blur", blur)
imgray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_OTSU)
cv2.imshow("images", np.hstack([image, out]))
cv2.imshow("thresh", thresh)
cv2.waitKey(0)

heir, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# cv2.imshow("image", image)
# cv2.waitKey(0)

copy = image.copy()
print(len(contours))

for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    print(area)

    if int(area) > 0:
        epsilon = 0.00000001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.drawContours(copy, [approx], -1, (255, 0, 0), 3)

        if area > 175:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(copy,(x,y),(x+w, y+h),(0,255,0),2)

cv2.imshow("turd", copy)
cv2.waitKey(0)

# status = np.zeros((len(contours), 1))
#
# for i,cnt1 in enumerate(contours):
#     x = i
#     if i != len(contours)-1:
#         for j,cnt2 in enumerate(contours[i+1:]):
#             areas = (cv2.contourArea(cnt1), cv2.contourArea(cnt2))
#             perimeters = (cv2.arcLength(cnt1, True), cv2.arcLength(cnt2, True))
#             x = x+1
#
#             dist = find_if_close(cnt1,cnt2)
#             if dist == True:
#                 val = min(status[i],status[x])
#                 status[x] = status[i] = val
#             else:
#                 if status[x]==status[i]:
#                     status[x] = i+1
#
# unified = []
# maximum = int(status.max())+1
# for i in range(maximum):
#     pos = np.where(status==i)[0]
#     if pos.size != 0:
#         cont = np.vstack(contours[i] for i in pos)
#         if cv2.contourArea(cont) > 3 and cv2.arcLength(cont, True) > 50:
#             print(cv2.contourArea(cont), cv2.arcLength(cont, True))
#             hull = cv2.convexHull(cont)
#             unified.append(hull)
#             x, y, w, h = cv2.boundingRect(hull)
#             cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
# cv2.drawContours(image,unified,-1,(255,0,0),2)
# cv2.imshow("image", image)
# cv2.waitKey(0)