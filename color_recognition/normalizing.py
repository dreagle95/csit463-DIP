import cv2
import os

dim = (16, 16)
stop_Images = []
for i in os.listdir("stop"):
    img = cv2.imread(os.path.join("stop", i))
    if img is not None:
        resized = (cv2.cvtColor(cv2.resize(img, dim), cv2.COLOR_BGR2GRAY))
        stop_Images.append(resized)


warn_Images = []
for i in os.listdir("warning"):
    img = cv2.imread(os.path.join("warning", i))
    if img is not None:
        resized = (cv2.cvtColor(cv2.resize(img, dim),cv2.COLOR_BGR2GRAY))
        warn_Images.append(img)

print(len(stop_Images))
print(len(warn_Images))

cv2.imshow("HI", stop_Images[0])
print(stop_Images[0].shape)
cv2.waitKey(0)