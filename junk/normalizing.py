import cv2
import os
import numpy as np

dim = (16, 16)
stopTemplate = np.zeros((16,16), dtype="uint8")
warnTemplate = np.zeros((16,16), dtype="uint8")

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
        warn_Images.append(resized)


for i in stop_Images:
    stopTemplate = stopTemplate + i
    #print(stopTemplate)
    #np.add(stopTemplate, i)
stopTemplate //= len(stop_Images)

for i in range(0, len(stopTemplate)):
    for j in range(0, len(stopTemplate)):
        if stopTemplate[i][j] > 255:
            stopTemplate[i][j] = 255

for i in warn_Images:
    np.add(warnTemplate, i)
warnTemplate //= len(warn_Images)

for i in range(0, len(warnTemplate)):
    for j in range(0, len(warnTemplate)):
        if warnTemplate[i][j] > 255:
            warnTemplate[i][j] = 255

print(len(stop_Images))
print(len(warn_Images))

print(stopTemplate)
cv2.imshow("HI", stopTemplate)
#print(stop_Images[2].shape)
cv2.waitKey(0)