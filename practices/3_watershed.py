import matplotlib.pyplot as plt
import numpy as np
import cv2

pennies = cv2.imread('pennies.jpg')
blurred = cv2.medianBlur(pennies, 25)
gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)

counters, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(counters)):
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(pennies, counters, i, (255, 0,0 ), 10)

dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
ret, fg = cv2.threshold(dist, 0.7 * dist.max(), 255, 0)
fg = np.uint8(fg)
confuse = cv2.subtract(thresh, fg)

ret, markers =cv2.connectedComponents(fg)
markers += 1

markers[confuse == 255] = 0

wmarkers = cv2.watershed(pennies, markers.copy())
counters, hierarchy = cv2.findContours(wmarkers.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(counters)):
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(pennies, counters, i, (255, 0,0 ), 10)

plt.subplot(121)
plt.imshow(pennies)
plt.subplot(122)
plt.imshow(wmarkers)
plt.show()