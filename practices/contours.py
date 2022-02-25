import matplotlib.pyplot as plt
import numpy as np
import cv2

image = cv2.imread("internal_external.png", cv2.IMREAD_GRAYSCALE)
out = np.zeros_like(image)

contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for i, contour in enumerate(contours):
    print(hierarchy[0][i])
    if hierarchy[0][i][3] == 0:
        cv2.drawContours(out, contours, i, 255, -1)

plt.subplot(121)
plt.imshow(out)
plt.subplot(122)
plt.imshow(image)
plt.show()