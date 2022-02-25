import matplotlib.pyplot as plt
import numpy as np
import cv2

image = cv2.cvtColor(cv2.imread("real_chessboard.jpg"), cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# dst = cv2.cornerHarris(gray, 2, 3, 0.04)
# dst = cv2.dilate(dst, None)
# image[dst > 0.01 * dst.max()] = [255, 0, 0]

corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 20)
cornres = np.int0(corners)

for corner in corners:
    x, y = corner.ravel()
    cv2.circle(image, (x, y), 3, (255, 0, 0))

# plt.subplot(121)
plt.imshow(image)
# plt.subplot(122)
# plt.imshow(dst)
plt.show()