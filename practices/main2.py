import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve

image = np.zeros((250, 150))
image[40:-40, 40:-40] = 1

kx = -1 * np.array([[-1, 0, 1]])
fx = convolve(image, kx)

ky = -1 * np.array([[-1], [0], [1]])
fy = convolve(image, ky)

magnitude = np.sqrt(fx ** 2 + fy ** 2)
phase = cv2.phase(fx, fy, angleInDegrees=True)
mask = np.zeros(image.shape + (3,), dtype="uint8")

mask[(magnitude != 0) & (phase >= 0) & (phase < 90)] = np.array([255, 0, 0])
mask[(magnitude != 0) & (phase >= 90) & (phase < 180)] = np.array([0, 255, 0])
mask[(magnitude != 0) & (phase >= 180) & (phase < 270)] = np.array([0, 0, 255])
mask[(magnitude != 0) & (phase >= 270) & (phase < 360)] = np.array([255, 255, 0])

plt.subplot(121)
plt.imshow(image)
plt.subplot(122)
plt.imshow(mask)
plt.show()

