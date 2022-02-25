from ctypes import pointer
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn import neighbors

template = cv2.imread("reeses_puffs.png", 0)
many = cv2.imread("many_cereals.jpg", 0)

# orb = cv2.ORB_create()
# key_points1, descriptors1 = orb.detectAndCompute(template, None)
# key_points2, descriptors2 = orb.detectAndCompute(many, None)
# matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# matches = matcher.match(descriptors1, descriptors2)
# print(matches[0].distance)
# matches = sorted(matches, key=lambda match: match.distance)
# image = cv2.drawMatches(template, key_points1, many, key_points2, matches, None)

n = 20

xk1 = 100 + np.random.randint(-25, 25, n)
yk1 = 100 + np.random.randint(-25, 25, n)
rk1 = np.repeat(1, n)

xk2 = 150 + np.random.randint(-25, 25, n)
yk2 = 150 + np.random.randint(-25, 25, n)
rk2 = np.repeat(2, n)

point = (127, 124)

knn = cv2.ml.KNearest_create()
train = np.stack([
    np.hstack([xk1, xk2]),
    np.hstack([yk1, yk2])
]).T.astype("f4")

responses = np.hstack([rk1, rk2]).reshape(-1, 1).astype("f4")

knn.train(train, cv2.ml.ROW_SAMPLE, responses)
ret, results, neighbours, dist = knn.findNearest(np.array(point).astype("f4").reshape(1, 2), 3)
print(ret, results, neighbours, dist)

plt.scatter(xk1, yk1, 80, 'r', '^')
plt.scatter(xk2, yk2, 80, 'b', 's')
plt.scatter([point[0]], [point[1]], 80, 'g', 'x')
plt.show()