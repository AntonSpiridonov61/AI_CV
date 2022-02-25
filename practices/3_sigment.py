import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib import cm

def make_colors(n):
    colors = []
    for i in range(n):
        colors.append(tuple(np.array(cm.tab10(i))[:3] * 255))
    return colors

road = cv2.imread('road_image.jpg')

road_copy = road.copy()
markers = np.zeros(road.shape[:-1], dtype="int32")
segments = np.zeros_like(road)

cv2.namedWindow("road_image", cv2.WINDOW_NORMAL)
cv2.namedWindow("segments", cv2.WINDOW_NORMAL)

current_marker = 1
updated = False
colors = make_colors(10)

print(colors[0])


def mouse_callback(event, x, y, flags, param):
    global updated
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(markers, (x, y), 10, (current_marker), -1)
        cv2.circle(road_copy, (x, y), 10, colors[current_marker], -1)
        updated = True

cv2.setMouseCallback("road_image", mouse_callback)

while True:
    cv2.imshow("road_image", road_copy)
    cv2.imshow("segments", segments)
    k = cv2.waitKey(1)
    if k == 27:
        break
    elif k == ord('c'):
        road_copy = road.copy()
        markers = np.zeros(road.shape[:-1], dtype="int32")
        segments = np.zeros_like(road)
    elif k > 0 and chr(k).isdigit():
        current_marker = int(chr(k))

    if updated:
        markers_copy = markers.copy()
        cv2.watershed(road, markers_copy)
        segments = np.zeros(road.shape, dtype=np.uint8)
        for color_ind in range(10):
            segments[markers_copy == (color_ind)] = colors[color_ind]

        updated = False

cv2.destroyAllWindows()