import matplotlib.pyplot as plt
import numpy as np
import cv2

cam = cv2.VideoCapture(1)
cv2.namedWindow("Camera", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("Water", cv2.WINDOW_KEEPRATIO)

lower = (5, 120, 200)
upper = (50, 255, 255)


while cam.isOpened():
    _, image = cam.read()
    image = cv2.flip(image, 1)
    blurred = cv2.GaussianBlur(image, (27, 27), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # blurred = cv2.medianBlur(image, 25)
    # gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(mask, 160, 255, cv2.THRESH_BINARY_INV)

    # counters, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # for i in range(len(counters)):
    #     if hierarchy[0][i][3] == -1:
    #         cv2.drawContours(image, counters, i, (255, 0,0 ), 2)

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    ret, fg = cv2.threshold(dist, 0.7 * dist.max(), 255, 0)
    fg = np.uint8(fg)
    confuse = cv2.subtract(mask, fg)

    ret, markers =cv2.connectedComponents(fg)
    markers += 1

    markers[confuse == 255] = 0

    wmarkers = cv2.watershed(image, markers.copy())
    # print(wmarkers)
    counters, hierarchy = cv2.findContours(wmarkers.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 200, 200)
    ]

    for i in range(len(counters)):
        if hierarchy[0][i][3] == -1:
            cv2.drawContours(image, counters, i, colors[i % 7], 2)

    print((len(counters) - 1) // 2)
    # print(hierarchy)

    cv2.imshow("Camera", image)
    cv2.imshow("Water", mask)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()