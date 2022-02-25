import matplotlib.pyplot as plt
import numpy as np
import cv2

sift = cv2.SIFT_create()
matcher = cv2.BFMatcher()

single = cv2.imread("biba2.jpg", 0)
print(single.shape)

key_points1, descriptors1 = sift.detectAndCompute(single, None)

cam = cv2.VideoCapture(1)
cv2.namedWindow("Camera", cv2.WINDOW_KEEPRATIO)

while cam.isOpened():
    
    _, many = cam.read()
    # many = cv2.flip(many, 1)
    # blurred = cv2.GaussianBlur(many, (11, 11), 0)
    image = cv2.cvtColor(many, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Camera", image)

    key_points2, descriptors2 = sift.detectAndCompute(image, None)
    if key_points2:
        matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

        best = []

        for m1, m2 in matches:
            if m1.distance < 0.75 * m2.distance:
                best.append([m1])

        if len(best) > 30:
            src_pts = np.float32([key_points1[m[0].queryIdx].pt for m in best]).reshape(-1, 1, 2)

            dst_pts = np.float32([key_points2[m[0].trainIdx].pt for m in best]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            h, w = single.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            cv2.polylines(image, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        else:
            print(f"Not enough matches - {len(best)}")

        matches_image = cv2.drawMatchesKnn(single, key_points1, image, key_points2, best, None)
        cv2.imshow("Camera", matches_image)


    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()



# plt.imshow(matches_image)
# plt.show()

