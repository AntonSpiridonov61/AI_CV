import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import face

def rotate_image(image, angle):
    center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_m = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(image, rot_m, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

image = face()

eye = image[230:370, 490:650, :]

methods = ["TM_CCOEFF", "TM_CCOEFF_NORMED",
            "TM_CCORR_NORMED",
            "TM_SQDIFF", "TM_SQDIFF_NORMED"]

for method in methods:
    reference = image.copy()

    # reference = cv2.GaussianBlur(reference, (91, 91), 0)
    # reference = rotate_image(reference, 40)
    # reference = np.flipud(reference).copy()
    # reference = cv2.resize(reference, None, fx=3, fy=3)

    result = cv2.matchTemplate(reference, eye, getattr(cv2, method))
    min_value, max_value, min_loc, max_loc = cv2.minMaxLoc(result)
    if "TM_SQDIFF" in method:
        loc = min_loc
    else:
        loc = max_loc

    loc_right = (loc[0] + eye.shape[1], loc[1] + eye.shape[0])

    plt.title(f"{min_value}-{max_value}, {min_loc}-{max_loc}")
    cv2.rectangle(reference, loc, loc_right, (255, 0, 255), 10)
    plt.figure(method)
    plt.subplot(121)
    plt.imshow(result)
    plt.subplot(122)
    plt.imshow(reference)

plt.show()