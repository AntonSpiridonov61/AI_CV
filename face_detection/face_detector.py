import cv2
import matplotlib.pyplot as plt
import numpy as np


def detector(img, classifier, classifier2, scaleFactor=None, minNeighbors=None):
    glasses = cv2.imread("face_detection/src/dealwithit.png")[350:-350, 200:-200]

    result = img.copy()
    rects = classifier.detectMultiScale(
        result, 
        scaleFactor=scaleFactor, 
        minNeighbors=minNeighbors
    )

    for (x, y, w, h) in rects:
        faces = result[y:y + h, x:x + w]
        # cv2.rectangle(result, (x, y), (x+w, y+h), (255, 255, 255))


        rects_eye = classifier2.detectMultiScale(
            faces, 
            scaleFactor=scaleFactor, minNeighbors=minNeighbors,
            minSize=(5, 5)
        )

        if len(rects_eye) == 2:
            x -= 10

            rects_eye = sorted(rects_eye, key=lambda x: x[0])

            x_eyes = x + rects_eye[0][0]
            y_eyes = y + rects_eye[0][1]

            width = rects_eye[1][0] + rects_eye[1][2]
            height = rects_eye[0][3]
            
            roi = result[y_eyes:y_eyes + height, x_eyes:x_eyes + width]

            glasses = cv2.resize(glasses, (width, height))
            glasses_gray = cv2.cvtColor(glasses, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(glasses_gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            result_bg = cv2.bitwise_and(roi, roi, mask=mask)
            result_fg = cv2.bitwise_and(glasses, glasses, mask=mask_inv)

            dst = cv2.add(result_bg, result_fg)
            result[y_eyes:y_eyes + height, x_eyes:x_eyes + width] = dst

    return result

# conf = cv2.imread("face_detection/src/solvay_conference.jpg")
cooper = cv2.imread("face_detection/src/cooper.jpg")

face_cascade = "face_detection/src/haarcascades/haarcascade_frontalface_default.xml"
eye_cascade = "face_detection/src/haarcascades/haarcascade_eye.xml"

face_classifier = cv2.CascadeClassifier(face_cascade)
eye_classifier = cv2.CascadeClassifier(eye_cascade)

# result = detector(cooper, face_classifier, eye_classifier, 1.2, 17)

# plt.imshow(result)
# plt.show()

cam = cv2.VideoCapture(1)
cv2.namedWindow("Camera", cv2.WINDOW_KEEPRATIO)

while cam.isOpened():
    _, image = cam.read()
    image = cv2.flip(image, 1)

    result = detector(image, face_classifier, eye_classifier, 1.1, 12)

    cv2.imshow("Camera", result)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()