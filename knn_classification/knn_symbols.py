import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
from skimage.measure import label, regionprops


def image_to_text(image, knn):
    text = ""
    letters = split_to_letters(image)

    dist = []
    for i in range(len(letters) - 1):
        dist.append(letters[i + 1][0] - letters[i][1])

    mean_dist = np.std(dist) * 0.5 + np.mean(dist)

    for i in range(len(letters)):
        # plt.imshow(letters[i][2])
        # plt.show()
        symbol = extract_features(letters[i][2])
        symbol = np.array(symbol, dtype="f4").reshape(1, 7)
        ret, _, _, _ = knn.findNearest(symbol, 3)
        text += chr(int(ret))

        if i < len(dist) and dist[i] > mean_dist:
            text += ' '

    return text


def split_to_letters(image):
    letters = []
    index_to_delete = []

    image[image > 0] = 1
    labeled = label(image)
    regions = regionprops(labeled)

    for region in regions:
        reg_bbox1 = region.bbox
        reg_center1 = region.centroid[1]

        for index, region2 in enumerate(regions):
            flag_found = False
            reg_bbox2 = region2.bbox
            reg_center2 = region2.centroid[1]

            if reg_bbox1[0] > reg_bbox2[2] and abs(reg_center1 - reg_center2) < 10:
                min_x = min(reg_bbox1[1], reg_bbox2[1])
                max_x = max(reg_bbox1[3], reg_bbox2[3])
                min_y = min(reg_bbox1[0], reg_bbox2[0])
                max_y = max(reg_bbox1[2], reg_bbox2[2])

                letters.append((min_x, max_x, image[min_y:min_x, max_y:max_x]))

                index_to_delete.append(index)
                flag_found = True
                break

        if not flag_found:
            letters.append((reg_bbox1[1], reg_bbox1[3], image[region.slice]))
            
    for index in index_to_delete:
        letters[index] = (None, None, None)

    letters = list(filter(lambda x: x[0] is not None, letters))
    letters.sort(key=lambda x: x[0], reverse=False)

    return letters


def extract_features(img):
    features = []

    _, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    ext_cnt = 0
    int_cnt = 0
    for i in range(len(hierarchy[0])):
        if hierarchy[0][i][-1] == -1:
            ext_cnt += 1
        elif hierarchy[0][i][-1] == 0:
            int_cnt += 1
    features.extend([ext_cnt, int_cnt])

    labeled = label(img)
    region = regionprops(labeled)[0]
    features.append(region.extent)

    centroid = np.array(region.local_centroid) / np.array(region.image.shape)
    features.extend(centroid)

    features.append(region.eccentricity)
    features.append(region.orientation)

    return features


train_dir = Path("knn_classification/out") / "train"
train_data = defaultdict(list)

for path in sorted(train_dir.glob("*")):
    if path.is_dir():
        for img_path in path.glob("*.png"):
            symbol = path.name[-1]
            image = cv2.imread(str(img_path), 0)
            binary = image.copy()
            binary[binary > 0] = 1
            train_data[symbol].append(binary)

features_array = []
responses = []

for i, symbol in enumerate(train_data):
    for img in train_data[symbol]:
        features = extract_features(img)
        features_array.append(features)
        responses.append(ord(symbol))

features_array = np.array(features_array, dtype="f4")
responses = np.array(responses, dtype="f4")

knn = cv2.ml.KNearest_create()

knn.train(features_array, cv2.ml.ROW_SAMPLE, responses)
# test_symbol = extract_features(train_data["o"][0])
# test_symbol = np.array(test_symbol, dtype="f4").reshape(1, 6)
# ret, results, neighbours, dist = knn.findNearest(test_symbol, 3)
# print(chr(int(ret)), results, neighbours, dist)

test_data = Path("knn_classification/out")
for path in sorted(test_data.glob("*.png")):
    image = cv2.imread(str(path), 0)
    print(image_to_text(image, knn))
