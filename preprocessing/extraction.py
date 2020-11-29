import os
from typing import List, Any

import cv2
import numpy as np


class ImageExtractor:
    threshold = 127
    artifacts = "../images/artifacts"

    def __init__(self, image_file, directory="../images/"):
        image_path = os.path.join(directory, image_file)
        self.image = cv2.imread(image_path)
        self.grayscale = None
        self.thresh = None

    def process(self):
        self.grayscale = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        _, self.thresh = cv2.threshold(self.grayscale, ImageExtractor.threshold, 255, cv2.ADAPTIVE_THRESH_MEAN_C)

    def extract_letters(self, height, width, eroding=3, save_artifacts=False) -> List[Any]:
        img_erode = cv2.erode(self.thresh, np.ones((eroding, eroding), np.uint8), iterations=2)
        contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        output = None
        if save_artifacts:
            output = self.image.copy()

        letters = []
        for idx, contour in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(contour)
            if hierarchy[0][idx][3] == 0:
                if save_artifacts:
                    cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 1)

                letter_crop = self.grayscale[y:y + h, x:x + w]
                size_max = max(w, h)
                letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
                if w > h:
                    y_pos = size_max // 2 - h // 2
                    letter_square[y_pos:y_pos + h, 0:w] = letter_crop
                elif w < h:
                    x_pos = size_max // 2 - w // 2
                    letter_square[0:h, x_pos:x_pos + w] = letter_crop
                else:
                    letter_square = letter_crop

                letters.append(((x, y), w, cv2.resize(letter_square, (height, width), interpolation=cv2.INTER_AREA)))

        if save_artifacts:
            try:
                os.mkdir(ImageExtractor.artifacts)
            except FileExistsError:
                pass
            cv2.imwrite(os.path.join(ImageExtractor.artifacts, "last.png"), output)

        letters.sort(key=lambda letter: letter[0])

        return letters


extractor = ImageExtractor("hello_painted.png")
extractor.process()
for i, image in enumerate(extractor.extract_letters(28, 28, save_artifacts=True)):
    cv2.imshow(f"Letter #{i}", image[2])
cv2.waitKey(0)
