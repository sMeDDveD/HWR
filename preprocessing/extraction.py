import os
from typing import List, Any

import cv2
import numpy as np

from preprocessing.letter import ExtractedLetter


class ImageExtractor:
    threshold = 127

    def __init__(self, image_file, directory="images"):
        image_path = os.path.join(directory, image_file)
        self.image = cv2.imread(image_path)
        self.grayscale = None
        self.thresh = None
        self.directory = directory

    def process(self):
        self.grayscale = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.grayscale = cv2.GaussianBlur(self.grayscale, (5, 5), 0)

        _, self.thresh = cv2.threshold(self.grayscale, ImageExtractor.threshold, 255,
                                       cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    def extract_letters(self, height, width, eroding=4, save_artifacts=False) -> List[Any]:
        opening = cv2.morphologyEx(self.thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
        img_erode = cv2.erode(opening, np.ones((eroding, eroding), np.uint8), iterations=3)

        contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        output = None
        if save_artifacts:
            output = self.image.copy()

        letters = []
        for idx, contour in enumerate(contours):
            # checking for parent contour
            if hierarchy[0][idx][3] == 0:
                (x, y, w, h) = cv2.boundingRect(contour)
                if save_artifacts:
                    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 1)

                mask = np.zeros_like(img_erode)
                cv2.drawContours(mask, [contour], 0, 255, -1)

                out = 255 * np.ones_like(img_erode)
                out[mask == 255] = self.thresh[mask == 255]

                letter_crop = out[y:y + h, x:x + w]
                size_max = max(w, h)

                # centering cropped letter inside the square
                letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
                if w > h:
                    y_pos = size_max // 2 - h // 2
                    letter_square[y_pos:y_pos + h, 0:w] = letter_crop
                elif w < h:
                    x_pos = size_max // 2 - w // 2
                    letter_square[0:h, x_pos:x_pos + w] = letter_crop
                else:
                    letter_square = letter_crop

                letters.append(
                    ExtractedLetter((x, y), cv2.resize(letter_square, (height, width), interpolation=cv2.INTER_AREA),
                                    w, h)
                )

        if save_artifacts:
            artifacts = os.path.join(self.directory, "artifacts")
            try:
                os.mkdir(artifacts)
            except FileExistsError:
                pass
            cv2.imwrite(os.path.join(artifacts, "last.png"), output)

        letters.sort()

        return letters


if __name__ == '__main__':
    extractor = ImageExtractor("sentence.png", "../images")
    extractor.process()
    for i, letter in enumerate(extractor.extract_letters(28, 28, save_artifacts=True)):
        cv2.imshow(f"Letter #{i}", letter.image)
    cv2.waitKey(0)
