import os
import typing as tp

import cv2
import numpy as np

from preprocessing.extracted_letter import ExtractedLetter


class ImageExtractor:
    THRESHOLD = 127
    ARTIFACTS_DIRECTORY = "artifacts"

    def __init__(self, image: tp.Union[str, np.ndarray]):
        self.image: np.ndarray
        if isinstance(image, str):
            self.image = cv2.imread(image)
        else:
            self.image = image

        self.thresh = ImageExtractor.threshold_it(self.image)

    @staticmethod
    def threshold_it(image: np.ndarray, blur=True):
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if blur:
            grayscale = cv2.GaussianBlur(grayscale, (5, 5), 0)

        _, thresh = cv2.threshold(grayscale, ImageExtractor.THRESHOLD, 255,
                                  cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        return thresh

    @staticmethod
    def square_it(image: np.ndarray, preferred_shape: tp.Tuple[int, int]) -> np.ndarray:
        c_height, c_width = image.shape
        longest = max(image.shape)
        squared_letter = 255 * np.ones(shape=[longest, longest], dtype=np.uint8)
        if c_width > c_height:
            squared_y = (longest - c_height) // 2
            squared_letter[squared_y:squared_y + c_height, 0:c_width] = image
        elif c_width < c_height:
            squared_x = (longest - c_width) // 2
            squared_letter[0:c_height, squared_x:squared_x + c_width] = image
        else:
            squared_letter = image

        return cv2.resize(squared_letter, preferred_shape, interpolation=cv2.INTER_AREA)

    def extract_letters(self, height, width, eroding=2, save_artifacts=False) -> tp.List[ExtractedLetter]:
        # removing some small defects
        opening = cv2.morphologyEx(self.thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        # increasing contours of letters for later use in connected components method
        img_erode = cv2.erode(opening, np.ones((eroding, eroding), np.uint8), iterations=3)

        contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        artifact = None
        if save_artifacts:
            artifact = self.image.copy()

        extracted: tp.List[ExtractedLetter] = []
        for index, contour in enumerate(contours):
            # checking for parent contour
            if hierarchy[0][index][3] == 0:
                (x, y, r_width, r_height) = cv2.boundingRect(contour)
                if save_artifacts:
                    cv2.rectangle(artifact, (x, y), (x + r_width, y + r_height), (0, 0, 255), 1)

                # making mask for the contour
                mask = np.zeros_like(img_erode)
                cv2.drawContours(mask, [contour], 0, 255, -1)

                # take only the part that is inside the current contour
                out = 255 * np.ones_like(img_erode)
                out[mask == 255] = self.thresh[mask == 255]

                letter_countours = out[y:y + r_height, x:x + r_width]

                coordinates = (x, y)
                image = ImageExtractor.square_it(letter_countours, (height, width))
                extracted.append(
                    ExtractedLetter(
                        image,
                        coordinates,
                        r_width,
                        r_height
                    )
                )

        if save_artifacts:
            try:
                os.mkdir(ImageExtractor.ARTIFACTS_DIRECTORY)
            except FileExistsError:
                pass
            cv2.imwrite(os.path.join(ImageExtractor.ARTIFACTS_DIRECTORY, "last.png"), artifact)

        return sorted(extracted)


if __name__ == '__main__':
    extractor = ImageExtractor("../images/photo.png")
    for i, letter in enumerate(extractor.extract_letters(28, 28, save_artifacts=True)):
        cv2.imshow(f"Letter #{i}", letter.image)
    cv2.waitKey(0)
