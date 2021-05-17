import os
import typing as tp

import cv2 as c
import numpy as np

from preprocessing.extracted_letter import ExtractedLetter


class ImageExtractor:
    THRESHOLD = 127
    ARTIFACTS_DIRECTORY = "artifacts"
    NO_PARENT = 0

    def __init__(self, image: tp.Union[str, np.ndarray]):
        self.image: np.ndarray
        if isinstance(image, str):
            self.image = c.imread(image)
        else:
            self.image = image

        self.thresh = ImageExtractor.threshold_it(self.image)

    @staticmethod
    def threshold_it(image: np.ndarray, blur=True):
        grayscale = c.cvtColor(image, c.COLOR_BGR2GRAY)
        if blur:
            grayscale = c.GaussianBlur(grayscale, (5, 5), 0)

        _, thresh = c.threshold(grayscale, ImageExtractor.THRESHOLD, 255,
                                  c.THRESH_OTSU + c.THRESH_BINARY)
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

        return c.resize(squared_letter, preferred_shape, interpolation=c.INTER_AREA)

    def _eroding(self, eroding) -> np.ndarray:
        # removing some small defects
        opening = c.morphologyEx(
            self.thresh,
            c.MORPH_OPEN,
            c.getStructuringElement(c.MORPH_ELLIPSE, (5, 5))
        )
        # increasing contours of letters for later use in connected components method
        img_erode = c.erode(
            opening,
            np.ones((eroding, eroding), np.uint8),
            iterations=2
        )

        return img_erode

    def extract_letters(self, height, width, eroding=2, save_artifacts=False) -> tp.List[ExtractedLetter]:
        eroded = self._eroding(eroding)
        all_occurrences, tree = c.findContours(
            eroded,
            c.RETR_TREE,
            c.CHAIN_APPROX_NONE
        )
        low_tree = tree[0]

        artifact = None
        if save_artifacts:
            artifact = self.image.copy()

        extracted: tp.List[ExtractedLetter] = []
        for index, text_element in enumerate(all_occurrences):
            # checking for parent contour
            if low_tree[index][3] == ImageExtractor.NO_PARENT:
                (r_x, r_y, r_width, r_height) = c.boundingRect(text_element)
                if save_artifacts:
                    c.rectangle(artifact, (r_x, r_y), (r_x + r_width, r_y + r_height), (0, 0, 255), 1)

                # making mask for the contour
                mask = np.zeros_like(eroded)
                c.drawContours(mask, [text_element], 0, 255, -1)

                # take only the part that is inside the current contour
                out = 255 * np.ones_like(eroded)
                out[mask == 255] = self.thresh[mask == 255]

                letter_countours = out[r_y:r_y + r_height, r_x:r_x + r_width]

                coordinates = (r_x, r_y)
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
            c.imwrite(os.path.join(ImageExtractor.ARTIFACTS_DIRECTORY, "last.png"), artifact)

        return sorted(extracted)


if __name__ == '__main__':
    extractor = ImageExtractor("../images/photo.png")
    for i, letter in enumerate(extractor.extract_letters(28, 28, save_artifacts=True)):
        c.imshow(f"Letter #{i}", letter.image)
    c.waitKey(0)
