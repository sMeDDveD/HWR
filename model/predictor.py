from typing import List

import keras.models
import numpy as np
from matplotlib import pyplot as plt

from preprocessing.letter import ExtractedLetter

MAPPING = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 66, 67, 68, 69,
           70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
           85, 86, 87, 88, 89, 90, 97, 98, 99, 100, 101, 102, 103, 104,
           105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
           117, 118, 119, 120, 121, 122]


def predicted_class_to_letter(prediction):
    return chr(MAPPING[prediction])


class Predictor:

    def __init__(self, path_to_model):
        self.model = keras.models.load_model(path_to_model)

    @staticmethod
    def _to_array(letter: ExtractedLetter):
        letter_array = np.expand_dims(letter.image, axis=0)
        letter_array = 1 - letter_array / 255.0
        return letter_array

    def predict(self, letter: ExtractedLetter):
        letter_array = Predictor._to_array(letter)
        letter_array = letter_array.reshape((1, 28, 28, 1))

        prediction = self.model.predict(letter_array)[0]

        result = np.argmax(prediction, axis=-1)
        letter.prediction = predicted_class_to_letter(result)

        for class_index in range(len(prediction)):
            if prediction[class_index] > 0.3:
                letter.candidates += predicted_class_to_letter(class_index)

    def predict_all(self, letters: List[ExtractedLetter]):
        for letter in letters:
            self.predict(letter)
