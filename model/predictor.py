import string
from typing import List

import keras.models
import numpy as np

from preprocessing.letter import ExtractedLetter

DIGITS = [ord(c) for c in string.digits]
LETTERS = [ord(c) for c in string.ascii_uppercase + string.ascii_lowercase]

# map classes to ascii codes
MAPPING = DIGITS + LETTERS


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

        # keep the most likely candidates
        for class_index in range(len(prediction)):
            if prediction[class_index] > 0.3:
                letter.candidates += predicted_class_to_letter(class_index)

    def predict_all(self, letters: List[ExtractedLetter]):
        for letter in letters:
            self.predict(letter)
