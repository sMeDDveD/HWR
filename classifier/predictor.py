import string
import typing as tp

import keras.models
import numpy as np

from classifier.predicted_letter import PredictedLetter
from preprocessing.extracted_letter import ExtractedLetter

DIGITS = [ord(c) for c in string.digits]
LETTERS = [ord(c) for c in string.ascii_uppercase + string.ascii_lowercase]

# map classes to ascii codes
MAPPING = DIGITS + LETTERS


def predicted_class_to_letter(predicted_label: int) -> str:
    return chr(MAPPING[predicted_label])


class Predictor:
    def __init__(self, path_to_model: str):
        self.model: keras.Model = keras.models.load_model(path_to_model)

    @staticmethod
    def _to_array(letter: ExtractedLetter) -> np.ndarray:
        letter_array = np.expand_dims(letter.image, axis=0)
        letter_array = 1 - letter_array / 255.0
        return letter_array

    def predict(self, extracted_letter: ExtractedLetter) -> PredictedLetter:
        letter_array = Predictor._to_array(extracted_letter)
        letter_array = letter_array.reshape((1, 28, 28, 1))

        prediction = self.model.predict(letter_array)[0]

        result = np.argmax(prediction, axis=-1)

        return PredictedLetter(
            predicted_class_to_letter(result),
            prediction,
            extracted_letter
        )

    def predict_all(self, extracted_letters: tp.List[ExtractedLetter]) -> tp.List[PredictedLetter]:
        data = np.array(
            [Predictor._to_array(letter).reshape((28, 28, 1)) for letter in extracted_letters]
        )

        predictions = self.model.predict(data)

        predicted_letters: tp.List[PredictedLetter] = []
        for letter, prediction in zip(extracted_letters, predictions):
            predicted_letters.append(
                PredictedLetter(
                    predicted_class_to_letter(np.argmax(prediction)),
                    prediction,
                    letter
                )
            )

        return predicted_letters
