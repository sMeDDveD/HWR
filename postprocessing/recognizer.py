import os.path

import numpy as np

from classifier.predictor import Predictor
from postprocessing.word_collector import letters_to_words, correct_word
from preprocessing.extraction import ImageExtractor


class Recognizer:
    def __init__(self, path_to_model):
        self.predictor = Predictor(path_to_model)

    def recognize(self, image: np.ndarray) -> str:
        extractor = ImageExtractor(image)

        extracted_letters = extractor.extract_letters(28, 28, eroding=5, save_artifacts=True)
        predicted_letters = self.predictor.predict_all(extracted_letters)
        words = letters_to_words(predicted_letters)
        return " ".join(words)
