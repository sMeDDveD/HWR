from model.predictor import Predictor
from postprocessing.word_collector import letters_to_words, correct_word
from preprocessing.extraction import ImageExtractor


class Recognizer:
    def __init__(self, path_to_model):
        self.predictor = Predictor(path_to_model)

    def recognize(self, image, directory) -> str:
        extractor = ImageExtractor(image, directory)
        extractor.process()

        letters = extractor.extract_letters(28, 28, save_artifacts=True)

        self.predictor.predict_all(letters)

        words = letters_to_words(letters)
        words = map(correct_word, words)

        return " ".join(words)
