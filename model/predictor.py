import keras.models

class Predictor:
    def __init__(self, path_to_model):
        self.model = keras.models.load_model(path_to_model)

Predictor("models/augmentation.h5")