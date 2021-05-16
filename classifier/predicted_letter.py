from dataclasses import dataclass

import numpy as np

from preprocessing.extracted_letter import ExtractedLetter


@dataclass
class PredictedLetter:
    probable_letter: str
    probabilities: np.ndarray

    extracted: ExtractedLetter
