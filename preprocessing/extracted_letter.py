import typing as tp
from dataclasses import dataclass

import numpy as np


@dataclass
class ExtractedLetter:
    image: np.ndarray
    coordinates: tp.Tuple[int, int]
    height: int = 28
    width: int = 28

    def __lt__(self, other: "ExtractedLetter"):
        return self.coordinates < other.coordinates
