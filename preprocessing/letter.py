class ExtractedLetter:
    @property
    def image(self):
        return self._image

    def __init__(self, coordinates, image, width=28, height=28):
        self._image = image
        self.coordinates = coordinates
        self.width = width
        self.height = height
        self.prediction = 0
        self.candidates = []

    def __lt__(self, other):
        return self.coordinates < other.coordinates
