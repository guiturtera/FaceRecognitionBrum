import numpy

class Picture():
    def __init__(self, name: str, encoding: numpy.ndarray) -> None:
        self.name = name
        self.encoding = encoding

