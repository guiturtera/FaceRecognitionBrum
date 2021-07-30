from face_recognition.api import face_encodings
import numpy

class Picture():
    def __init__(self, name: str, encodings, face_locations) -> None:
        self.name = name
        self.encodings = encodings
        self.face_locations = face_locations

