from genericpath import exists
from operator import mod
import pickle
from typing import List
import face_recognition
import os
import cv2
from face_recognition.api import face_encodings, face_locations
import json

from picture import Picture

KNOWN_FACES_DIR = "../known-images"
LOADED_FACES_FILE = f"../loaded-faces.pickle"
UNKNOWN_FACES_DIR = "../unknown-images"
TOLERANCE = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = "cnn"   # Could also be hog

def load_pictures_objects(pictures_path: str):
    pictures = []
    for name in os.listdir(pictures_path):

        image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}")
        encoding = face_recognition.face_encodings(image)[0]
        pictures.append(Picture(name, encoding))

    return pictures

def serialize_pictures(pictures: List[Picture], fileToCreate: str):
    with open(fileToCreate, 'wb') as file:
        pickle.dump(pictures, file, protocol=pickle.HIGHEST_PROTOCOL)

def deserialize_pictures(fileToRead):
    with open(fileToRead, "rb") as file:
        pictures = pickle.load(file)

    return pictures

pictures = []

print("loading all unknown faces...")
if exists(LOADED_FACES_FILE):
    print("deserializing object")
    pictures = deserialize_pictures(LOADED_FACES_FILE)
else:
    print("loading images")
    pictures = load_pictures_objects(KNOWN_FACES_DIR)
    serialize_pictures(pictures, LOADED_FACES_FILE)

known_faces = []
known_names = []

for i in pictures:
    known_faces.append(i.encoding)
    known_names.append(i.name)

print('done!')

print(known_faces)
input()

print(type(known_faces[0]))
print("processing unknown face")
input()
for filename in os.listdir(UNKNOWN_FACES_DIR):
    #print(filename)
    image = face_recognition.load_image_file(f"{UNKNOWN_FACES_DIR}/{filename}")
    locations = face_recognition.face_locations(image, model=MODEL)
    encodings = face_recognition.face_encodings(image, locations)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for face_encoding, face_location, in zip(encodings, locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        match = None
        if True in results:
            match = known_names[results.index(True)]
            print(f"Match found: {match}")

            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])

            color = [200, 162, 47]
            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 22)
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
            cv2.putText(image, match, (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200,200,200), FONT_THICKNESS)

    cv2.imshow(filename, image)
    cv2.waitKey(200000)
    cv2.destroyWindow(filename)