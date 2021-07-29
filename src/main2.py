import face_recognition
import cv2
import numpy as np
import os
import glob

from pkg_resources import normalize_path

faces_encodings = []
faces_names = []

cur_directory = os.path.dirname(os.getcwd())
images_path = os.path.join(cur_directory, "images/")
#images_path = os.path.normpath(images_path)

face_encodings = []
faces_names = []

list_of_files = [f for f in glob.glob(images_path + '*.jpg')]
number_of_images = len(list_of_files)
names = list_of_files.copy()

for i in range(number_of_images):
    globals()['image_{}'.format(i)] = face_recognition.load_image_file(list_of_files[i])
    globals()['image_encoding_{}'.format(i)] = face_recognition.face_encodings(globals()['image_{}'.format(i)])[0]
    face_encodings.append(globals()['image_encoding_{}'.format(i)])

for i in range(len(names)):
    names[i] = os.path.basename(names[i]).replace('.jpg', "")
    faces_names.append(names[i])

print(faces_names)