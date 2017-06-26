import argparse
import os
from tempfile import TemporaryFile

import numpy as np
import cv2
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument("-s", "--source-dir",
                    help='Source images directory', required=True)

parser.add_argument("-o", "--output-dir",
                    help='Output images directory', required=True)

parser.add_argument("-d", "--output-dimensions", default='200x200',
                    help='E.g 50x50')

eye_cascade = cv2.CascadeClassifier('/Users/boazsh/opencv/data/haarcascades/haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier('/Users/boazsh/opencv/data/haarcascades/haarcascade_frontalface_default.xml')

def next_image(source_dir):

    for subdir, dirs, files in os.walk(source_dir):
        for file_name in files:
            filepath = subdir + os.sep + file_name
            file = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if file is not None:
                yield file_name, file

def invalid_dimensions(img, normal_dimensions):
    image_dimensions = img.shape
    if abs(normal_dimensions[0] - image_dimensions[0]) > (normal_dimensions[0] / 2):
        return True

    if abs(normal_dimensions[1] - image_dimensions[1]) > (normal_dimensions[1] / 2):
        return True

    return False

def detect_eyes(img, required_dimensions):

    all_eyes = []
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    required_ratio = required_dimensions[0] / float(required_dimensions[1])

    for (x,y,w,h) in faces:

        eyes = eye_cascade.detectMultiScale(img[y:y+h, x:x+w])

        for (ex,ey,ew,eh) in eyes:

            if eh / float(ew)  != required_ratio:
                ew = int(eh / required_ratio)

            all_eyes.append(img[y+ey:y+ey+eh, x+ex:x+ex+ew])

    return all_eyes

def normalize_dimensions(img, normal_dimensions):

    mul = normal_dimensions[0] / float(img.shape[0])
    return cv2.resize(img, (0,0), fx=mul, fy=mul)

def save_to_outputdir(output_dir, img, img_name):
    cv2.imwrite(os.sep.join([output_dir, img_name]), img)

if __name__ == '__main__':

    args = parser.parse_args()
    normal_dimensions = [int(val) for val in args.output_dimensions.split('x')]
    i = 0
    columns = ['x_' + str(i) for i in range(normal_dimensions[0] * normal_dimensions[1])] + ['y']
    train_mat = []

    for img_name, img in next_image(args.source_dir):

        eyes = detect_eyes(img, normal_dimensions)

        for j, eye in enumerate(eyes):
            # if invalid_dimensions(img, normal_dimensions):
            #     print 'Invalid image dimensions: ' + img_name
            #     continue

            scaled = normalize_dimensions(eye, normal_dimensions)
            new_name = img_name.split('.')[0] + '_' + str(i) + '_' + str(j) + '.' + img_name.split('.')[1]
            save_to_outputdir(args.output_dir, scaled, new_name)
            train_mat.append(scaled.reshape(-1))
            i += 1

    np.save(args.output_dir + os.sep + 'train_data', np.array(train_mat))