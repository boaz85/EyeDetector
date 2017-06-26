import argparse
import os

import cv2
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("-p", "--positive-examples",
                    help='Positive train images directory path', required=True)

parser.add_argument("-n", "--negative-examples",
                    help='Negative train images directory path', required=True)

parser.add_argument("-o", "--output-file",
                    help='Output file name', required=True)

def next_image(source_dir):

    for subdir, dirs, files in os.walk(source_dir):
        for file_name in files:
            filepath = subdir + os.sep + file_name
            file = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if file is not None:
                yield file_name, file

if __name__ == '__main__':

    args = parser.parse_args()

    train_mat = []

    for _, image in next_image(args.positive_examples):
        train_mat.append(np.append(image.reshape(-1), [1]))

    for _, image in next_image(args.negative_examples):
        train_mat.append(np.append(image.reshape(-1), [0]))

    train_mat = np.array(train_mat)

    np.save(args.output_file, train_mat)