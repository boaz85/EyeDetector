import argparse

import cv2
import numpy as np
from matplotlib import pyplot as plt, patches
from mpl_toolkits.mplot3d import Axes3D

Axes3D
from Classifiers import LogisticRegressionClassifier

parser = argparse.ArgumentParser()

parser.add_argument("-o", "--optimized-theta",
                    help='Optimized theta file', required=True)

parser.add_argument("-i", "--input",
                    help='Input image path to locate the eyes in', required=True)

parser.add_argument("-t", "--threshold", type=float,
                    help='Match threshold', required=True)

def get_input_data(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def get_next_window(img, target_size, step_size, scale_factor=1.3):

    window_size = np.array(target_size)

    while window_size[0] < img.shape[0] and window_size[1] < img.shape[1]:

        h_offset = 0
        v_offset = 0

        while v_offset + window_size[1] < img.shape[0]:
            while h_offset + window_size[0] < img.shape[1]:

                window = img[v_offset: v_offset + window_size[1], h_offset: h_offset + window_size[0]]
                window = cv2.resize(window, tuple(target_size))
                yield (h_offset, v_offset), (window_size[1], window_size[0]), window

                h_offset += step_size

            h_offset = 0
            v_offset += step_size

        window_size = (window_size * scale_factor).astype(int)

def display_results(matches):

    fig,ax = plt.subplots(1)
    ax.imshow(input_image, cmap='gray')
    rect = None

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    t_text = fig.text(0.05, 0.95, '', transform=fig, fontsize=14, verticalalignment='top', bbox=props)

    for match in sorted(matches, key=lambda x: x['match'], reverse=True):
        if rect:
            rect.remove()
        t_text.set_text('Match: ' + str(match['match']))
        rect = patches.Rectangle(match['position'], match['dimensions'][0], match['dimensions'][1], linewidth=1,
                                 edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.pause(2)

if __name__ == '__main__':

    args = parser.parse_args()

    classifier = LogisticRegressionClassifier()
    optimized_theta = np.load('optimized_theta.npy')

    input_image = get_input_data(args.input)

    matches = []

    for (x, y), (w, h), window in get_next_window(input_image, [30, 30], 10):
        match = classifier.classify(window, optimized_theta)
        if match > args.threshold:
            matches.append({'match': match, 'position': (x, y), 'dimensions': (w, h), 'img': window})

    print '{matches} matches found'.format(matches=len(matches))
    display_results(matches)

