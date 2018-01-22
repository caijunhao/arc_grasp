import numpy as np
import cv2

import argparse
import os

parser = argparse.ArgumentParser(description='create tensorflow record')
parser.add_argument('--data_path', default='training/label-aug', type=str, help='Path to data set.')
parser.add_argument('--output_path', default='training/label_aug_map', type=str, help='Output path.')
args = parser.parse_args()


def convert_label_to_map(label):
    r = (label == 0).astype(np.uint8)  # bad grasp point
    g = (label == 128).astype(np.uint8)  # good grasp point
    b = (label == 255).astype(np.uint8)  # background
    label_map = np.stack([b, g, r], axis=2)
    return label_map


# def convert_label_to_map(label):
#     r = ((label == 0)*255.0).astype(np.uint8)  # bad grasp point
#     g = ((label == 128)*255.0).astype(np.uint8)  # good grasp point
#     b = ((label == 255)*255.0).astype(np.uint8)  # background
#     label_map = np.stack([b, g, r], axis=2)
#     return label_map


def main():
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    file_list = os.listdir(args.data_path)
    for file_name in file_list:
        label = cv2.imread(os.path.join(args.data_path, file_name), cv2.IMREAD_ANYDEPTH)
        label_map = convert_label_to_map(label)
        cv2.imwrite(os.path.join(args.output_path, file_name), label_map)


if __name__ == '__main__':
    main()
