import numpy as np
import cv2

import argparse
import os

parser = argparse.ArgumentParser(description='create tensorflow record')
parser.add_argument('--data_path', default='training/depth', type=str, help='Path to data set.')
parser.add_argument('--output_path', default='training/encoded_depth', type=str, help='Output path.')
args = parser.parse_args()


def encode_depth(depth):
    r = depth / 256 / 256
    g = depth / 256
    b = depth % 256
    # encoded_depth = np.stack([r, g, b], axis=2).astype(np.uint8)
    encoded_depth = np.stack([b, g, r], axis=2).astype(np.uint8)  # use bgr order due to cv2 format
    return encoded_depth


def main():
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    file_list = os.listdir(args.data_path)
    for file_name in file_list:
        depth = cv2.imread(os.path.join(args.data_path, file_name), cv2.IMREAD_ANYDEPTH)
        encoded_depth = encode_depth(depth)
        cv2.imwrite(os.path.join(args.output_path, file_name), encoded_depth)


if __name__ == '__main__':
    main()

