import cv2
import numpy as np

import os
import argparse

parser = argparse.ArgumentParser(description='process labels')
parser.add_argument('--data_dir',
                    default='/data/arc_mitprinceton_grasping_dataset/parallel-jaw-grasping-dataset',
                    type=str)
parser.add_argument('--color_img_dir',
                    default='/data/arc_mitprinceton_grasping_dataset/parallel-jaw-grasping-dataset/heightmap-color',
                    type=str)
parser.add_argument('--depth_img_dir',
                    default='/data/arc_mitprinceton_grasping_dataset/parallel-jaw-grasping-dataset/heightmap-depth',
                    type=str)
parser.add_argument('--label_dir',
                    default='/data/arc_mitprinceton_grasping_dataset/parallel-jaw-grasping-dataset/label',
                    type=str)
parser.add_argument('--output_dir',
                    default='/data/arc_mitprinceton_grasping_dataset/parallel-jaw-grasping-dataset/training',
                    type=str)
args = parser.parse_args()

if not os.path.exists(args.data_dir):
    os.makedirs(args.output_dir)
    os.makedirs(os.path.join(args.output_dir, 'color'))
    os.makedirs(os.path.join(args.output_dir, 'depth'))
    os.makedirs(os.path.join(args.output_dir, 'label'))

with open(args.data_dir + 'train-processed-split.txt', 'w') as f:
    idx = f.readline()
    good_grasp_pixel_label = np.loadtxt(os.path.join(args.label_dir, idx+'.good.txt'))
    bad_grasp_pixel_label = np.loadtxt(os.path.join(args.label_dir, idx+'.bad.txt'))





