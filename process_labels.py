import cv2
import numpy as np

import os
import argparse
import matplotlib.pyplot as plt
import math
from image_procesing import rotate
from image_process2 import rotate_bound

parser = argparse.ArgumentParser(description='process labels')
parser.add_argument('--data_dir',
                    default='/home/caijunhao/Desktop/training',
                    type=str)
parser.add_argument('--heightmapColor_dir',
                    default='/data/arc_mitprinceton_grasping_dataset/parallel-jaw-grasping-dataset/heightmap-color',
                    type=str)
parser.add_argument('--heightmapDepth_dir',
                    default='/data/arc_mitprinceton_grasping_dataset/parallel-jaw-grasping-dataset/heightmap-depth',
                    type=str)
parser.add_argument('--label_dir',
                    default='/data/arc_mitprinceton_grasping_dataset/parallel-jaw-grasping-dataset/label1',
                    type=str)
parser.add_argument('--output_dir',
                    default='/home/caijunhao/Desktop/training',
                    type=str)
args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
    os.makedirs(os.path.join(args.output_dir, 'color'))
    os.makedirs(os.path.join(args.output_dir, 'depth'))
    os.makedirs(os.path.join(args.output_dir, 'label'))
    os.makedirs(os.path.join(args.output_dir, 'label-aug'))

f1 = open(args.data_dir + 'test-processed-split.txt', 'w')
f2 = open(args.data_dir + 'train-processed-split.txt', 'w')
# with open(args.data_dir + 'train-processed-split.txt', 'w') as f:
#     idx = f.readline()
#     good_grasp_pixel_label = np.loadtxt(os.path.join(args.label_dir, idx+'.good.txt'))
#     bad_grasp_pixel_label = np.loadtxt(os.path.join(args.label_dir, idx+'.bad.txt'))

labelFiles = []
labelFiles_list = os.listdir(args.label_dir)
for filename in labelFiles_list:
    if filename.endswith('good.txt'):
        labelFiles.append(filename)

i = 0
for sampleIdx in labelFiles:
    sampleName = sampleIdx[:6]+'.png'

    # Load color and depth heightmaps and zero-pad into 320x320 patch
    heightmapColor = np.zeros((320, 320, 3), dtype=np.uint8)
    h1 = cv2.imread(os.path.join(args.heightmapColor_dir, sampleName), cv2.IMREAD_UNCHANGED)
    heightmapColor[48:272, :320, :] = cv2.cvtColor(h1, cv2.COLOR_BGR2RGB)

    heightmapDepth = np.zeros((320, 320), dtype=np.uint16)
    heightmapDepth[48:272, :320] = cv2.imread(os.path.join(args.heightmapDepth_dir, sampleName), cv2.IMREAD_ANYDEPTH)

    # Load manual grasp labels
    try:
        goodGraspPixLabels = np.loadtxt(os.path.join(args.label_dir, sampleIdx[:6]+'.good.txt'))
    except:
        goodGraspPixLabels = []

    try:
        badGraspPixLabels = np.loadtxt(os.path.join(args.label_dir, sampleIdx[:6]+'.bad.txt'))
    except:
        badGraspPixLabels = []

    # Shift grasp labels (due to padding)
    goodGraspPixLabels[:, 1::2] = goodGraspPixLabels[:, 1::2]+48
    badGraspPixLabels[:, 1::2] = badGraspPixLabels[:, 1::2]+48

    plt.figure(1)
    plt.imshow(heightmapColor)
    # Generate heat map labels for good grasp annotations
    goodGraspLabels = np.zeros((40, 40, 16), dtype=np.uint8)
    for graspIdx in np.arange(goodGraspPixLabels.shape[0]):

        plt.plot([goodGraspPixLabels[graspIdx, 0], goodGraspPixLabels[graspIdx, 2]],
                 [goodGraspPixLabels[graspIdx, 1], goodGraspPixLabels[graspIdx, 3]])
        graspSampleCenter = np.mean((goodGraspPixLabels[graspIdx, :2], goodGraspPixLabels[graspIdx, 2:]), axis=0)
        graspSampleCenterDownsample = np.round((graspSampleCenter - 1)/8 + 1).astype(np.int)

        # Compute grasping direction and angle w.r.t. heightmap
        graspDirection = (goodGraspPixLabels[graspIdx, :2] - goodGraspPixLabels[graspIdx, 2:])/ \
                         np.linalg.norm(goodGraspPixLabels[graspIdx, :2] - goodGraspPixLabels[graspIdx, 2:])
        diffAngle = math.atan2(graspDirection[0] * 0 - graspDirection[1] * 1,
                               graspDirection[0] * 1 + graspDirection[1] * 0)
        diffAngle = diffAngle/math.pi*180

        while diffAngle < 0:
            diffAngle = diffAngle+360

        # Compute heat maps for each grasping direction
        rotIdx = np.round(diffAngle / (45 / 2.0) + 1).astype(np.int)
        goodGraspLabels[graspSampleCenterDownsample[1], graspSampleCenterDownsample[0], rotIdx-1] = 1
        rotIdx = np.mod(rotIdx-1+8, 16)+1
        goodGraspLabels[graspSampleCenterDownsample[1], graspSampleCenterDownsample[0], rotIdx-1] = 1

    plt.show()

    # Generate heat map labels for bad grasp annotations
    badGraspLabels = np.zeros((40, 40, 16), dtype=np.uint8)
    for graspIdx in np.arange(badGraspPixLabels.shape[0]):
        graspSampleCenter = np.mean((badGraspPixLabels[graspIdx, :2], badGraspPixLabels[graspIdx, 2:]), axis=0)
        graspSampleCenterDownsample = np.round((graspSampleCenter - 1) / 8 + 1).astype(np.int)

        # Compute grasping direction and angle w.r.t. heightmap
        graspDirection = (badGraspPixLabels[graspIdx, :2] - badGraspPixLabels[graspIdx, 2:]) / \
                         np.linalg.norm(badGraspPixLabels[graspIdx, :2] - badGraspPixLabels[graspIdx, 2:])
        diffAngle = math.atan2(graspDirection[0] * 0 - graspDirection[1] * 1,
                               graspDirection[0] * 1 + graspDirection[1] * 0)
        diffAngle = diffAngle / math.pi * 180

        while diffAngle < 0:
            diffAngle = diffAngle + 360

        # Compute heat maps for each grasping direction
        rotIdx = np.round(diffAngle / (45 / 2.0) + 1).astype(np.int)
        badGraspLabels[graspSampleCenterDownsample[1], graspSampleCenterDownsample[0], rotIdx-1] = 1
        rotIdx = np.mod(rotIdx - 1 + 8, 16) + 1
        badGraspLabels[graspSampleCenterDownsample[1], graspSampleCenterDownsample[0], rotIdx-1] = 1

    for rotIdx in np.arange(16):
        rotAngle = 360 -(45/2.0)*rotIdx

        sampleHeightmapColor = rotate(heightmapColor, rotAngle)
        sampleHeightmapDepth = rotate(heightmapDepth, rotAngle)
        sampleHeightmapLabel = np.ones((40, 40), dtype=np.uint8)*255
        sampleHeightmapLabelAug = np.ones((40, 40), dtype=np.uint8)*255
        goodGraspInd = rotate(goodGraspLabels[:, :, rotIdx], rotAngle)>0
        badGraspInd = np.array(rotate(badGraspLabels[:, :, rotIdx], rotAngle)>0)
        badGraspInd_ = np.array(rotate_bound(badGraspLabels[:, :, rotIdx], rotAngle)>0)
        plt.figure(2)
        plt.imshow(np.array(badGraspInd).astype(np.uint8))
        plt.figure(3)
        plt.imshow(badGraspLabels[:, :, rotIdx])
        # Dilate grasping labels (data augmentation)
        plt.show()

        ele_1 = cv2.getStructuringElement(shape=0, ksize=(3, 3))
        ele_2 = cv2.getStructuringElement(shape=0, ksize=(3, 3))
        ele_3 = cv2.getStructuringElement(shape=0, ksize=(5, 5))

        goodGraspIndAug = np.array(cv2.dilate(cv2.dilate(goodGraspInd.astype(np.uint8), ele_1), ele_2)) | \
                          np.array(cv2.dilate(goodGraspInd.astype(np.uint8), ele_3))

        badGraspIndAug = np.array(cv2.dilate(cv2.dilate(badGraspInd.astype(np.uint8), ele_1), ele_2)) | \
                          np.array(cv2.dilate(badGraspInd.astype(np.uint8), ele_3))

        # Combine good and bad grasp labels
        sampleHeightmapLabel[badGraspInd] = 0
        sampleHeightmapLabel[goodGraspInd] = 128
        sampleHeightmapLabelAug[badGraspIndAug.astype(np.bool)] = 0
        sampleHeightmapLabelAug[goodGraspIndAug.astype(np.bool)] = 128
        sampleHeightmapColor = cv2.cvtColor(sampleHeightmapColor, cv2.COLOR_RGB2BGR)

        # Save rotated heighmaps and grasp labels for training/testing
        cv2.imwrite(args.output_dir + '/color/' + sampleIdx[:6] + '-%02d.png' % rotIdx, sampleHeightmapColor)
        cv2.imwrite(args.output_dir + '/depth/' + sampleIdx[:6] + '-%02d.png' % rotIdx, sampleHeightmapDepth)
        cv2.imwrite(args.output_dir + '/label/' + sampleIdx[:6] + '-%02d.png' % rotIdx, sampleHeightmapLabel)
        cv2.imwrite(args.output_dir + '/label-aug/' + sampleIdx[:6] + '-%02d.png' % rotIdx, sampleHeightmapLabelAug)

        str_ = sampleIdx[:6] + '-%02d' % rotIdx
        if (i+1)%5 == 0:
            f1.write(str_+'\n')
        else:
            if np.sum((sampleHeightmapLabel<255).astype(np.int))>0:
                f2.write(str_+'\n')

        i=i+1
        # plt.figure(2)
        # plt.imshow(sampleHeightmapColor)
        # plt.figure(3)
        # plt.imshow(sampleHeightmapLabel)
       # plt.show()