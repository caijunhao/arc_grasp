import numpy as np
import cv2

import argparse
import glob
import os

parser = argparse.ArgumentParser(description='process labels')
parser.add_argument('--data_dir', required=True, type=str, help='path to the data')
parser.add_argument('--config_dir', default='', type=str, help='path to the initialize configuration')
parser.add_argument('--output', required=True, type=str, help='path to save the process data')
args = parser.parse_args()


def get_grasp_center(grasp_labels):
    # grasp_label: numpy array
    row, _ = grasp_labels.shape
    grasp_center = np.empty((row, 2), dtype=np.float32)
    grasp_center[:, 0] = (grasp_labels[:, 0] + grasp_labels[:, 2]) / 2.0
    grasp_center[:, 1] = (grasp_labels[:, 1] + grasp_labels[:, 3]) / 2.0
    return grasp_center


def get_grasp_angle(grasp_label):
    pt1 = grasp_label[0:2]
    pt2 = grasp_label[2:]
    angle = np.arctan2(pt2[1]-pt1[1], pt2[0]-pt1[1]) - np.pi / 2
    if angle < 0:
        angle += np.pi * 2
    if angle > np.pi:
        angle -= np.pi
    return angle


def get_neglect_points(rect):
    # rect_min = np.asanyarray([[rect[1], rect[0]]])  # top left corner
    # rect_max = np.asanyarray([[rect[1] + rect[3], rect[0] + rect[2]]])  # bottom right corner
    x, y = np.meshgrid(np.arange(rect[1], rect[1] + rect[3]), np.arange(rect[0], rect[0] + rect[2]))
    neglect_points = np.stack((x.flatten(), y.flatten()), axis=1)
    return neglect_points


def rotate(points, center, angle):
    points = points.copy()
    h = center[0]
    w = center[1]
    points[:, 0] -= h
    points[:, 1] -= w
    rotate_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                              [np.sin(angle), np.cos(angle)]])
    points = np.dot(rotate_matrix, points.T).T
    points[:, 0] += h
    points[:, 1] += w
    return points


def main():
    if not os.path.exists(args.output):
        os.mkdir(args.output)
        os.mkdir(os.path.join(args.output, 'color'))
        os.mkdir(os.path.join(args.output, 'depth'))
        os.mkdir(os.path.join(args.output, 'label'))
        os.mkdir(os.path.join(args.output, 'label-aug'))
    f_o = open(os.path.join(args.output, 'file_name.txt'), 'w')

    # load data
    data_dir = os.listdir(args.data_dir)
    for data_id in data_dir:
        parent_dir = os.path.join(args.data_dir, data_id)
        b_depth_height_map = cv2.imread(os.path.join(parent_dir, 'background_depth_height_map.png'),
                                        cv2.IMREAD_ANYDEPTH).astype(np.float32)
        label_files = os.listdir(os.path.join(parent_dir, 'label'))
        with open(os.path.join(parent_dir, 'file_name.txt'), 'r') as f:
            file_names = f.readlines()
        for file_name in file_names:
            file_name = file_name[:-1]
            color = cv2.imread(os.path.join(parent_dir, 'height_map_color', file_name+'.png'))
            depth = cv2.imread(os.path.join(parent_dir, 'height_map_depth', file_name+'.png'),
                               cv2.IMREAD_ANYDEPTH).astype(np.float32)
            diff_depth = b_depth_height_map - depth
            diff_depth[np.where(diff_depth < 0)] = 0
            # diff_depth += 30
            diff_depth = diff_depth.astype(np.uint16)
            # pad color and depth images
            pad_size = 44
            pad_color = np.zeros((288, 288, 3), dtype=np.uint8)
            pad_color[pad_size:pad_size + 200, pad_size:pad_size + 200, :] = color
            pad_depth = np.zeros((288, 288), dtype=np.uint16)
            pad_depth[pad_size:pad_size + 200, pad_size:pad_size + 200] = diff_depth
            # rect = np.loadtxt(os.path.join(parent_dir, 'label', file_name+'.rectangle.txt')).astype(np.int)
            # neglect_points = get_neglect_points(rect) + pad_size
            neglect_points = np.loadtxt(os.path.join(parent_dir, 'label', file_name+'.object_points.txt')) + pad_size
            if file_name+'.good.txt' in label_files:
                good_pixel_labels = np.loadtxt(os.path.join(parent_dir, 'label', file_name+'.good.txt')) + pad_size
                grasp_centers = get_grasp_center(good_pixel_labels)
                angle = get_grasp_angle(good_pixel_labels[0])
                angle_idx = int(round(angle/((11.25/360.0)*np.pi*2)))
                for i in range(16):
                    grasp_label = np.zeros((36, 36, 3), dtype=np.uint8)  # bgr for opencv
                    grasp_label[..., 0] = 255
                    rotated_neglect_points = rotate(neglect_points, (144, 144), (11.25*i/360.0)*np.pi*2)
                    rotated_neglect_points = np.round(rotated_neglect_points / 8.0).astype(np.int)
                    grasp_label[rotated_neglect_points[:, 0], rotated_neglect_points[:, 1], 0] = 0
                    if i == angle_idx:
                        rotated_grasp_centers = rotate(grasp_centers, (144, 144), (11.25*i/360.0)*np.pi*2)
                        rotated_grasp_centers = np.round(rotated_grasp_centers / 8.0).astype(np.int)
                        grasp_label[rotated_grasp_centers[:, 0], rotated_grasp_centers[:, 1], 1] = 255
                    mtx = cv2.getRotationMatrix2D((144, 144), 11.25*i, 1)
                    rotated_color = cv2.warpAffine(pad_color, mtx, (288, 288))
                    rotated_depth = cv2.warpAffine(pad_depth, mtx, (288, 288))
                    cv2.imwrite(os.path.join(args.output, 'label', data_id+'-'+file_name+'-{:02d}.png'.format(i)),
                                grasp_label)
                    cv2.imwrite(os.path.join(args.output, 'color', data_id+'-'+file_name+'-{:02d}.png'.format(i)),
                                rotated_color)
                    cv2.imwrite(os.path.join(args.output, 'depth', data_id+'-'+file_name+'-{:02d}.png'.format(i)),
                                rotated_depth)
                    f_o.write(data_id + '-' + file_name + '-{:02d}\n'.format(i))
            else:
                bad_pixel_labels = np.loadtxt(os.path.join(parent_dir, 'label', file_name + '.bad.txt')) + pad_size
                grasp_centers = get_grasp_center(bad_pixel_labels)
                angle = get_grasp_angle(bad_pixel_labels[0])
                angle_idx = round(angle / ((11.25 / 360.0) * np.pi * 2))
                for i in range(16):
                    grasp_label = np.zeros((36, 36, 3), dtype=np.uint8)  # bgr for opencv
                    grasp_label[..., 0] = 255
                    rotated_neglect_points = rotate(neglect_points, (144, 144), (11.25 * i / 360.0) * np.pi * 2)
                    rotated_neglect_points = np.round(rotated_neglect_points / 8.0).astype(np.int)
                    grasp_label[rotated_neglect_points[:, 0], rotated_neglect_points[:, 1], 0] = 0
                    if i == angle_idx:
                        rotated_grasp_centers = rotate(grasp_centers, (144, 144), (11.25 * i / 360.0) * np.pi * 2)
                        rotated_grasp_centers = np.round(rotated_grasp_centers / 8.0).astype(np.int)
                        grasp_label[rotated_grasp_centers[:, 0], rotated_grasp_centers[:, 1], 2] = 255
                    mtx = cv2.getRotationMatrix2D((144, 144), 11.25 * i, 1)
                    rotated_color = cv2.warpAffine(pad_color, mtx, (288, 288))
                    rotated_depth = cv2.warpAffine(pad_depth, mtx, (288, 288))
                    cv2.imwrite(os.path.join(args.output, 'label', data_id+'-'+file_name + '-{:02d}.png'.format(i)),
                                grasp_label)
                    cv2.imwrite(os.path.join(args.output, 'color', data_id+'-'+file_name + '-{:02d}.png'.format(i)),
                                rotated_color)
                    cv2.imwrite(os.path.join(args.output, 'depth', data_id+'-'+file_name + '-{:02d}.png'.format(i)),
                                rotated_depth)
                    f_o.write(data_id+'-'+file_name+'-{:02d}\n'.format(i))
    f_o.close()


if __name__ == '__main__':
    main()

