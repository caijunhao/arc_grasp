import cv2
import numpy as np
from skimage import morphology

import argparse

parser = argparse.ArgumentParser(description='get height map')
parser.add_argument('--origin', default='demo/bin-position.txt', type=str)
parser.add_argument('--color_image', default='demo/input-0.color.png', type=str)
parser.add_argument('--depth_image', default='demo/input-0.depth.png', type=str)
parser.add_argument('--background_color_image', default='demo/background-0.color.png', type=str)
parser.add_argument('--background_depth_image', default='demo/background-0.depth.png', type=str)
parser.add_argument('--camera_intrinsics', default='demo/camera-0.intrinsics.txt', type=str)
parser.add_argument('--camera_pose', default='demo/camera-0.pose.txt', type=str)
parser.add_argument('--voxel_size', default=0.002, type=float)
args = parser.parse_args()

# read images and camera parameters
bin_middle_bottom = np.loadtxt(args.origin)
img = cv2.imread(args.color_image).astype(np.float32)/255.0
depth_img = cv2.imread(args.depth_image, cv2.IMREAD_ANYDEPTH).astype(np.float32)/10000.0
background_img = cv2.imread(args.background_color_image).astype(np.float32)/255.0
background_depth_img = cv2.imread(args.background_depth_image, cv2.IMREAD_ANYDEPTH).astype(np.float32)/10000.0
camera_intrinsics = np.loadtxt(args.camera_intrinsics)
camera_pose = np.loadtxt(args.camera_pose)

# do background substraction
foreground_mask_color = ~(np.sum(np.abs(img-background_img) < 0.3, 2) == 3)
foreground_mask_depth = (background_depth_img != 0) & (np.abs(depth_img - background_depth_img) > 0.02)
foreground_mask = foreground_mask_color | foreground_mask_depth

# project depth into camera space
x, y = np.meshgrid(range(img.shape[1]), range(img.shape[0]))
camera_x = np.reshape((x - camera_intrinsics[0, 2]) * depth_img / camera_intrinsics[0, 0], -1)
camera_y = np.reshape((y - camera_intrinsics[1, 2]) * depth_img / camera_intrinsics[1, 1], -1)
camera_z = np.reshape(depth_img, -1)
camera_points = np.stack([camera_x, camera_y, camera_z], axis=1)

# transform points to world coordinates
world_points = np.matmul(camera_points, camera_pose[0:3, 0:3].T) + camera_pose[0:3, 3].T

# get height map
height_map = np.zeros((200, 300))
height_map_color = np.zeros((200, 300, 3))
grid_origin = bin_middle_bottom - [0.3, 0.2, 0]
grid_mapping = np.stack([np.round((world_points[:, 0] - grid_origin[0]) / args.voxel_size),
                         np.round((world_points[:, 1] - grid_origin[1]) / args.voxel_size),
                         world_points[:, 2] - grid_origin[2]], axis=1)

# compute height map color
valid_pixel = (grid_mapping[:, 0] >= 0) & (grid_mapping[:, 0] <= 299) & \
              (grid_mapping[:, 1] >= 0) & (grid_mapping[:, 1] <= 199)  # & (grid_mapping[:, 2] > 0)
color_points = np.reshape(img, (-1, 3))
valid_grid_w = grid_mapping[valid_pixel, :][:, 0].astype(np.int64)
valid_grid_h = grid_mapping[valid_pixel, :][:, 1].astype(np.int64)
height_map_color[valid_grid_h, valid_grid_w, :] = color_points[valid_pixel, :]

# compute real height map with background substraction
valid_pixel = (grid_mapping[:, 0] >= 0) & (grid_mapping[:, 0] <=299) & \
              (grid_mapping[:, 1] >= 0) & (grid_mapping[:, 1] <= 199) & \
              (grid_mapping[:, 2] > 0)
valid_depth = (np.reshape(foreground_mask, -1) & (camera_z != 0)) != 0
valid_grid_w = grid_mapping[valid_pixel & valid_depth, :][:, 0].astype(np.int64)
valid_grid_h = grid_mapping[valid_pixel & valid_depth, :][:, 1].astype(np.int64)
height_map[valid_grid_h, valid_grid_w] = grid_mapping[valid_pixel & valid_depth, 2]

# find missing depth and project background depth into camera space
missing_depth = np.reshape((depth_img == 0) & (background_depth_img > 0), -1)
x, y = np.meshgrid(range(img.shape[1]), range(img.shape[0]))
camera_x = np.reshape((x - camera_intrinsics[0, 2]) * background_depth_img / camera_intrinsics[0, 0], -1)
camera_y = np.reshape((y - camera_intrinsics[1, 2]) * background_depth_img / camera_intrinsics[1, 1], -1)
camera_z = np.reshape(background_depth_img, -1)
missing_camera_points = np.stack([camera_x[missing_depth], camera_y[missing_depth], camera_z[missing_depth]], axis=1)
missing_world_points = np.matmul(missing_camera_points, camera_pose[0:3, 0:3].T) + camera_pose[0:3, 3].T

# get missing depth height map
missing_height_map = np.zeros((200, 300))
grid_origin = bin_middle_bottom - [0.3, 0.2, 0]
grid_mapping = np.stack([np.round((missing_world_points[:, 0] - grid_origin[0]) / args.voxel_size),
                         np.round((missing_world_points[:, 1] - grid_origin[1]) / args.voxel_size),
                         missing_world_points[:, 2] - grid_origin[2]], axis=1)
valid_pixel = (grid_mapping[:, 0] >= 0) & (grid_mapping[:, 0] <= 299) & \
              (grid_mapping[:, 1] >= 0) & (grid_mapping[:, 1] <= 199)
valid_grid_w = grid_mapping[valid_pixel, :][:, 0].astype(np.int64)
valid_grid_h = grid_mapping[valid_pixel, :][:, 1].astype(np.int64)
missing_height_map[valid_grid_h, valid_grid_w] = 1

noise_pixel = ~morphology.remove_small_objects(missing_height_map > 0, 50)
missing_height_map[noise_pixel] = 0

noise_pixel = ~morphology.remove_small_objects(height_map > 0, 50)
height_map[noise_pixel] = 0

# height cannot exceed 30cm above bottom of tote
height_map[height_map > 0.3] = 0.3
height_map[(height_map == 0) & (missing_height_map == 1)] = 0.03

height_map_color = np.flipud(height_map_color)
height_map = np.flipud(height_map)

depth_data = (np.pad(height_map,
                     ((12, 12), (10, 10)),
                     'constant',
                     constant_values=0) * 10000.0).astype(np.uint16)
color_data = (np.pad(height_map_color,
                     ((12, 12), (10, 10), (0, 0)),
                     'constant',
                     constant_values=0) * 255.0).astype(np.uint8)
cv2.imwrite('demo/raw-heightmap.color.png', color_data)
cv2.imwrite('demo/raw-heightmap.depth.png', depth_data)

