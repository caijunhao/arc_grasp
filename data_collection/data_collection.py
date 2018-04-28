import pyrealsense2 as rs

import numpy as np
import cv2

import urx

import argparse
import random
import copy
import os

parser = argparse.ArgumentParser(description='get coordinate pair')
parser.add_argument('--ip', required=True, type=str, help='ip address to the urx')  # 172.18.196.173
parser.add_argument('--port', required=True, type=str, help='port to the gripper')
parser.add_argument('--a', default=0.3, type=float, help='tool acceleration [m/s^2]')
parser.add_argument('--v', default=0.3, type=float, help='tool speed [m/s]')
parser.add_argument('--width', default=200, type=int, help='width of crop image')
parser.add_argument('--height', default=200, type=int, help='height of crop image')
parser.add_argument('--ratio', default=0.002, type=float, help='ratio ...')
parser.add_argument('--z_scale', default=1.34, type=float, help='z-scale factor')
parser.add_argument('--hover_distance', default=0.1538, type=float, help='hover distance before grasp')
parser.add_argument('--origin', default='origin.txt', type=str, help='file of origin point')
parser.add_argument('--camera_location', default='camera_location.txt', type=str,
                    help='file of camera location which is the initial pose of urx'
                         ' in order for the camera to take whole image')
parser.add_argument('--depth_dir', default='background_depth_height_map.png', type=str,
                    help='path to the background depth height map.')
parser.add_argument('--projection_matrix', default='projection_matrix.txt', type=str, help='file of projection matrix')
parser.add_argument('--output', required=True, type=str, help='directory to save data.')
args = parser.parse_args()


def get_center_point(image, depth, b_depth):
    # find the contour of the object using image methods and get the center point of the object
    image = image.copy()
    diff_depth = b_depth - depth - 20
    diff_depth[np.where(diff_depth < 0)] = 0
    diff_depth[np.where(diff_depth > 2000)] = 0
    diff_depth_copy = diff_depth.astype(np.uint8)

    ele = cv2.getStructuringElement(shape=0, ksize=(3, 3))
    rectangle_list = []
    while len(rectangle_list) == 0:
        edge_img = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, ele)

        _, edge_img = cv2.threshold(edge_img, thresh=20, maxval=255, type=cv2.THRESH_BINARY)
        binary_img = cv2.cvtColor(edge_img, cv2.COLOR_BGR2GRAY)
        binary_img = cv2.dilate(binary_img, ele)
        binary_img = cv2.erode(binary_img, np.ones((3, 3)))
        _, cons, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rectangle_list = [cv2.boundingRect(con) for con in cons if 200 < cv2.contourArea(con) < 10000]
        if len(rectangle_list) == 0 and np.sum(diff_depth_copy == 0) > 20000:
            print 'no object found in color image, try on depth'
            edge_img = cv2.morphologyEx(diff_depth_copy, cv2.MORPH_GRADIENT, ele)
            _, edge_img = cv2.threshold(edge_img, thresh=10, maxval=255, type=cv2.THRESH_BINARY)
            binary_img = cv2.dilate(binary_img, ele)
            binary_img = cv2.erode(binary_img, np.ones((3, 3)))
            _, cons, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rectangle_list = [cv2.boundingRect(con) for con in cons if 200 < cv2.contourArea(con) < 10000]
        if len(rectangle_list) == 0:
            print 'no object found both in color and depth, try again.'

    rect = random.choice(rectangle_list)  # x,y,w,h for opencv coordinate system; y,x,w,h for image coordinate system
    x_min = rect[1]
    y_min = rect[0]
    x_max = rect[1] + rect[3]
    y_max = rect[0] + rect[2]
    ids = np.stack(np.where(diff_depth > 0), axis=0).T
    ids = np.asanyarray([[idx[0], idx[1]] for idx in ids if x_min < idx[0] < x_max and y_min < idx[1] < y_max])
    # w = np.random.randint(rect[2] - rect[2]*0.7, rect[2]*0.7)
    # h = np.random.randint(rect[3] - rect[3]*0.7, rect[3]*0.7)
    # center_points = [rect[0]+w, rect[1]+h]
    center_points = ids[np.random.randint(0, ids.shape[0])].tolist()  # x, y for image coordinate system
    # block_size = 4
    # x, y = np.meshgrid(np.arange(center_points[0] - block_size / 2, center_points[0] + block_size / 2),
    #                    np.arange(center_points[1] - block_size / 2, center_points[1] + block_size / 2))
    # sample_points = np.stack((x.flatten(), y.flatten()), axis=1)
    # z = np.average(diff_depth[sample_points[:, 0], sample_points[:, 1]]) * 0.0001 - 0.01
    z = max(diff_depth[center_points[0], center_points[1]] * 0.0001 - 0.025, 0)
    cv2.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2)
    cv2.circle(image, (center_points[1], center_points[0]), 1, color=(128, 128, 128), thickness=10)
    cv2.imshow('warp image', image)
    cv2.waitKey(1)
    return center_points[::-1], rect, ids, z  # y, x for opencv coordinate system


def get_label(center_point, theta, position, ratio=2):
    # return points with order [x1, y1, x2, y2]
    gripper_ratio = 85.0 / 255.0
    width = gripper_ratio * position / ratio
    c_h = center_point[1]
    c_w = center_point[0]
    h = [delta_h for delta_h in range(-3, 4)]
    w = [-width / 2, width / 2]
    points = np.asanyarray([[hh, w[0], hh, w[1]] for hh in h])
    rotate_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                              [np.sin(theta), np.cos(theta)]])
    points[:, 0:2] = np.dot(rotate_matrix, points[:, 0:2].T).T
    points[:, 2:] = np.dot(rotate_matrix, points[:, 2:].T).T
    points = points + np.asanyarray([[c_h, c_w, c_h, c_w]])
    points = np.floor(points).astype(np.int)
    return points


def draw_label(points, width, height, color=(0, 255, 0)):
    label = np.ones((height, width, 3), dtype=np.uint8) * 255
    for point in points:
        pt1 = (point[1], point[0])
        pt2 = (point[3], point[2])
        cv2.line(label, pt1, pt2, color)
    return label


def main():
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    data_id = len(os.listdir(args.output))
    data_path = os.path.join(args.output, '{:04d}'.format(data_id))
    color_path = os.path.join(data_path, 'color')
    depth_path = os.path.join(data_path, 'depth')
    height_map_color_path = os.path.join(data_path, 'height_map_color')
    height_map_depth_path = os.path.join(data_path, 'height_map_depth')
    label_path = os.path.join(data_path, 'label')

    os.mkdir(data_path)
    os.mkdir(color_path)
    os.mkdir(depth_path)
    os.mkdir(height_map_color_path)
    os.mkdir(height_map_depth_path)
    os.mkdir(label_path)

    # initialize urx
    rob = urx.Robot(args.ip)
    rob.set_tcp((0, 0, 0, 0, 0, 0))
    rob.set_payload(0.5, (0, 0, 0))
    gripper = urx.RobotiqGripper(port=args.port)
    gripper.activation_request()
    gripper.set_speed(0x40)
    gripper.set_force(0x80)

    # initialize realsense
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline.start(config)
    align = rs.align(rs.stream.color)
    i = 57
    while i:
        _ = pipeline.wait_for_frames()
        i -= 1

    # load configuration
    camera_location = np.loadtxt(args.camera_location)
    rob.movel(camera_location, acc=args.a, vel=args.v)
    init_angle = rob.getj()[-1]
    origin = np.loadtxt(args.origin)
    mtx = np.loadtxt(args.projection_matrix)

    b_depth_height_map = cv2.imread(args.depth_dir, cv2.IMREAD_ANYDEPTH)
    cv2.imwrite(data_path+'/background_depth_height_map.png', b_depth_height_map)
    b_depth_height_map = b_depth_height_map.astype(np.float32)

    f = open(os.path.join(data_path, 'file_name.txt'), 'w')

    k = 0
    num_success = 0
    num_fail = 0

    try:
        for i in range(0, 10):
            print('--------i= %s' % i + '. Attempt to grasp without knowledge--------')
            # align the depth frame to color frame
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.first(rs.stream.color)
            depth_frame = aligned_frames.get_depth_frame()
            color_img = np.asanyarray(color_frame.get_data())
            depth_img = (np.asanyarray(depth_frame.get_data()) * args.z_scale).astype(np.uint16)
            warp_color_img = cv2.warpPerspective(color_img, mtx, (args.width, args.height))
            warp_depth_img = cv2.warpPerspective(depth_img, mtx, (args.width, args.height)).astype(np.float32)
            # get center point of object chosen among objects seen by the camera
            center_point, rect, object_pts, z = get_center_point(warp_color_img, warp_depth_img, b_depth_height_map)
            episode = []
            # approach to the target pose
            pose1 = copy.deepcopy(origin)
            pose1[0] += center_point[1] * args.ratio  # height
            pose1[1] += center_point[0] * args.ratio  # width
            pose1[2] += args.hover_distance
            rob.movel(pose1, acc=args.a, vel=args.v)
            episode.append(pose1)

            # rotate end effector angle
            joint_angle = rob.getj()
            delta = np.random.uniform(0, np.pi)
            angle = init_angle + delta
            joint_angle[-1] = angle
            rob.movej(joint_angle, args.a, args.v)
            pose2 = rob.getl()
            episode.append(pose2)

            # reach to target pose with end effector rotated
            pose3 = rob.getl()
            pose3[2] = pose3[2] - args.hover_distance + z
            rob.movel(pose3, acc=args.a, vel=args.v)
            episode.append(pose3)

            # close gripper and obtain gripper status
            print('Try grasp:')
            gripper.gripper_close()
            rob.movel(pose2, acc=args.a, vel=args.v)
            status = gripper.get_object_detection_status()
            position = gripper.get_gripper_pos()

            if (status == 1 or status == 2) and position < 217:
                print('Grasp success!')
                rob.movel(pose3, acc=args.a, vel=args.v)
                gripper.gripper_open()
                rob.movel(pose2, acc=args.a, vel=args.v)
                rob.movel(camera_location, acc=args.a, vel=args.v)
                for j in range(1):
                    frames = pipeline.wait_for_frames()
                    aligned_frames = align.process(frames)
                    color_frame = aligned_frames.first(rs.stream.color)
                    depth_frame = aligned_frames.get_depth_frame()
                    color_img = np.asanyarray(color_frame.get_data())
                    depth_img = (np.asanyarray(depth_frame.get_data()) * args.z_scale).astype(np.uint16)
                    warp_color_img = cv2.warpPerspective(color_img, mtx, (args.width, args.height))
                    warp_depth_img = cv2.warpPerspective(depth_img, mtx, (args.width, args.height))
                    cv2.imwrite(color_path+'/{:06d}.png'.format(k), color_img)
                    cv2.imwrite(depth_path+'/{:06d}.png'.format(k), depth_img)
                    cv2.imwrite(height_map_color_path+'/{:06d}.png'.format(k), warp_color_img)
                    cv2.imwrite(height_map_depth_path+'/{:06d}.png'.format(k), warp_depth_img)
                    _, rect, object_pts, _ = get_center_point(warp_color_img, warp_depth_img, b_depth_height_map)
                    points = get_label(center_point, -delta, 256-position)
                    label = draw_label(points, args.width, args.height)
                    cv2.imwrite(label_path+'/{:06d}.png'.format(k), label)
                    np.savetxt(label_path+'/{:06d}.good.txt'.format(k), points)
                    np.savetxt(label_path+'/{:06d}.rectangle.txt'.format(k), np.asanyarray(rect))
                    np.savetxt(label_path+'/{:06d}.object_points.txt'.format(k), np.asanyarray(object_pts))
                    f.write('{:06d}\n'.format(k))
                    k += 1

                    print('i=%s, j=%s' % (i, j) + ', all good grasp labels written.')
                    num_success = num_success+1
                    print('Now, let\'s grasp it again same to last pose to move to another place.')
                    rob.movel(episode[0], acc=args.a, vel=args.v)
                    rob.movel(episode[1], acc=args.a, vel=args.v)
                    rob.movel(episode[2], acc=args.a, vel=args.v)
                    gripper.gripper_close()

                    pose4 = copy.deepcopy(origin)
                    # generate a new center_point randomly.
                    center_point = [np.random.randint(57, args.width - 57), np.random.randint(57, args.height - 57)]
                    pose4[0] += center_point[1] * args.ratio
                    pose4[1] += center_point[0] * args.ratio
                    pose4[2] += args.hover_distance
                    rob.movel(pose4, acc=args.a, vel=args.v)
                    episode[0] = pose4

                    joint_angle = rob.getj()
                    delta = np.random.uniform(0, np.pi)
                    angle = init_angle + delta
                    joint_angle[-1] = angle
                    rob.movej(joint_angle, args.a, args.v)
                    pose5 = rob.getl()
                    episode[1] = pose5

                    pose6 = rob.getl()
                    pose6[2] = pose6[2] - args.hover_distance + z
                    rob.movel(pose6, acc=args.a, vel=args.v)
                    episode[2] = pose6

                    gripper.gripper_open()
                    print('OK,move to another place.Notice that we remember the route and angle unless j=1.')
                    rob.movel(episode[1], acc=args.a, vel=args.v)
                    rob.movel(camera_location, acc=args.a, vel=args.v)
            elif position > 217:
                print('Grasp fails!')
                cv2.imwrite(color_path + '/{:06d}.png'.format(k), color_img)
                cv2.imwrite(depth_path + '/{:06d}.png'.format(k), depth_img)
                cv2.imwrite(height_map_color_path + '/{:06d}.png'.format(k), warp_color_img)
                cv2.imwrite(height_map_depth_path + '/{:06d}.png'.format(k), warp_depth_img)
                points = get_label(center_point, -delta, 256)
                label = draw_label(points, args.width, args.height, (0, 0, 255))
                cv2.imwrite(label_path + '/{:06d}.png'.format(k), label)
                np.savetxt(label_path + '/{:06d}.bad.txt'.format(k), points)
                np.savetxt(label_path + '/{:06d}.rectangle.txt'.format(k), np.asanyarray(rect))
                np.savetxt(label_path + '/{:06d}.object_points.txt'.format(k), np.asanyarray(object_pts))
                f.write('{:06d}\n'.format(k))

                k += 1
                print('i=%s' % i + ', all bad grasp labels written.')
                num_fail = num_fail+1

                gripper.gripper_open()
                rob.movel(camera_location, acc=args.a, vel=args.v)
            else:
                gripper.gripper_open()
                rob.movel(camera_location, acc=args.a, vel=args.v)

    finally:
        rob.close()
        pipeline.stop()
        f.close()
        print("Success: " + str(num_success) + ", fail: " + str(num_fail))


if __name__ == '__main__':
    main()
