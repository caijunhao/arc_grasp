import pyrealsense2 as rs

import numpy as np
import cv2

import urx

import argparse
import random
import copy
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='get coordinate pair')
parser.add_argument('--ip', required=True, type=str, help='ip address to the urx')  # 172.18.196.173
parser.add_argument('--port', required=True, type=str, help='port to the gripper')
parser.add_argument('--a', default=0.3, type=float, help='tool acceleration [m/s^2]')
parser.add_argument('--v', default=0.3, type=float, help='tool speed [m/s]')
parser.add_argument('--width', default=200, type=int, help='width of crop image')
parser.add_argument('--height', default=200, type=int, help='height of crop image')
parser.add_argument('--ratio', default=0.002, type=float, help='ratio ...')
parser.add_argument('--hover_distance', default=0.1538, type=float, help='hover distance before grasp')
parser.add_argument('--origin', default='origin.txt', type=str, help='file of origin point')
parser.add_argument('--camera_location', default='camera_location.txt', type=str,
                    help='file of camera location which is the initial pose of urx'
                         ' in order for the camera to take whole image')
parser.add_argument('--projection_matrix', default='projection_matrix.txt', type=str, help='file of projection matrix')
args = parser.parse_args()


def get_center_point(image, mtx, width, height):
    # find the contour of the object using image methods and get the center point of the object
    rectangle_list = []
    while len(rectangle_list) == 0:
        warp_image = cv2.warpPerspective(image, mtx, (width, height))
        ele = cv2.getStructuringElement(shape=0, ksize=(3, 3))
        edge_img = cv2.morphologyEx(warp_image, cv2.MORPH_GRADIENT, ele)

        _, edge_img = cv2.threshold(edge_img, thresh=20, maxval=255, type=cv2.THRESH_BINARY)
        binary_img = cv2.cvtColor(edge_img, cv2.COLOR_BGR2GRAY)
        binary_img = cv2.dilate(binary_img, ele)
        binary_img = cv2.erode(binary_img, np.ones((3, 3)))
        _, contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rectangle_list = [cv2.boundingRect(contour) for contour in contours if cv2.contourArea(contour) > 200]
        if len(rectangle_list) == 0:
            print 'no object found'
    rect = random.choice(rectangle_list)  # x,y,w,h for opencv coordinate system; y,x,w,h for image coordinate system
    w = np.random.randint(0, rect[2])
    h = np.random.randint(0, rect[3])
    center_points = [rect[0]+w, rect[1]+h]
    cv2.rectangle(warp_image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2)
    cv2.circle(warp_image, (center_points[0], center_points[1]), 1, color=(128, 128, 128), thickness=10)
    cv2.imshow('warp image', warp_image)
    cv2.waitKey(1)
    return center_points, rect


def get_label(center_point, theta, position, ratio=2):
    gripper_ratio = 85.0 / 255.0
    width = gripper_ratio * position / ratio
    c_h = center_point[1]
    c_w = center_point[0]
    h = [delta_h for delta_h in range(-5, 6)]
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
    # initialize urx
    rob = urx.Robot(args.ip)
    rob.set_tcp((0, 0, 0, 0, 0, 0))
    rob.set_payload(0.5, (0, 0, 0))
    gripper = urx.RobotiqGripper(port=args.port)
    gripper.activation_request()
    gripper.set_speed(0x80)
    gripper.set_force(0xFF)
    # initialize realsense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)
    align = rs.align(rs.stream.color)
    i = 7
    while i:
        _ = pipeline.wait_for_frames()
        i -= 7
    # load configuration
    camera_location = np.loadtxt(args.camera_location)
    rob.movel(camera_location, acc=args.a, vel=args.v)
    init_angle = rob.getj()[-1]
    origin = np.loadtxt(args.origin)
    mtx = np.loadtxt(args.projection_matrix)

    try:
        for i in range(60,90):
            print('--------i= %s' % i + '. Attempt to grasp without knowledge--------')
            # align the depth frame to color frame
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.first(rs.stream.color)
            depth_frame = aligned_frames.get_depth_frame()
            color_img = np.asanyarray(color_frame.get_data())
            depth_img = np.asanyarray(depth_frame.get_data())
            warp_color_img = cv2.warpPerspective(color_img, mtx, (args.width, args.height))
            warp_depth_img = cv2.warpPerspective(depth_img, mtx, (args.width, args.height))
            # get center point of object chosen among objects seen by the camera
            center_point, rect = get_center_point(color_img, mtx, args.width, args.height)

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
            pose3[2] -= args.hover_distance
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
                for j in range(2):
                    frames = pipeline.wait_for_frames()
                    aligned_frames = align.process(frames)
                    color_frame = aligned_frames.first(rs.stream.color)
                    depth_frame = aligned_frames.get_depth_frame()
                    color_img = np.asanyarray(color_frame.get_data())
                    depth_img = np.asanyarray(depth_frame.get_data())
                    warp_color_img = cv2.warpPerspective(color_img, mtx, (args.width, args.height))
                    warp_depth_img = cv2.warpPerspective(depth_img, mtx, (args.width, args.height))
                    cv2.imwrite('color/{:06d}_{:06d}.png'.format(i, j), color_img)
                    cv2.imwrite('depth/{:06d}_{:06d}.png'.format(i, j), depth_img)
                    cv2.imwrite('height_map_color/{:06d}_{:06d}.png'.format(i, j), warp_color_img)
                    cv2.imwrite('height_map_depth/{:06d}_{:06d}.png'.format(i, j), warp_depth_img)
                    _, rect = get_center_point(color_img, mtx, args.width, args.height)
                    points = get_label(center_point, -delta, 256-position)
                    label = draw_label(points, args.width, args.height)
                    cv2.imwrite('label/{:06d}_{:06d}.png'.format(i, j), label)
                    np.savetxt('label/{:06d}_{:06d}.good.txt'.format(i, j), points)
                    np.savetxt('label/{:06d}_{:06d}.rectangle.txt'.format(i, j), np.asanyarray(rect))

                    print('i=%s, j=%s' % (i, j) + ', all good grasp labels written.')
                    print('Now, let\'s grasp it again same to last pose to move to another place.')
                    rob.movel(episode[0], acc=args.a, vel=args.v)
                    rob.movel(episode[1], acc=args.a, vel=args.v)
                    rob.movel(episode[2], acc=args.a, vel=args.v)
                    gripper.gripper_close()


                    pose4 = copy.deepcopy(origin)
                    # Generate a new center_point randomly.
                    center_point = [np.random.randint(27, args.width - 27), np.random.randint(27, args.height - 27)]
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
                    pose6[2] -= args.hover_distance
                    rob.movel(pose6, acc=args.a, vel=args.v)
                    episode[2] = pose6

                    gripper.gripper_open()
                    print('OK,move to another place.Notice that we remember the route and angle unless j=1.')
                    rob.movel(episode[1], acc=args.a, vel=args.v)
                    rob.movel(camera_location, acc=args.a, vel=args.v)
            elif (status == 0 or status == 3) and position > 217:
                print('Grasp fails!')
                cv2.imwrite('color/{:06d}_{:06d}.png'.format(i, 0), color_img)
                cv2.imwrite('depth/{:06d}_{:06d}.png'.format(i, 0), depth_img)
                cv2.imwrite('height_map_color/{:06d}_{:06d}.png'.format(i, 0), warp_color_img)
                cv2.imwrite('height_map_depth/{:06d}_{:06d}.png'.format(i, 0), warp_depth_img)
                points = get_label(center_point, -delta, 256)
                label = draw_label(points, args.width, args.height, (0, 0, 255))
                cv2.imwrite('label/{:06d}_{:06d}.png'.format(i, 0), label)
                np.savetxt('label/{:06d}_{:06d}.bad.txt'.format(i, 0), points)
                np.savetxt('label/{:06d}_{:06d}.rectangle.txt'.format(i, 0), np.asanyarray(rect))
                print('i=%s' % i + ', all bad grasp labels written.')
                gripper.gripper_open()
                rob.movel(camera_location, acc=args.a, vel=args.v)
            else:
                gripper.gripper_open()
                rob.movel(camera_location, acc=args.a, vel=args.v)


    finally:
        rob.close()
        pipeline.stop()


if __name__ == '__main__':
    main()
