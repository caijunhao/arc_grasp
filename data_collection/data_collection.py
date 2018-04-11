#-*- coding: utf-8 -*

import pyrealsense2 as rs

import numpy as np
import cv2

import urx

import argparse
import random
import copy

parser = argparse.ArgumentParser(description='get coordinate pair')
parser.add_argument('--ip', required=True, type=str, help='ip address to the urx')  # 172.18.196.173
parser.add_argument('--port', required=True, type=str, help='port to the gripper')
parser.add_argument('--a', default=0.3, type=float, help='tool acceleration [m/s^2]')
parser.add_argument('--v', default=0.3, type=float, help='tool speed [m/s]')
parser.add_argument('--width', default=200, type=int, help='width of crop image')
parser.add_argument('--height', default=200, type=int, help='height of crop image')
parser.add_argument('--ratio', default=0.002, type=float, help='ratio ...')
parser.add_argument('--hover_distance', default=0.1536, type=float, help='hover distance before grasp')
parser.add_argument('--origin', default='origin.txt', type=str, help='file of origin point')
parser.add_argument('--camera_location', default='camera_location.txt', type=str,
                    help='file of camera location which is the initial pose of urx'
                         ' in order for the camera to take whole image')
parser.add_argument('--projection_matrix', default='projection_matrix.txt', type=str, help='file of projection matrix')
args = parser.parse_args()


def get_center_point(image, mtx, width, height):
    # find the contour of the object using image methods and get the center point of the object
    warp_image = cv2.warpPerspective(image, mtx, (width, height))
    ele = cv2.getStructuringElement(shape=0, ksize=(3, 3))
    edge_img = cv2.morphologyEx(warp_image, cv2.MORPH_GRADIENT, ele)

    _, edge_img = cv2.threshold(edge_img, thresh=20, maxval=255, type=cv2.THRESH_BINARY)
    binary_img = cv2.cvtColor(edge_img, cv2.COLOR_BGR2GRAY)
    binary_img = cv2.dilate(binary_img, ele)
    binary_img = cv2.erode(binary_img, np.ones((3, 3)))
    _, contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 将轮廓信息转换成(x, y)坐标，并加上矩形的高度和宽度
    rectangle_list = [cv2.boundingRect(contour) for contour in contours if cv2.contourArea(contour) > 200]
    center_points = [[bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2] for bbox in rectangle_list]
    return random.choice(center_points)


def main():
    # initialize urx
    rob = urx.Robot(args.ip)
    rob.set_tcp((0, 0, 0, 0, 0, 0))
    rob.set_payload(0.5, (0, 0, 0))
    gripper = urx.RobotiqGripper(port=args.port)
    gripper.activation_request()
    gripper.set_speed(0xFF)
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

    origin = np.loadtxt(args.origin)
    mtx = np.loadtxt(args.projection_matrix)
    try:
        while True:
            # align the depth frame to color frame
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.first(rs.stream.color)
            depth_frame = aligned_frames.get_depth_frame()
            color_img = np.asanyarray(color_frame.get_data())
            depth_img = np.asanyarray(depth_frame.get_data())
            # get center point of object chosen among objects seen by the camera
            center_point = get_center_point(color_img, mtx, args.width, args.height)

            episode = []
            # approach to the target pose
            pose1 = copy.deepcopy(origin)
            pose1[0] += center_point[1] * args.ratio
            pose1[1] += center_point[0] * args.ratio
            pose1[2] += args.hover_distance
            rob.movel(pose1, acc=args.a, vel=args.v)
            episode.append(pose1)

            # rotate end effector angle
            joint_angle = rob.getj()
            joint_angle[-1] = np.random.uniform(0, np.pi)
            rob.movej(joint_angle, args.a, args.v)
            pose2 = rob.getl()
            episode.append(pose2)

            # reach to target pose with end effector rotated
            pose3 = rob.getl()
            pose3[2] -= args.hover_distance
            rob.movel(pose3, acc=args.a, vel=args.v)
            episode.append(pose3)

            # close gripper and obtain gripper status
            gripper.gripper_close()
            rob.movel(pose2, acc=args.a, vel=args.v)
            status = gripper.get_object_detection_status()

            while status == 1 or status == 2:
                rob.movel(episode[2], acc=args.a, vel=args.v)
                gripper.gripper_open()
                rob.movel(episode[1], acc=args.a, vel=args.v)
                rob.movel(camera_location, acc=args.a, vel=args.v)
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                color_frame = aligned_frames.first(rs.stream.color)
                depth_frame = aligned_frames.get_depth_frame()
                color_img = np.asanyarray(color_frame.get_data())
                depth_img = np.asanyarray(depth_frame.get_data())
                rob.movel(episode[0], acc=args.a, vel=args.v)
                rob.movel(episode[1], acc=args.a, vel=args.v)
                rob.movel(episode[2], acc=args.a, vel=args.v)
                gripper.gripper_close()

                pose4 = copy.deepcopy(origin)
                pose4[0] += np.random.randint(20, args.height - 20) * args.ratio
                pose4[1] += np.random.randint(20, args.width - 20) * args.ratio
                pose4[2] += args.hover_distance
                rob.movel(pose4, acc=args.a, vel=args.v)
                episode[0] = pose4

                joint_angle = rob.getj()
                joint_angle[-1] = np.random.uniform(0, np.pi)
                rob.movej(joint_angle, args.a, args.v)
                pose5 = rob.getl()
                episode[1] = pose5

                pose6 = rob.getl()
                pose6[2] -= args.hover_distance
                rob.movel(pose6, acc=args.a, vel=args.v)
                episode[2] = pose6

                rob.movel(pose5, acc=args.a, vel=args.v)
                status = gripper.get_object_detection_status()

                # # move to a new position
                # pose4 = copy.deepcopy(origin)
                # pose4[0] += np.random.randint(20, args.height - 20) * args.ratio
                # pose4[1] += np.random.randint(20, args.width - 20) * args.ratio
                # pose4[2] += args.hover_distance
                # rob.movel(pose4, acc=args.a, vel=args.v)
                # # randomly rotate end effector angle
                # joint_angle = rob.getj()
                # joint_angle[-1] = np.random.uniform(0, np.pi)
                # rob.movej(joint_angle, args.a, args.v)
                # pose4 = rob.getl()
                # # reach to new position
                # pose5 = rob.getl()
                # pose5[2] -= args.hover_distance
                # rob.movel(pose5, acc=args.a, vel=args.v)
                # gripper.gripper_open()
                # rob.movel(camera_location, acc=args.a, vel=args.v)

            gripper.gripper_open()
            rob.movel(camera_location, acc=args.a, vel=args.v)

    finally:
        rob.close()
        pipeline.stop()


if __name__ == '__main__':
    main()
