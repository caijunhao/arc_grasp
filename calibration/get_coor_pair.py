import pyrealsense2 as rs
import numpy as np
import cv2

import argparse

import urx

parser = argparse.ArgumentParser(description='get coordinate pair')
parser.add_argument('--ip', required=True, type=str, help='ip address to the urx')  # 172.18.196.173
parser.add_argument('--a', default=0.3, type=float, help='tool acceleration [m/s^2]')
parser.add_argument('--v', default=0.05, type=float, help='tool speed [m/s]')
parser.add_argument('--intrinsic', default='intrinsic.txt', type=str, help='path to camera intrinsic')
parser.add_argument('--dist', default='dist.txt', type=str, help='path to the camera distortion parameter')
args = parser.parse_args()


def main():
    rob = urx.Robot(args.ip)
    rob.set_tcp((0, 0, 0, 0, 0, 0))
    rob.set_payload(0.5, (0, 0, 0))
    pose = rob.getl()
    rob.movel(pose, acc=args.a, vel=args.v, wait=False)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            img = np.asanyarray(color_frame.get_data())
            cv2.imshow('RealSense', img)
            cv2.waitKey(1)

            markerCorners, markerIds, DetectionParameters = cv2.aruco.detectMarkers(img, aruco_dict)


    finally:
        pipeline.stop()


if __name__ == '__main__':
    main()

