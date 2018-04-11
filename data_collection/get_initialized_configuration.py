import pyrealsense2 as rs

import numpy as np
import cv2

import argparse

import urx

parser = argparse.ArgumentParser(description='get coordinate pair')
parser.add_argument('--ip', required=True, type=str, help='ip address to the urx')  # 172.18.196.173
parser.add_argument('--a', default=0.3, type=float, help='tool acceleration [m/s^2]')
parser.add_argument('--v', default=0.05, type=float, help='tool speed [m/s]')
parser.add_argument('--width', default=200, type=int, help='width of crop image')
parser.add_argument('--height', default=200, type=int, help='height of crop image')
args = parser.parse_args()


class Corner(object):
    def __init__(self):
        self.points = []

    def callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            p = [x, y]
            self.points.append(p)


def main():
    # get background image, including color and depth image
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)
    align = rs.align(rs.stream.color)
    # do nothing, just omit first 17 frames
    i = 17
    while i:
        _ = pipeline.wait_for_frames()
        i -= 1
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.first(rs.stream.color)
    depth_frame = aligned_frames.get_depth_frame()
    color_img = np.asanyarray(color_frame.get_data())
    depth_img = np.asanyarray(depth_frame.get_data())
    cv2.imwrite('background_color.png', color_img)
    cv2.imwrite('background_depth.png', depth_img)

    # get camera location, because sr300 is mounted on ur, we only need to return the current pose
    rob = urx.Robot(args.ip)
    rob.set_tcp((0, 0, 0, 0, 0, 0))
    rob.set_payload(0.5, (0, 0, 0))
    pose = rob.getl()
    np.savetxt('camera_location.txt', np.asanyarray(pose))

    # get projection matrix
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.first(rs.stream.color)
    color_img = np.asanyarray(color_frame.get_data())
    cv2.imshow('raw image', color_img)
    cv2.waitKey(1)
    p = Corner()
    # click corners in clockwise direction
    cv2.setMouseCallback('raw image', p.callback)
    cv2.waitKey(0)
    cv2.setMouseCallback('raw image', p.callback)
    cv2.waitKey(0)
    cv2.setMouseCallback('raw image', p.callback)
    cv2.waitKey(0)
    cv2.setMouseCallback('raw image', p.callback)
    cv2.waitKey(0)
    src = np.float32(p.points)
    dst = np.float32([[0, 0],
                      [args.width - 1, 0],
                      [args.width - 1, args.height - 1],
                      [0, args.height - 1]])
    mtx = cv2.getPerspectiveTransform(src, dst)
    np.savetxt('projection_matrix.txt', mtx)
    cv2.destroyWindow('raw image')
    pipeline.stop()
    rob.close()

    # get orientation


if __name__ == '__main__':
    main()
