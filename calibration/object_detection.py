import pyrealsense2 as rs

import numpy as np
import cv2

import argparse
import copy

parser = argparse.ArgumentParser(description='object detection')
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
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)
    align = rs.align(rs.stream.color)

    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
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
                      [args.width-1, 0],
                      [args.width - 1, args.height - 1],
                      [0, args.height-1]])
    mtx = cv2.getPerspectiveTransform(src, dst)
    cv2.destroyWindow('raw image')
    
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.first(rs.stream.color)
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue
        color_img = np.asanyarray(color_frame.get_data())
        depth_img = np.asanyarray(depth_frame.get_data())
        cv2.imshow('raw image', color_img)
        cv2.waitKey(1)
        warp_color_img = cv2.warpPerspective(color_img, mtx, (args.width, args.height))
        warp_depth_img = cv2.warpPerspective(depth_img, mtx, (args.width, args.height))
        cv2.imshow('warp image',
                   np.hstack((warp_color_img,
                              cv2.applyColorMap(cv2.convertScaleAbs(warp_depth_img, alpha=0.3), cv2.COLORMAP_JET))))
        cv2.waitKey(1)
        ele = cv2.getStructuringElement(shape=0, ksize=(3, 3))
        edge_img = cv2.morphologyEx(warp_color_img, cv2.MORPH_GRADIENT, ele)
        _, edge_img = cv2.threshold(edge_img, thresh=20, maxval=255, type=cv2.THRESH_BINARY)
        binary_img = cv2.cvtColor(edge_img, cv2.COLOR_BGR2GRAY)
        binary_img = cv2.dilate(binary_img, ele)
        binary_img = cv2.erode(binary_img, np.ones((3, 3)))
        _, contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rectangle_list = [cv2.boundingRect(contour) for contour in contours if cv2.contourArea(contour) > 200]
        # center_point = [[bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2] for bbox in rectangle_list]
        # np.savetxt('center.txt', np.asanyarray(center_point))
        warp_color_img_copy = copy.deepcopy(warp_color_img)
        [cv2.rectangle(warp_color_img_copy,
                       (bbox[0], bbox[1]),
                       (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                       (0, 255, 0), 2) for bbox in rectangle_list]
        cv2.imshow('bbox', warp_color_img_copy)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
