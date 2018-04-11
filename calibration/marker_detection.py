import pyrealsense2 as rs
import numpy as np
import cv2

intrinsic = np.loadtxt('intrinsic.txt')
dist = np.expand_dims(np.loadtxt('dist.txt'), axis=0)


aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
# marker = cv2.aruco.drawMarker(aruco_dict, 177, 200)
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)
align = rs.align(rs.stream.color)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.first(rs.stream.color)
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue
        img = np.asanyarray(color_frame.get_data())
        depth_img = np.asanyarray(depth_frame.get_data())
        cv2.imshow('img', img)
        cv2.waitKey(1)
        markerCorners, markerIds, DetectionParameters = cv2.aruco.detectMarkers(img, aruco_dict)
        detected_img = cv2.aruco.drawDetectedMarkers(img, markerCorners, markerIds)
        cv2.imshow('marker', detected_img)
        cv2.waitKey(1)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 0.05, intrinsic, dist)
        if rvecs is not None:
            axis_img = cv2.aruco.drawAxis(detected_img, intrinsic, dist, rvecs, tvecs, 0.1)
            mass_point = np.floor(np.squeeze(np.average(markerCorners[0], axis=1)))
            cv2.circle(axis_img, (mass_point[0], mass_point[1]), 1, color=(128, 128, 128), thickness=10)
            cv2.imshow('axis', axis_img)
            cv2.waitKey(1)
            print mass_point
            print depth_img[int(mass_point[0]), int(mass_point[1])]
finally:
    pipeline.stop()
