import pyrealsense2 as rs
import numpy as np
import cv2

# configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue
        img = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cv2.imshow('RealSense', img)
        cv2.waitKey(1)

        ret, corners = cv2.findChessboardCorners(gray, (3, 3), corners=None)  # , flags=cv2.CALIB_CB_FAST_CHECK
        print ret
        print corners
        if ret:
            print corners.shape
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            img = cv2.drawChessboardCorners(img, (3, 3), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(20)
finally:
    pipeline.stop()
