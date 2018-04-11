import pyrealsense2 as rs
import numpy as np
import cv2

# configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((5*6, 3), np.float32)
objp[:, :2] = np.mgrid[0:6, 0:5].T.reshape(-1, 2) * 14

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

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

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (6, 5), None)
        print ret
        print corners
        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (6, 5), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(2000)
            break
    cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print ret
    print mtx.shape
    print dist.shape
    print rvecs
    print tvecs
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    img = np.asanyarray(color_frame.get_data())
    # cv2.imwrite('uncalibrate.png', img)
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    # cv2.imwrite('calibresult.png', dst)
finally:
    pipeline.stop()

mean_error = 0
tot_error = 0
for i in xrange(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    tot_error += error

print "total error: ", mean_error/len(objpoints)
