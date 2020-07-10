import numpy as np
import cv2
import glob
from orientation import get_orientation, restore_orientation
import os

cols = 8
rows = 5

work_dir = "GoPro"

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((cols*rows, 3), np.float32)
objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
objp *= 30

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob(work_dir + '/*.jpg')
for fname in images:
    img = cv2.imread(fname)

    ############################################
    # 1. Extract EXIF data from a image
    orientation = get_orientation(fname)

    # 2. Restore the image based on orientation information
    restored_image = restore_orientation(img, orientation)
    ############################################

    gray = cv2.cvtColor(restored_image, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

    # If found, add object points, image points (after refining them)
    if ret is True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        img = cv2.drawChessboardCorners(restored_image, (cols, rows), corners2, ret)

        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        scale = 0.2
        h, w = restored_image.shape[:2]
        h, w = int(h * scale), int(w * scale)
        cv2.resizeWindow('image', (w, h))
        cv2.imshow('image', restored_image)
        k = cv2.waitKey(500)
        # if k == 27:  # esc key
        #     cv2.destroyAllWindows()
        print(fname)
    else:
        print("No corners in", fname)
cv2.destroyAllWindows()

# Calibration
ret_cal, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Check whether there is an undistortion directory
if os.path.isdir(work_dir + '/undistort_perspective'):
    pass
else:
    os.mkdir(work_dir + '/undistort_perspective')

# Undistortion
for fname in images:
    img_name = fname.split("\\")[1]
    img = cv2.imread(fname)

    ############################################
    # 1. Extract EXIF data from a image
    orientation = get_orientation(fname)

    # 2. Restore the image based on orientation information
    restored_image = restore_orientation(img, orientation)
    ############################################

    h, w = restored_image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # 1. Using cv2.undistort()
    # undistort
    dst = cv2.undistort(restored_image, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w1, h1 = roi
    dst = dst[y:y+h1, x:x+w1]
    cv2.imwrite(work_dir + '/undistort/' + img_name.split(".")[0] + '_undistort.png', dst)

    # 2. Using remapping
    # undistort
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
    dst = cv2.remap(restored_image, mapx, mapy, cv2.INTER_LINEAR)

    # crop the image
    x, y, w2, h2 = roi
    dst = dst[y:y+h2, x:x+w2]
    cv2.imwrite(work_dir + '/undistort/' + img_name.split(".")[0] + '_remapping.png', dst)

# Reprojection error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

print("total error: ", mean_error/len(objpoints))
