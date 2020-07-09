import numpy as np
import cv2
import glob
from orientation import get_orientation, restore_orientation

cols = 8
rows = 5

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((cols*rows, 3), np.float32)
objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1,2)
# objp[:, 0] = np.mgrid[rows-1:-1:-1, cols-1:-1:-1].T.reshape(-1, 2)[:, 1]
# objp[:, 1] = np.mgrid[0:rows, cols-1:-1:-1].T.reshape(-1, 2)[:, 0]
# # objp[:, 1] = np.mgrid[rows-1:-1:-1, 0:cols].T.reshape(-1, 2)[:, 0]
objp *= 30

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob('GoPro/*.jpg')
# images = glob.glob('GoPro3/*.jpg')
# images = glob.glob('images2/*.jpg')
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

# Undistortion
img = cv2.imread(images[0])
h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# # 1. Using cv2.undistort()
# # undistort
# dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
#
# # crop the image
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
# cv2.imwrite('calibresult_GoPro_undistort.png', dst)

# 2. Using remapping
# undistort
mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult_GoPro_remapping.png',dst)

# Reprojection error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

print("total error: ", mean_error/len(objpoints))
