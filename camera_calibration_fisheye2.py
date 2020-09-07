import cv2
assert cv2.__version__[0] >= '3', 'The fisheye module requires opencv version >= 3.0.0'
import numpy as np
import glob

#######################################################
# Camera calibration with extracting frames of videos #
#######################################################

work_dir = "../00_data/camera_calibration"
frame_rate = 30     # intervals for extracting frames
print("======================================")
print("=== Working directory:", work_dir, "===")
print("=== Frame rate:", frame_rate, "===================")

CHECKERBOARD = (9, 6)    # rows, cols
subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW

objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)  # right, down

def undistort_video(img, K, D, DIM):
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    hori_concat = cv2.hconcat([img, undistorted_img])
    h, w = hori_concat.shape[:2]

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    scale = 0.3
    h, w = int(h * scale), int(w * scale)
    cv2.resizeWindow('image', (w, h))
    cv2.imshow("image", hori_concat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


_img_shape = None
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


vidcaps = glob.glob(work_dir + '/*.MP4')
for fname in vidcaps:
    print("======================================")
    print("The Video:", fname)
    vidcap = cv2.VideoCapture(fname)
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    while vidcap.isOpened():
        ret_vidcap, img = vidcap.read()
        if ret_vidcap and (int(vidcap.get(1) % frame_rate) == 0):
            if _img_shape == None:
                _img_shape = img.shape[:2]
            else:
                assert _img_shape == img.shape[:2], "All images must share the same size."
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
            # If found, add object points, image points (after refining them)
            print(int(vidcap.get(1)) / frame_rate, "th image", ret)
            if ret:
                objpoints.append(objp)
                cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
                imgpoints.append(corners)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
                cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                scale = 0.5
                h, w = img.shape[:2]
                h, w = int(h * scale), int(w * scale)
                cv2.resizeWindow('image', (w, h))
                cv2.imshow('image', img)
                k = cv2.waitKey(500)
        elif ret_vidcap is False and int(vidcap.get(1)) >= length:
            # print(ret_vidcap, int(vidcap.get(1)))
            break

    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))

    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

    rms, _, _, _, _ = \
        cv2.fisheye.calibrate(
            objpoints,
            imgpoints,
            gray.shape[::-1],
            K,
            D,
            rvecs,
            tvecs,
            calibration_flags,
            (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )
    print("Found " + str(N_OK) + " valid images for calibration")
    print("DIM=" + str(_img_shape[::-1]))
    print("K=np.array(" + str(K.tolist()) + ")")
    print("D=np.array(" + str(D.tolist()) + ")")

    vidcap = cv2.VideoCapture(fname)
    while vidcap.isOpened():
        ret_vidcap, img = vidcap.read()
        if ret_vidcap and (int(vidcap.get(1)) % frame_rate == 0):
            undistort_video(img, K, D, _img_shape[::-1])
        elif ret_vidcap is False and int(vidcap.get(1)) >= length:
            break
print("Done")
