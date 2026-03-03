# derived from https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
# and https://learnopencv.com/camera-calibration-using-opencv/
# all OpenCV source code copyrights respected

import numpy as np
import cv2 as cv
import glob

# prepare the process

# define the dimensions of chessboard in square=intersections (not real-world size of each square)
# the chessboard dimensions are the number of INTERSECTIONS BETWEEN squares, not the number
# of SQUARES between INTERSECTIONS. So a 10x7 chessboard has 11 squares across and 8 squares high
CHESSBOARD = (10,7)

# width (in pixels) of the user's preferred preview window size, for convenience in resizing data being displayed to fit
SCREENWIDTH = 1000
 
# define termination criteria for cornerSubPix refinement
# 0.001 is the accuracy (called Epsilon) and 30 is the max iteration count
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(9,6,0)
objp = np.zeros((CHESSBOARD[0]*CHESSBOARD[1],3), np.float32)
objp[:,:2] = np.mgrid[0:CHESSBOARD[0],0:CHESSBOARD[1]].T.reshape(-1,2)
 
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
 
images = glob.glob('*.jpg')
 
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
 
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    # we are passing the two optional args (which default to true):
    # AdaptiveThresh Use adaptive thresholding to convert the image to black and white, rather than a fixed threshold level (computed from the average image brightness). default true.
    # NormalizeImage Normalize the image gamma with cv.equalizeHist before applying fixed or adaptive thresholding. default true.
    ret, corners = cv.findChessboardCorners(gray, CHESSBOARD, cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE)
 
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
 
        # refine the corner location into sub-pixel decimal coordinates
        # by analyzing a window of surrounding pixels and computing gradients to
        # estimate where the intersectino fell between pixel centers
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # provide some feedback on progress
        print (fname + ":" + str(corners.shape[0]))
    else:
        print (fname + "Failed")

print ("Calibrating Camera.")
# this will fail if the set of calibration images aren't all the same dimensions (shape)
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

if ret < 2.0:
    print ("Calibration successful, RMS re-projection error: " + str(ret))
    print ("Intrinsic Camera Matrix")
    print (mtx)
    print ("Lens Distortion Coefficients")
    print (dist)
    print ("Rotation Vectors")
    print (rvecs)
    print ("Translation Vectors")
    print (tvecs)

