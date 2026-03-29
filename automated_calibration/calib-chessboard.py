import cv2
import numpy as np
import glob
import os

IMAGE_FOLDER   = "/home/aayushi/Documents/Lepton/chessboard"
IMAGE_GLOB     = "*.tiff"
SQUARE_SIZE_MM = 25.0
BOARD_COLS     = 4
BOARD_ROWS     = 4
CLAHE_CLIP     = 4.0
CLAHE_TILE     = (8, 8)
BG_PERCENTILE  = 20
SAVE_FILE      = "camera_params_chessboard.npz"

PATTERN_SIZE  = (BOARD_COLS, BOARD_ROWS)
SB_FLAGS      = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
CLASSIC_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
clahe_op      = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)

objp = np.zeros((BOARD_ROWS * BOARD_COLS, 3), np.float32)
objp[:, :2] = np.mgrid[0:BOARD_COLS, 0:BOARD_ROWS].T.reshape(-1, 2)
objp *= SQUARE_SIZE_MM

def process(path):
    raw  = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY) if raw.ndim == 3 else raw.copy()
    gray8 = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    enhanced = clahe_op.apply(gray8)

    bg_thresh  = int(np.percentile(enhanced, BG_PERCENTILE))
    board_mask = enhanced > bg_thresh
    board_only = np.zeros_like(enhanced)
    board_only[board_mask] = enhanced[board_mask]
    otsu_val, _ = cv2.threshold(board_only, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    threelevel = np.zeros_like(enhanced)
    threelevel[board_mask] = 100
    threelevel[board_mask & (enhanced > int(otsu_val))] = 220

    return enhanced, threelevel

def detect(enhanced, threelevel):
    candidates = [
        enhanced,
        threelevel,
        cv2.GaussianBlur(enhanced, (5, 5), 0),
        cv2.filter2D(enhanced, -1, np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])),
    ]
    for candidate in candidates:
        found, corners = cv2.findChessboardCornersSB(candidate, PATTERN_SIZE, SB_FLAGS)
        if found:
            return True, corners
        found, corners = cv2.findChessboardCorners(candidate, PATTERN_SIZE, CLASSIC_FLAGS)
        if found:
            crit    = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(candidate, corners, (11, 11), (-1, -1), crit)
            return True, corners
    return False, None

image_paths = sorted(glob.glob(os.path.join(IMAGE_FOLDER, IMAGE_GLOB)))

obj_points  = []
img_points  = []
image_size  = None

for path in image_paths:
    enhanced, threelevel = process(path)
    found, corners = detect(enhanced, threelevel)
    if found:
        obj_points.append(objp)
        img_points.append(corners)
        if image_size is None:
            image_size = (enhanced.shape[1], enhanced.shape[0])

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, image_size, None, None)

np.savez(SAVE_FILE, ret=ret, K=K, dist=dist, rvecs=rvecs, tvecs=tvecs)

print(f"Images used       : {len(obj_points)}/{len(image_paths)}")
print(f"RMS error         : {ret:.4f} px")
print(f"Camera matrix K   :\n{np.array2string(K, precision=4, suppress_small=True)}")
print(f"Distortion coeffs : {np.array2string(dist, precision=6, suppress_small=True)}")
print(f"Saved to          : {os.path.abspath(SAVE_FILE)}")