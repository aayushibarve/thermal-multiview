import cv2
import numpy as np
import json
import os

save_file = '/home/aayushi/semester-project/annotated_points.json'  
image_folder = '/home/aayushi/Documents/Lepton/capture_march_11'
calib_output_file = '/home/aayushi/semester-project/camera_params.npz'

rows, cols = 4, 4
spacing = 200/7  #in mm

with open(save_file, 'r') as f:
    annotated_data = json.load(f)

objp = np.zeros((rows*cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * spacing

obj_points = []  
img_points = []  


for img_name, points in annotated_data.items():
    if len(points) != rows*cols:
        print(f"Skipping {img_name}: expected {rows*cols} points, got {len(points)}")
        continue

    obj_points.append(objp)
    img_points.append(np.array(points, dtype=np.float32))


example_img_path = os.path.join(image_folder, next(iter(annotated_data.keys())))
example_img = cv2.imread(example_img_path)
h, w = example_img.shape[:2]


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, (w, h), None, None
)

print("Camera matrix:\n", mtx)
print("\nDistortion coefficients:\n", dist.ravel())
print("\nReprojection error:", ret)

np.savez(calib_output_file, camera_matrix=mtx, dist_coeffs=dist, reprojection_error=ret, rvecs=rvecs, tvecs=tvecs)
print(f"\nCalibration parameters saved to {calib_output_file}")

# For loading the data
# data = np.load('/home/semester-project/camera_params.npz')
# mtx = data['camera_matrix']
# dist = data['dist_coeffs']
# print("Loaded camera matrix:\n", mtx)
