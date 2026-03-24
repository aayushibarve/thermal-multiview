import cv2
import os
import numpy as np
import json

image_folder = '/home/aayushi/Documents/Lepton/capture_march_20_proc'
save_file = '/home/aayushi/semester-project/annotated_points_march_20.json'

if os.path.exists(save_file):
    with open(save_file, 'r') as f:
        annotated_data = json.load(f)
else:
    annotated_data = {}

current_points = []
current_image_name = None
image_list = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.tiff'))])
idx = 0

DISPLAY_SCALE = 6 

def show_image_with_points(img, points):
    h, w = img.shape[:2]
    img_resized = cv2.resize(img, (w*DISPLAY_SCALE, h*DISPLAY_SCALE), interpolation=cv2.INTER_NEAREST)

    for p in points:
        cv2.circle(img_resized, (int(p[0]*DISPLAY_SCALE), int(p[1]*DISPLAY_SCALE)), 5, (0, 0, 255), -1)

    cv2.imshow("Annotate", img_resized)

def mouse_callback(event, x, y, flags, param):
    global current_points
    if event == cv2.EVENT_LBUTTONDOWN:
        orig_x = int(x / DISPLAY_SCALE)
        orig_y = int(y / DISPLAY_SCALE)
        # print(f"Point added: ({orig_x}, {orig_y})")
        current_points.append([orig_x, orig_y])


while idx < len(image_list):
    current_image_name = image_list[idx]
    img_path = os.path.join(image_folder, current_image_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Cannot read {current_image_name}, skipping")
        idx += 1
        continue

    current_points = annotated_data.get(current_image_name, [])

    cv2.namedWindow("Annotate")
    cv2.setMouseCallback("Annotate", mouse_callback)

    while True:
        show_image_with_points(img, current_points)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('n'):  # Next image
            annotated_data[current_image_name] = current_points
            idx += 1
            break
        elif key == ord('p'):  # Previous image
            annotated_data[current_image_name] = current_points
            idx = max(idx-1, 0)
            break
        elif key == ord('r'):  # Remove last point
            if current_points:
                removed = current_points.pop()
                # print(f"Removed point: {removed}")
        elif key == ord('s'):  # Save progress
            annotated_data[current_image_name] = current_points
            with open(save_file, 'w') as f:
                json.dump(annotated_data, f, indent=2)
            print("Progress saved")
        elif key == 27:  # ESC to exit
            annotated_data[current_image_name] = current_points
            with open(save_file, 'w') as f:
                json.dump(annotated_data, f, indent=2)
            print("Exiting and saving progress")
            exit(0)

with open(save_file, 'w') as f:
    json.dump(annotated_data, f, indent=2)
cv2.destroyAllWindows()
