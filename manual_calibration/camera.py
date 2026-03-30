import subprocess
subprocess.run([
    "v4l2-ctl",
    "--device=/dev/video4",
    "--set-fmt-video=width=160,height=120,pixelformat=Y16 "
])
device = "/dev/video4" #Change depending on device

info = subprocess.run(["v4l2-ctl", "--device", device, "--all"],
                      capture_output=True, text=True)
#print(info.stdout)

import cv2
import numpy as np

fourcc = cv2.VideoWriter.fourcc('Y','1','6',' ') # fourcc = four digit format codec
print('Format set')
cap = cv2.VideoCapture('/dev/video4', cv2.CAP_V4L2) # Stuck here
print(cap.getBackendName())
if not cap.isOpened():
    print("ERROR: Camera did not open")
    exit()
print('STarting capture')
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
cap.set(cv2.CAP_PROP_FOURCC, fourcc)
cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

print("CAP_PROP_FOURCC:", cap.get(cv2.CAP_PROP_FOURCC))
print("CAP_PROP_CONVERT_RGB:", cap.get(cv2.CAP_PROP_CONVERT_RGB))

folder = "/home/aayushi/Documents/Lepton/capture_march_20" #Change depending on device
folder2 = "/home/aayushi/Documents/Lepton/capture_march_20_proc" #Change depending on device
frame_buf = []
img_count=0 #Change this to the label you want the image to start frm to avoid overwriting images you already have in the target folder
#Starts from cap_{img_count}.tiff
while True:
    ret, frame = cap.read()
    if not ret:
        break

    data = frame.view(np.uint16)

    # print(frame.shape, frame.dtype)

    # For display only (do NOT use for quantitative analysis)
    data2 = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)
    data2 = np.uint8(data2)
    data2 = cv2.applyColorMap(data2, cv2.COLORMAP_INFERNO)
    disp = cv2.resize(data2, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    
    frame_buf.append(data)

    # Use CV2 to show image
    cv2.imshow('Lepton Radiometric (Y16)', disp)
    key = cv2.waitKey(1) & 0xFF

    # Press 'c' to capture image
    if key == ord('c'):
        #Change filename if needed
        filename = f"{folder}/cap_{img_count}.tiff"
        cv2.imwrite(filename, data) #Raw images in folder
        filename = f"{folder2}/cap_{img_count}.tiff"
        cv2.imwrite(filename, data2) #Normalized images in folder 2
        print(f"Saved: {filename}")
        img_count += 1

    # Press 'q' to quit
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


#UNCOMMENT THIS BLOCK IF YOU WANT TO SAVE ALL FRAMES CAPTURED SINCE CODE STARTED RUNNING
#PRESS q TO END THE LOOP

# num    = 0
# while len(frame_buf)>0:
#     cv2.imwrite(f'{folder}/cap_{num}.tiff', frame_buf.pop(0))
#     num += 1
