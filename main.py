# IMPORT PACKAGES AND MODULES
import sys
import os
import cv2
import glob
import numpy as np

# IMPORT USER-DEFINED MODULES
import videostoframes
import calibrateCamera
from undistortImage import undistort
from detectMarkers import detect
from locateCamera import locate
from filterImage import threshold

# Define used parameters
camera = "sony" # "sony", "gopro1", "gopro2
calib_path = "H:/data/calibration/" + camera + "/"
data_path = "H:/data/aruco/C0032 - Trim/"

# Define used basis
basis =  "rough" # "rough", "smooth"

# Load image and define reference pattern in clockwise order in world frame (3D) in [mm]
match basis:
    case "smooth":
        pattern = 0.001 * np.array([[12.5, 6.2, 0], [99.5, 8, 0], [98.6, 95, 0], [11.5, 93.2, 0],
                            [498.8, 8.2, 0], [586, 8.2, 0], [586.5, 95.4, 0], [499.2, 95, 0],
                            [499.5, 503, 0], [586.7, 503.9, 0], [585.4, 591.2, 0], [498, 590.3, 0],
                            [16.2, 503.1, 0], [103.3, 503.2, 0], [102.7, 590.5, 0], [15.6, 590.5, 0]]
                            )
    case "rough":
        pattern = 0.001 * np.array([[12.6, 12.8, 0], [98.5, 12.3, 0], [98.7, 98.5, 0], [13.1, 98.8, 0],
                            [499.2, 12.7, 0], [585.3, 12.8, 0], [585.1, 98.7, 0], [499.1, 98.6, 0],
                            [499.5, 501.2, 0], [585.5, 501.1, 0], [585.6, 587.1, 0], [499.6, 587.1, 0],
                            [12.7, 501.2, 0], [98.3, 501.2, 0], [98.4, 587.5, 0], [12.6, 587.3, 0]]
                            )


# Calibrate camera
if os.path.isfile("calibration/" + camera + "/K.txt"):
    print("Camera already calibrated. Loading calibration parameters...")
    K = np.loadtxt("calibration/" + camera + "/K.txt")  # calibration matrix[3x3]
    d = np.loadtxt("calibration/" + camera + "/d.txt")  # distortion coefficients[5x1]
else:
    print("Calibrating camera...")
    rep, K, d, rvec, tvec, X_W = calibrateCamera.calibrate(camera, view=True, check=True)

# Get frames from video
# videostoframes.videostoframes(data_path, "C0032 - Trim.MP4") # "/*.MP4" for all files

# Get input images
input_files = data_path + "/*.JPG"
images = glob.glob(input_files)

for img_path in images:
    # Load image
    image = cv2.imread(img_path)

    # Undistort images
    img_undst = undistort(K, d, image)

    # Detect markers
    marker = "DICT_4X4_50"
    img_det, corners, ids = detect(img_undst, marker)

    # Locate camera
    rvec, tvec = locate(corners, pattern, K, d)

    # Warp perspective

    # Threshold image
    img_thr = threshold(img_undst)

    # Measure distances