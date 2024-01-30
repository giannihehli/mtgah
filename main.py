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

# Define used parameters
camera = "sony" # "sony", "gopro1", "gopro2
calib_path = "H:/data/calibration/" + camera + "/"
data_path = "H:/data/aruco/C0032 - Trim/"

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

for image_path in images:
    # Undistort images
    img_undst = undistort(K, d, data_path, image_path)
    # Detect markers
    marker = "DICT_4X4_50"
    img_det, corners, ids = detect(marker, img_undst)
