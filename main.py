# IMPORT PACKAGES AND MODULES
import sys
import os
import cv2
import glob
import numpy as np

# IMPORT USER-DEFINED MODULES
from videostoframes import extract
import calibrateCamera
from undistortImage import undistort
from detectMarkers import detect
from locateCamera import locate
from warpPerspective import warp
from filterImage import threshold
from measureDistance import measure

# Define used parameters
camera = "sony" # "sony", "gopro1", "gopro2
calib_path = "H:/data/calibration/" + camera + "/"
data_path = "H:/data/tests/sony_hs/"

# Calibrate camera
if os.path.isfile("calibration/" + camera + "/K.txt"):
    print("Camera already calibrated. Loading calibration parameters...")
    K = np.loadtxt("calibration/" + camera + "/K.txt")  # calibration matrix[3x3]
    d = np.loadtxt("calibration/" + camera + "/d.txt")  # distortion coefficients[5x1]
else:
    print("Calibrating camera...")
    rep, K, d, rvec, tvec, X_W = calibrateCamera.calibrate(camera, view=True, check=True)

# Extract frames from all videos
extract(data_path, "*.MP4")

for vid_path in glob.glob(data_path + "*.MP4"):
    # Get video name
    vid = os.path.splitext(os.path.basename(vid_path))[0]

    # Define used parameters according to video name
    layout = vid.split('_')[0]
    basis = vid.split('_')[1]
    diameter = vid.split('_')[2]
    height = vid.split('_')[3]

    print(f'Processing frames: {vid} with layout: {layout}, basis: {basis}, diameter: {diameter}, height: {height}')

    # Define reference pattern in clockwise order in world frame (3D) in [0.1mm]
    match basis:
        case "sm":
            pattern = 10 * np.array([[12.5, 6.2, 0], [99.5, 8, 0], [98.6, 95, 0], [11.5, 93.2, 0],
                                [498.8, 8.2, 0], [586, 8.2, 0], [586.5, 95.4, 0], [499.2, 95, 0],
                                [499.5, 503, 0], [586.7, 503.9, 0], [585.4, 591.2, 0], [498, 590.3, 0],
                                [16.2, 503.1, 0], [103.3, 503.2, 0], [102.7, 590.5, 0], [15.6, 590.5, 0]]
                                )
        case "r8":
            pattern = 10 * np.array([[12.6, 12.8, 0], [98.5, 12.3, 0], [98.7, 98.5, 0], [13.1, 98.8, 0],
                                [499.2, 12.7, 0], [585.3, 12.8, 0], [585.1, 98.7, 0], [499.1, 98.6, 0],
                                [499.5, 501.2, 0], [585.5, 501.1, 0], [585.6, 587.1, 0], [499.6, 587.1, 0],
                                [12.7, 501.2, 0], [98.3, 501.2, 0], [98.4, 587.5, 0], [12.6, 587.3, 0]]
                                )


    # Get input images
    input_files = data_path + vid + "/*.JPG"
    images = glob.glob(input_files)

    # Initialize lists for measured distances
    horizontal_diameter = []
    vertical_radius = []

    for img_path in images:
        # Load image
        print("Load image: ", img_path)
        image = cv2.imread(img_path)

        # Undistort images
        img_undst = undistort(K, d, image)
#        cv2.imshow("image_undist", cv2.resize(img_undst, (1920, 1080)))
#        cv2.waitKey(0)

        # Detect markers
        marker = "DICT_4X4_50"
        img_det, corners, ids = detect(img_undst, marker)
#        cv2.imshow("image_det", cv2.resize(img_det, (1920, 1080)))
#        cv2.waitKey(0)

        # Warp perspective
        img_warp, M = warp(corners, pattern, img_undst)
#        cv2.imshow("image_warp", cv2.resize(img_warp, (1080, 1080)))
#        cv2.waitKey(0)

        # Threshold image
        img_thr, _ = threshold(img_warp)
#        cv2.imshow("image_thr", cv2.resize(img_thr, (1080, 1080)))
#        cv2.waitKey(0)

        # Measure distances
        d_horizontal, r_vertical, img_mes = measure(img_warp, img_thr)
#         cv2.imshow("image_mes", cv2.resize(img_mes, (1080, 1080)))
#        cv2.waitKey(0)

        horizontal_diameter.append(d_horizontal)
        vertical_radius.append(r_vertical)
        
