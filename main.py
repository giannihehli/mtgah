# IMPORT PACKAGES AND MODULES
import sys, os, cv2
import numpy as np

# IMPORT USER-DEFINED MODULES
import videostoframes
import calibration


# Define used parameters
camera = "sony"
data_path = "H:/data/calibration/" + camera + "/"

# Calibrate camera
calibration.calibrate(camera, True)