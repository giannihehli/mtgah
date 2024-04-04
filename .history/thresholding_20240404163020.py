# Importing libraries
import numpy as np
import cv2

# Importing user-defined modules
from undistortImage import undistort

def thresholding(image):
    pass

if __name__ == "__main__":
    
    # Define used camera
    camera = "sony_hs" # "sony", "sony_hs", "gopro1", "gopro2

    # Import calibration parameters
    K = np.loadtxt("calibration/" + camera + "/K.txt")  # calibration matrix[3x3]
    d = np.loadtxt("calibration/" + camera + "/d.txt")  # distortion coefficients[2x1]
    
    # Import image
    imgage = cv2.imread("H:/data/tests/sony/C0027 - Trim/C0027 - Trim_60", cv2.IMREAD_GRAYSCALE)

    # Undistort image
    img_undst = undistort(K, d, image)