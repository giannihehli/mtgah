# Importing libraries
import numpy as np
import cv2

# Importing user-defined modules
from undistortImage import undistort

def thresholding(image):
    img_blur = cv2.GaussianBlur(image, (5, 5), 0)

    cv2.imshow("img_blur", img_blur)

    _, img_thr = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cv2.imshow("img_thr", img_thr)

    return image

if __name__ == "__main__":
    
    # Define used camera
    camera = "sony_hs" # "sony", "sony_hs", "gopro1", "gopro2

    # Import calibration parameters
    K = np.loadtxt("calibration/" + camera + "/K.txt")  # calibration matrix[3x3]
    d = np.loadtxt("calibration/" + camera + "/d.txt")  # distortion coefficients[2x1]
    
    # Import image
    image = cv2.imread("H:/data/tests/sony/C0027 - Trim/C0027 - Trim_60", cv2.IMREAD_GRAYSCALE)

    cv2.imshow("image", image)

    # Undistort image
    img_undst = undistort(K, d, image)

    # Threshold image
    img_thr = thresholding(img_undst)

