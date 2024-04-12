# Importing libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Importing user-defined modules
from undistortImage import undistort

def threshold(image):
    # Convert image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    img_gb = cv2.GaussianBlur(image, (5, 5), 0)
    # Apply bilateral filter
    img_bf = cv2.bilateralFilter(image,9,75,75)

    _, img_thr_gb = cv2.threshold(img_gb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, img_thr_bf = cv2.threshold(img_bf, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return img_thr_gb, img_thr_bf

if __name__ == "__main__":
    
    # Define used camera
    camera = "sony_hs" # "sony", "sony_hs", "gopro1", "gopro2

    # Import calibration parameters
    K = np.loadtxt("calibration/" + camera + "/K.txt")  # calibration matrix[3x3]
    d = np.loadtxt("calibration/" + camera + "/d.txt")  # distortion coefficients[2x1]
    
    # Import image
    image = cv2.imread("H:/data/tests/sony/C0027 - Trim/C0027 - Trim_60.jpg", cv2.IMREAD_GRAYSCALE)

    cv2.imshow("image", image)
    cv2.waitKey(0)

    # Undistort image
    img_undst = undistort(K, d, image)

    # Threshold image
    img_thr_gb, img_thr_bf = threshold(img_undst)
    
    f, axarr = plt.subplots(2, 2,sharex=True, sharey=True)
    axarr[0,0].imshow(img_gb)
    axarr[0,1].imshow(img_bf)
    axarr[1,0].imshow(img_thr_gb)
    axarr[1,1].imshow(img_thr_bf)
    plt.show()

