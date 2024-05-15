# Importing libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Importing user-defined modules
from undistortImage import undistort
from detectMarkers import detect
from warpPerspective import warp

def threshold(image, kernel_size, sigma_color, sigma_space, filter_threshold):
    # Convert image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    img_gb = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    # Apply bilateral filter
    img_bf = cv2.bilateralFilter(image, kernel_size, sigma_color, sigma_space)

    # Apply Otsu thresholding
#    _, img_thr_gb = cv2.threshold(img_gb, filter_threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#    _, img_thr_bf = cv2.threshold(img_bf, filter_threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    _, img_thr_gb = cv2.threshold(img_gb, filter_threshold, 255, cv2.THRESH_BINARY)
    _, img_thr_bf = cv2.threshold(img_bf, filter_threshold, 255, cv2.THRESH_BINARY)

    return img_thr_gb, img_thr_bf

if __name__ == "__main__":
    
    # Define used camera
    camera = "sony_hs" # "sony", "sony_hs", "gopro1", "gopro2

    # Define used marker
    marker = 'DICT_4X4_50'

    # Define gaussian blur kernel size for Gaussian blur in/and bilateral filter
    kernel_size = 25 # must be positive and odd

    # Define bilateral filter
    sigma_color = 15
    sigma_space = 35

    # Define filter threshold
    filter_threshold = 90

    # Define pattern
    pattern = 10 * np.array([[12.6, 12.8, 0], [98.5, 12.3, 0], [98.7, 98.5, 0], [13.1, 98.8, 0],
                    [499.2, 12.7, 0], [585.3, 12.8, 0], [585.1, 98.7, 0], [499.1, 98.6, 0],
                    [499.5, 501.2, 0], [585.5, 501.1, 0], [585.6, 587.1, 0], [499.6, 587.1, 0],
                    [12.7, 501.2, 0], [98.3, 501.2, 0], [98.4, 587.5, 0], [12.6, 587.3, 0]]
                    )

    # Import calibration parameters
    K = np.loadtxt("calibration/" + camera + "/K.txt")  # calibration matrix[3x3]
    d = np.loadtxt("calibration/" + camera + "/d.txt")  # distortion coefficients[2x1]
    
    # Import image
    image = cv2.imread("data/f_r8_d113_h40_200.jpg")

    """ cv2.imshow("image", image)
    cv2.waitKey(0) """

    # Undistort image
    img_undst = undistort(K, d, image)

    
    # Detect markers
    img_det, corners, ids = detect(img_undst, marker)
#    cv2.imshow('image_det', cv2.resize(img_det, (1920, 1080)))
#    cv2.waitKey(0)

    # Warp perspective
    img_warp, M = warp(corners, pattern, img_undst)
#    cv2.imshow('image_warp', cv2.resize(img_warp, (1080, 1080)))
#    cv2.waitKey(0)

    # Threshold image
    img_thr_gb, img_thr_bf = threshold(img_warp, kernel_size, sigma_color, sigma_space, filter_threshold)
    
    fig, axis = plt.subplots(2, 2,sharex=True, sharey=True)
    axis[0, 0].set_title('Gausian Blur')
    axis[0, 0].imshow(img_thr_gb)
    axis[0, 1].set_title('Bilateral Filter')
    axis[0, 1].imshow(img_thr_bf)
    axis[1, 0].set_title('Original Image')
    axis[1, 0].imshow(img_warp)
    axis[1, 1].set_title('Original Image')
    axis[1, 1].imshow(img_warp)
    plt.show()

