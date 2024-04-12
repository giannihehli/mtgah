# Importing libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Importing user-defined modules
from undistortImage import undistort
from locateCamera import locate
from detectMarkers import detect


def warp(corners, pattern, image):
    
    src = np.array([[corners[0][0], corners[0][1]], [corners[5][0], corners[5][1]], [corners[10][0], corners[10][1]], [corners[15][0], corners[15][1]]], dtype=np.float32)

    dst = 10000* np.array([[pattern[0][0], pattern[0][1]], [pattern[5][0], pattern[5][1]], [pattern[10][0], pattern[10][1]], [pattern[15][0], pattern[15][1]]], dtype=np.float32)

    # Get perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)

    # Wrap the image
    img_warp = cv2.warpPerspective(image, M, (int(6000), int(6000)))

    return img_warp

if __name__ == "__main__":

    # Define used camera
    camera = "sony_hs" # "sony", "sony_hs", "gopro1", "gopro2

    # Define used basis
    basis =  "rough" # "rough", "smooth"

    # Import calibration parameters
    K = np.loadtxt("calibration/" + camera + "/K.txt")  # calibration matrix[3x3]
    d = np.loadtxt("calibration/" + camera + "/d.txt")  # distortion coefficients[2x1]
    
    # Load image and define reference pattern in clockwise order in world frame (3D) in [mm]
    match basis:
        case "smooth":
            image = cv2.imread("data/DSC00233.JPG")
            pattern = 0.001 * np.array([[12.5, 6.2, 0], [99.5, 8, 0], [98.6, 95, 0], [11.5, 93.2, 0],
                                [498.8, 8.2, 0], [586, 8.2, 0], [586.5, 95.4, 0], [499.2, 95, 0],
                                [499.5, 503, 0], [586.7, 503.9, 0], [585.4, 591.2, 0], [498, 590.3, 0],
                                [16.2, 503.1, 0], [103.3, 503.2, 0], [102.7, 590.5, 0], [15.6, 590.5, 0]]
                                )
        case "rough":
            image = cv2.imread("data/C0037 - Trim_0.JPG")
            pattern = 0.001 * np.array([[12.6, 12.8, 0], [98.5, 12.3, 0], [98.7, 98.5, 0], [13.1, 98.8, 0],
                                [499.2, 12.7, 0], [585.3, 12.8, 0], [585.1, 98.7, 0], [499.1, 98.6, 0],
                                [499.5, 501.2, 0], [585.5, 501.1, 0], [585.6, 587.1, 0], [499.6, 587.1, 0],
                                [12.7, 501.2, 0], [98.3, 501.2, 0], [98.4, 587.5, 0], [12.6, 587.3, 0]]
                                )


    cv2.imshow("image", image)
    cv2.waitKey(0)

    # Undistort image
    img_undst = undistort(K, d, image)

    # Show undistorted image
    cv2.imshow('img_undst', img_undst) # Initial Capture
    cv2.waitKey(0)

    # Detect markers
    marker = "DICT_4X4_50" 
    img_det, corners, ids = detect(img_undst, marker) 

    img_warp = warp(corners, pattern, img_undst)

    # Display the transformed image
    
    f, axarr = plt.subplots(2, 1)
    axarr[0].imshow(img_undst)
    axarr[1].imshow(img_warp)
    plt.show()

    cv2.imshow('warped', cv2.resize(img_warp, (1080, 1080))) # Transformed Capture
    cv2.waitKey(0)