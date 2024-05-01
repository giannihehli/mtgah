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

    dst = np.array([[pattern[0][0], pattern[0][1]], [pattern[5][0], pattern[5][1]], [pattern[10][0], pattern[10][1]], [pattern[15][0], pattern[15][1]]], dtype=np.float32)

    # Get perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)

    # Wrap the image
    img_warp = cv2.warpPerspective(image, M, (int(6000), int(6000)), cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    return img_warp, M

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
        case 'r0-pa':
            image = cv2.imread("data/f_r0-pa_d113_h40_100.JPG")
            pattern = 100 * np.array([[0.9, 0.84, 0], [8.96, 0.82, 0], [8.95, 8.92, 0], [0.87, 8.93, 0],
                                     [50.87, 0.82, 0], [58.95, 0.82, 0], [58.9, 8.91, 0], [50.83, 8.9, 0],
                                     [50.74, 51.07, 0], [58.83, 51.02, 0], [58.88, 59.1, 0], [50.8, 59.14, 0],
                                     [0.86, 51.02, 0], [8.94, 51.07, 0], [8.88, 59.17, 0], [0.79, 59.11, 0]])
        case 'r0-pe':
            image = cv2.imread("data/f_r0-pe_d113_h40_100.JPG")
            pattern = 100 * np.array([[59.07, 0.87, 0], [59.18, 8.96, 0], [51.07, 8.95, 0], [50.98, 0.86, 0],
                                     [59.18, 50.87, 0], [59.13, 58.93, 0], [51.06, 58.9, 0], [51.09, 50.82, 0],
                                     [8.91, 50.74, 0], [8.96, 58.83, 0], [0.88, 58.88, 0], [0.85, 50.8, 0],
                                     [8.89, 0.86, 0], [8.92, 8.94, 0], [0.82, 8.88, 0], [0.81, 0.8, 0]])
        case "smooth":
            image = cv2.imread("data/DSC00233.JPG")
            pattern = 10 * np.array([[12.5, 6.2, 0], [99.5, 8, 0], [98.6, 95, 0], [11.5, 93.2, 0],
                                [498.8, 8.2, 0], [586, 8.2, 0], [586.5, 95.4, 0], [499.2, 95, 0],
                                [499.5, 503, 0], [586.7, 503.9, 0], [585.4, 591.2, 0], [498, 590.3, 0],
                                [16.2, 503.1, 0], [103.3, 503.2, 0], [102.7, 590.5, 0], [15.6, 590.5, 0]]
                                )
        case "rough":
            image = cv2.imread("data/C0037 - Trim_0.JPG")
            pattern = 10 * np.array([[12.6, 12.8, 0], [98.5, 12.3, 0], [98.7, 98.5, 0], [13.1, 98.8, 0],
                                [499.2, 12.7, 0], [585.3, 12.8, 0], [585.1, 98.7, 0], [499.1, 98.6, 0],
                                [499.5, 501.2, 0], [585.5, 501.1, 0], [585.6, 587.1, 0], [499.6, 587.1, 0],
                                [12.7, 501.2, 0], [98.3, 501.2, 0], [98.4, 587.5, 0], [12.6, 587.3, 0]]
                                )


    cv2.imshow("image", cv2.resize(image, (1920, 1080)))
    cv2.waitKey(0)

    # Undistort image
    img_undst = undistort(K, d, image)

    # Show undistorted image
    cv2.imshow('img_undst', cv2.resize(img_undst, (1920, 1080))) # Initial Capture
    cv2.waitKey(0)

    # Detect markers
    marker = "DICT_4X4_50" 
    img_det, corners, ids = detect(img_undst, marker) 
    cv2.imshow('img_det', cv2.resize(img_det, (1920, 1080))) # Detected Capture
    cv2.waitKey(0)

    img_warp, M = warp(corners, pattern, img_undst)
    cv2.imshow('img_warp', cv2.resize(img_warp, (1080, 1080))) # Transformed Capture
    cv2.waitKey(0)  

    # Check transformation
    rpe = 0
    img_warp_det, corners_warp, ids_warp = detect(img_warp, marker)
    cv2.imshow('img_warp_det', cv2.resize(img_warp_det, (1080, 1080))) # Detected Capture
    cv2.waitKey(0)

    print("Transformation matrix: \n", M)
    for i in range(len(corners)):
        print('corners ', i, ': ', [corners[i][0], corners[i][1]])
        img_pts = M.dot([corners[i][0], corners[i][1], 1])/M[2, :].dot([corners[i][0], corners[i][1], 1])
        print('img_pts ', i, ': ', img_pts[0:2])
        print('corner_warp ', i, ': ', corners_warp[i][0], corners_warp[i][1])
        rpe = rpe + np.linalg.norm([corners_warp[i][0], corners_warp[i][1]] - img_pts[0:2])

    print('RPE: ', rpe/len(pattern))    

    # Display the transformed image
    
    f, axarr = plt.subplots(2, 1)
    axarr[0].imshow(img_undst)
    axarr[1].imshow(img_warp)
    plt.show()