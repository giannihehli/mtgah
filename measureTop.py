# Importing libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy import linalg
from datetime import datetime

# Importing user-defined modules
from undistortImage import undistort
from detectMarkers import detect
from warpPerspective import warp
from filterImage import threshold

def measuretop(image, image_thr, search_width):
#    print('start time: ', datetime.now())

    # Define search window
    horizontal_y = 3000
    vertical_x = 3000
    vertical_top = 500

    # Define pre-search thresholds
    pre_search_threshold = 300

    # Draw search windows
    image_windows = image.copy()
    cv2.rectangle(image_windows, (int(vertical_x-search_width/2), int(vertical_top)), (int(vertical_x + search_width/2), int(horizontal_y-search_width/2)), (0, 0, 255), 20)

    # Draw search direction
    cv2.arrowedLine(image_windows, (int(vertical_x), int(vertical_top)), (int(vertical_x), int(vertical_top+500)), (255, 0, 0), 20)
  
    # Define pre-search parameter
    y_top = vertical_top

    # Pre-search for edges in middle of top search region
    while image_thr[int(y_top), int(vertical_x)] == 0:
        y_top += 1

    # Define top loop parameters
    x_top = vertical_x - search_width/2
    y_top = y_top - pre_search_threshold

    # Search for edges from left top to right bottom
    while image_thr[int(y_top), int(x_top)] == 0:
        y_top += 1
        x_top = vertical_x - search_width/2
        while image_thr[int(y_top), int(x_top)] == 0 and x_top < vertical_x + search_width/2:
            x_top += 1           

    """ print('top match value: ', image_thr[int(y_top), int(x_top)])
    print('top match coordinates: ', x_top, y_top) """
  
    # Post-process results
    x_top = 0
    count_top = 0

    # Look for mean of y_left and y_right
    for i in range(search_width):
        if image_thr[int(y_top), int(vertical_x - search_width/2 + i)] == 255:
            x_top = x_top + vertical_x - search_width/2 + i
            count_top += 1        

    # Calculate mean for y_left and y_right
    x_top = x_top / count_top
    
    """ print('top match value: ', image_thr[int(y_top), int(x_top)])
    print('top match coordinates: ', x_top, y_top)

    print('end time: ', datetime.now()) """

    # Display results on image
    cv2.circle(image_windows, (int(x_top), int(y_top)), 10, (0, 255, 0), 5) 
    
    # Plot result
    """ fig, axis = plt.subplots(1, 2,sharex=True, sharey=True)
    axis[0].imshow(image_windows)
    axis[1].imshow(image_thr)
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.show() """

    return y_top, image_windows

if __name__ == "__main__":
       
    # Define used camera
    camera = "sony" # "sony", "sony_hs", "gopro1", "gopro2

    # Define used marker
    marker = 'DICT_4X4_50'

    # Define gaussian blur kernel size for Gaussian blur in/and bilateral filter
    kernel_size = 25 # must be positive and odd

    # Define bilateral filter
    sigma_color = 15
    sigma_space = 35

    # Define filter threshold
    filter_threshold = 90
    
    # Define ROI width for measurement
    search_width = 200

    # Import calibration parameters
    K = np.loadtxt("calibration/" + camera + "/K.txt")  # calibration matrix[3x3]
    d = np.loadtxt("calibration/" + camera + "/d.txt")  # distortion coefficients[2x1]
    
    # Define used basis
    basis =  "rough" # "rough", "smooth"

    # Load image and define reference pattern in clockwise order in world frame (3D) in [mm]
    match basis:
        case "smooth":
            pattern = 10 * np.array([[12.5, 6.2, 0], [99.5, 8, 0], [98.6, 95, 0], [11.5, 93.2, 0],
                                [498.8, 8.2, 0], [586, 8.2, 0], [586.5, 95.4, 0], [499.2, 95, 0],
                                [499.5, 503, 0], [586.7, 503.9, 0], [585.4, 591.2, 0], [498, 590.3, 0],
                                [16.2, 503.1, 0], [103.3, 503.2, 0], [102.7, 590.5, 0], [15.6, 590.5, 0]]
                                )
        case "rough":
            pattern = 10 * np.array([[12.6, 12.8, 0], [98.5, 12.3, 0], [98.7, 98.5, 0], [13.1, 98.8, 0],
                                [499.2, 12.7, 0], [585.3, 12.8, 0], [585.1, 98.7, 0], [499.1, 98.6, 0],
                                [499.5, 501.2, 0], [585.5, 501.1, 0], [585.6, 587.1, 0], [499.6, 587.1, 0],
                                [12.7, 501.2, 0], [98.3, 501.2, 0], [98.4, 587.5, 0], [12.6, 587.3, 0]]
                                )

    # Import image
    image = cv2.imread("data/f_r8_d113_h40_160.JPG")

    # Undistort image
    img_undst = undistort(K, d, image)
    
    # Detect markers
    marker = "DICT_4X4_1000"
    img_det, corners, ids = detect(img_undst, marker)

    """ cv2.imshow("image_det", cv2.resize(img_det, (1920, 1080)))
    cv2.waitKey(0) """

    # Warp perspective
    img_warp, M = warp(corners, pattern, img_undst)

    """ cv2.imshow("image_warp", cv2.resize(img_warp, (1080, 1080)))
    cv2.waitKey(0) """

    # Detect markers in warped image
    img_warp_det, corners_warp, ids_warp = detect(img_warp, marker)

    """ cv2.imshow("image_warp", cv2.resize(img_warp_det, (1080, 1080)))
    cv2.waitKey(0) """

    # Check size of detected markers
    for marker in range(4):
        for corner in range(3):
#            print('warpped marker', marker, ' L', corner, ': ', linalg.norm([corners_warp[marker*4+corner+1][0] - corners_warp[marker*4+corner][0], corners_warp[marker*4+corner+1][1] - corners_warp[marker*4+corner][1]]))
#            print('pattern marker', marker, ' L', corner, ': ', 10000*linalg.norm([pattern[marker*4+corner+1][0] - pattern[marker*4+corner][0], pattern[marker*4+corner+1][1] - pattern[marker*4+corner][1]]))
            print('marker', marker+1, ' diff', corner+1, ': ', linalg.norm([corners_warp[marker*4+corner+1][0] - corners_warp[marker*4+corner][0], corners_warp[marker*4+corner+1][1] - corners_warp[marker*4+corner][1]])-linalg.norm([pattern[marker*4+corner+1][0] - pattern[marker*4+corner][0], pattern[marker*4+corner+1][1] - pattern[marker*4+corner][1]]))

    # Threshold image
    img_thr_gb, img_thr_bf = threshold(img_warp, kernel_size, sigma_color, sigma_space, filter_threshold)

    # Measure distance
    _, img_mes_gb = measuretop(img_warp, img_thr_gb, search_width)
    _, img_mes_bf = measuretop(img_warp, img_thr_bf, search_width)

    # Plot result
    fig, axis = plt.subplots(2, 2,sharex=True, sharey=True)
    axis[0, 0].set_title('Gaussian Blur Threshold')
    axis[0, 0].imshow(img_thr_gb)
    axis[0, 0].set_title('Bilateral Filter Threshold')
    axis[0, 1].imshow(img_thr_bf)
    axis[0, 0].set_title('Gaussian Blur Measurement')
    axis[1, 0].imshow(img_mes_gb)
    axis[0, 0].set_title('bilateral Filter Measurement')
    axis[1, 1].imshow(img_mes_bf)
    plt.show()
