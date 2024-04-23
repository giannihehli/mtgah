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

def measure(image, image_thr):
#    print('start time: ', datetime.now())
    
    # Define horizontal search window
    horizontal_left = 300
    horizontal_right = 5700
    horizontal_y = 3000
    horizontal_width = 300

    # Define vertical search window
    vertical_x = 3000
    vertical_width = 300
    vertical_bottom = 5500

    # Draw search windows
    image_windows = image.copy()
    cv2.rectangle(image_windows, (int(horizontal_left), int(horizontal_y-horizontal_width/2)), (int(horizontal_right), int(horizontal_y + horizontal_width/2)), (0, 0, 255), 20)
    cv2.rectangle(image_windows, (int(vertical_x-vertical_width/2), int(vertical_bottom)), (int(vertical_x + vertical_width/2), int(horizontal_y+horizontal_width/2)), (0, 0, 255), 20)

    # Draw search direction
    cv2.arrowedLine(image_windows, (int(horizontal_left), int(horizontal_y)), (int(horizontal_left+500), int(horizontal_y)), (255, 0, 0), 20)
    cv2.arrowedLine(image_windows, (int(horizontal_right), int(horizontal_y)), (int(horizontal_right-500), int(horizontal_y)), (255, 0, 0), 20)
    cv2.arrowedLine(image_windows, (int(vertical_x), int(vertical_bottom)), (int(vertical_x), int(vertical_bottom-500)), (255, 0, 0), 20)

    # Define left loop parameters
    x_left = horizontal_left
    y_left = horizontal_y - horizontal_width/2

    # Search for edges from left top to right bottom
    while image_thr[int(y_left), int(x_left)] == 0:
        x_left += 1
        y_left = horizontal_y - horizontal_width/2
        while image_thr[int(y_left), int(x_left)] == 0 and y_left < horizontal_y + horizontal_width/2:
            y_left += 1
            
    """ print('left match value: ', image_thr[int(y_left),int(x_left)])
    print('left match coordinates: ', x_left, y_left)  """  

    # Define right loop parameters
    x_right = horizontal_right
    y_right = horizontal_y - horizontal_width/2

    # Search for edges from right top to left bottom
    while image_thr[int(y_right), int(x_right)] == 0:
        x_right -= 1
        y_right = horizontal_y - horizontal_width/2
        while image_thr[int(y_right), int(x_right)] == 0 and y_right < horizontal_y + horizontal_width/2:
            y_right += 1           

    """ print('right match value: ', image_thr[int(y_right), int(x_right)])
    print('right match coordinates: ', x_right, y_right) """
  

    # Define bottom loop parameters
    x_bottom = vertical_x - vertical_width/2
    y_bottom = vertical_bottom

    # Search for edges from left bottom to right top
    while image_thr[int(y_bottom), int(x_bottom)] == 0:
        y_bottom -= 1
        x_bottom = vertical_x - vertical_width/2
        while image_thr[int(y_bottom), int(x_bottom)] == 0 and x_bottom < vertical_x + vertical_width/2:
            x_bottom += 1           

    """ print('bottom match value: ', image_thr[int(y_bottom), int(x_bottom)])
    print('bottom match coordinates: ', x_bottom, y_bottom) """
  
    # Post-process results
    y_left = 0
    count_left = 0    
    y_right = 0
    count_right = 0
    x_bottom = 0
    count_bottom = 0

    # Look for mean of y_left and y_right
    for i in range(horizontal_width):
        if image_thr[int(horizontal_y - horizontal_width/2 + i), int(x_left)] == 255:
            y_left = y_left + horizontal_y - horizontal_width/2 + i
            count_left += 1
            
    for i in range(horizontal_width):        
        if image_thr[int(horizontal_y - horizontal_width/2 + i), int(x_right)] == 255:
            y_right = y_right + horizontal_y - horizontal_width/2 + i
            count_right += 1

    for i in range(vertical_width):
        if image_thr[int(y_bottom), int(vertical_x - vertical_width/2 + i)] == 255:
            x_bottom = x_bottom + vertical_x - vertical_width/2 + i
            count_bottom += 1        

    # Calculate mean for y_left and y_right
    y_left = y_left / count_left
    y_right = y_right / count_right
    x_bottom = x_bottom / count_bottom
    
    """ print('left match value: ', image_thr[int(y_left),int(x_left)])
    print('left match coordinates: ', x_left, y_left)

    print('right match value: ', image_thr[int(y_right), int(x_right)])
    print('right match coordinates: ', x_right, y_right)

    print('bottom match value: ', image_thr[int(y_bottom), int(x_bottom)])
    print('bottom match coordinates: ', x_bottom, y_bottom)

    print('end time: ', datetime.now()) """

    # Display results on image
    cv2.circle(image_windows, (int(x_left), int(y_left)), 10, (0, 255, 0), 5)
    cv2.circle(image_windows, (int(x_right), int(y_right)), 10, (0, 255, 0), 5) 
    cv2.circle(image_windows, (int(x_bottom), int(y_bottom)), 10, (0, 255, 0), 5) 

    """    # Plot result
    f, axarr = plt.subplots(1, 2,sharex=True, sharey=True)
    axarr[0].imshow(image_windows)
    axarr[1].imshow(image_thr)
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.show() """

    return x_right, x_left, y_bottom, image_windows

if __name__ == "__main__":
    
    # Define used camera
    camera = "sony" # "sony", "sony_hs", "gopro1", "gopro2

    # Import calibration parameters
    K = np.loadtxt("calibration/" + camera + "/K.txt")  # calibration matrix[3x3]
    d = np.loadtxt("calibration/" + camera + "/d.txt")  # distortion coefficients[2x1]
    
    # Define used basis
    basis =  "smooth" # "rough", "smooth"

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
    marker = "DICT_4X4_50"
    img_det, corners, ids = detect(img_undst, marker)

    cv2.imshow("image_det", cv2.resize(img_det, (1920, 1080)))
    cv2.waitKey(0)

    # Warp perspective
    img_warp, M = warp(corners, pattern, img_undst)
    cv2.imshow("image_warp", cv2.resize(img_warp, (1080, 1080)))
    cv2.waitKey(0)

    img_warp_det, corners_warp, ids_warp = detect(img_warp, marker)

    # Check size of detected markers
    for marker in range(4):
        for corner in range(3):
#            print('warpped marker', marker, ' L', corner, ': ', linalg.norm([corners_warp[marker*4+corner+1][0] - corners_warp[marker*4+corner][0], corners_warp[marker*4+corner+1][1] - corners_warp[marker*4+corner][1]]))
#            print('pattern marker', marker, ' L', corner, ': ', 10000*linalg.norm([pattern[marker*4+corner+1][0] - pattern[marker*4+corner][0], pattern[marker*4+corner+1][1] - pattern[marker*4+corner][1]]))
            print('marker', marker+1, ' diff', corner+1, ': ', linalg.norm([corners_warp[marker*4+corner+1][0] - corners_warp[marker*4+corner][0], corners_warp[marker*4+corner+1][1] - corners_warp[marker*4+corner][1]])-10000*linalg.norm([pattern[marker*4+corner+1][0] - pattern[marker*4+corner][0], pattern[marker*4+corner+1][1] - pattern[marker*4+corner][1]]))
    cv2.imshow("image_warp", cv2.resize(img_warp, (1080, 1080)))
    cv2.waitKey(0)

    # Threshold image
    img_thr_gb, img_thr_bf = threshold(img_warp)

    # Measure distance
    measure(img_warp, img_thr_gb)