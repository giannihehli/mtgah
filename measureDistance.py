# Importing libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy import linalg
import time

# Importing user-defined modules
from undistortImage import undistort
from detectMarkers import detect
from warpPerspective import warp
from filterImage import threshold

def get_extrema(img_roi):

    # Get shape of region of interest
    nrows, ncols = np.shape(img_roi)

    # Define grid of indices
    indices_2d, _ = np.mgrid[0:nrows, 0:ncols]

    # Define booleans for whole roi
    img_bool = img_roi == 255
    
    # Get minimum and maximum index of roi
    top_row = np.amin(indices_2d[img_bool])
    bot_row = np.amax(indices_2d[img_bool])

    # Define 1D indices
    indices_1d = np.arange(ncols)

    top_bool = img_roi[top_row, :] == 255
    bot_bool = img_roi[bot_row, :] == 255
    
    top_col = int(np.median(indices_1d[top_bool]))
    bot_col = int(np.median(indices_1d[bot_bool]))

    top = np.array([top_row, top_col])
    bottom = np.array([bot_row, bot_col])

    return top, bottom

def measure(image, image_thr, search_width, top_search):
#    print('start time: ', datetime.now())
    
    # Define horizontal search window
    horizontal_left = int(500)
    horizontal_right = int(5500)
    horizontal_y = int(3000)

    # Define vertical search window
    vertical_x = int(3000)
    vertical_top = int(3000)
    vertical_bottom = int(5500)

    # Create copy of image for drawing search windows
    image_windows = image.copy()

    # For last frame set vertical search window to top search window
    if top_search:
        vertical_top = int(500)

    # Draw search windows
    cv2.rectangle(image_windows, (horizontal_left, horizontal_y-search_width//2), (horizontal_right, horizontal_y + search_width//2), (0, 0, 255), 20)
    cv2.rectangle(image_windows, (vertical_x + search_width//2, vertical_top), (vertical_x-search_width//2, vertical_bottom), (0, 0, 255), 20)

    # Draw search direction
    cv2.arrowedLine(image_windows, (horizontal_left, horizontal_y), (horizontal_left+500, horizontal_y), (255, 0, 0), 20)
    cv2.arrowedLine(image_windows, (horizontal_right, horizontal_y), (horizontal_right-500, horizontal_y), (255, 0, 0), 20)
    cv2.arrowedLine(image_windows, (vertical_x, vertical_bottom), (vertical_x, vertical_bottom-500), (255, 0, 0), 20)

    # Get extrema of vertical search window
    top, bottom = get_extrema(image_thr[vertical_top:vertical_bottom, vertical_x-search_width//2:vertical_x+search_width//2])

    # Get extrema of horizontal search window
    left, right = get_extrema((image_thr[horizontal_y-search_width//2:horizontal_y+search_width//2, horizontal_left:horizontal_right]).T)

    # Calculate the image coordinates of the extrema
    x_left = left[0] + horizontal_left
    y_left = horizontal_y - search_width//2 + left[1]
    x_right = right[0] + horizontal_left
    y_right = horizontal_y - search_width//2 + right[1]
    y_top = vertical_top + top[0]
    x_top = vertical_x - search_width//2 + top[1]
    y_bottom = vertical_top + bottom[0]
    x_bottom = vertical_x - search_width//2 + bottom[1]

    # Display extrema results on image
    cv2.circle(image_windows, (int(x_left), int(y_left)), 10, (0, 255, 0), 5)
    cv2.circle(image_windows, (int(x_right), int(y_right)), 10, (0, 255, 0), 5) 
    cv2.circle(image_windows, (int(x_bottom), int(y_bottom)), 10, (0, 255, 0), 5)
    if top_search:
        cv2.arrowedLine(image_windows, (vertical_x, vertical_top), (vertical_x, vertical_top+500), (255, 0, 0), 20)
        cv2.circle(image_windows, (int(x_top), int(y_top)), 10, (0, 255, 0), 5)

    return x_right, x_left, y_top, y_bottom, image_windows

if __name__ == "__main__":
    
    # Define used camera
    camera = "sony_hs" # "sony", "sony_hs", "gopro1", "gopro2
    
    # Define gaussian blur kernel size for Gaussian blur in/and bilateral filter
    kernel_size = 35 # must be positive and odd

    # Define bilateral filter
    sigma_color = 80
    sigma_space = 35

    # Define filter threshold
    filter_threshold = 100
    
    # Define ROI width for measurement
    search_width = int(1000)

    # Import calibration parameters
    K = np.loadtxt("G:/data/pipeline_tests/camera/calibration/K.txt")  # calibration matrix[3x3]
    d = np.loadtxt("G:/data/pipeline_tests/camera/calibration/d.txt")  # distortion coefficients[2x1]
    
    # Define used basis
    basis =  "r0-pa" # "rough", "smooth"

    # Define reference pattern in clockwise order in world frame (3D) in [0.1mm]
    match basis:
        case 'r0-pa': # changed from earlier pa
            pattern = 100 * np.array([[0.9, 0.84, 0], [8.96, 0.82, 0], [8.95, 8.92, 0], [0.87, 8.93, 0],
                                    [50.87, 0.82, 0], [58.95, 0.82, 0], [58.9, 8.91, 0], [50.83, 8.9, 0],
                                    [50.74, 51.07, 0], [58.83, 51.02, 0], [58.88, 59.1, 0], [50.8, 59.14, 0],
                                    [0.86, 51.02, 0], [8.94, 51.07, 0], [8.88, 59.17, 0], [0.79, 59.11, 0]])
        case 'r0-pe': # changed from earlier pe
            pattern = 100 * np.array([[59.07, 0.87, 0], [59.18, 8.96, 0], [51.07, 8.95, 0], [50.98, 0.86, 0],
                                    [59.18, 50.87, 0], [59.13, 58.93, 0], [51.06, 58.9, 0], [51.09, 50.82, 0],
                                    [8.91, 50.74, 0], [8.96, 58.83, 0], [0.88, 58.88, 0], [0.85, 50.8, 0],
                                    [8.89, 0.86, 0], [8.92, 8.94, 0], [0.82, 8.88, 0], [0.81, 0.8, 0]])
        case 'r2-pa':
            pattern = 100 * np.array([[1.06, 0.82, 0], [9.15, 0.82, 0], [9.12, 8.92, 0], [1.05, 8.92, 0],
                                    [50.7, 0.91, 0], [58.77, 0.94, 0], [58.75, 9.04, 0], [50.66, 9, 0],
                                    [50.57, 51.09, 0], [58.63, 50.99, 0], [58.73, 59.07, 0], [50.66, 59.14, 0],
                                    [1.08, 51.02, 0], [9.15, 51.02, 0], [9.12, 59.13, 0], [1.04, 59.11, 0]])
        case 'r2-pe':
            pattern = 100 * np.array([[59.1, 1.06, 0], [59.12, 9.15, 0], [51.02, 9.12, 0], [51, 1.05, 0],
                                    [59.04, 50.7, 0], [58.99, 58.77, 0], [58.88, 58.75, 0], [50.96, 50.66, 0],
                                    [8.87, 50.57, 0], [8.91, 58.63, 0], [0.85, 58.73, 0], [0.81, 50.66, 0],
                                    [8.9, 1.08, 0], [8.91, 9.15, 0], [0.81, 9.12, 0], [0.86, 1.04, 0]])
        case 'r4-pa':
            pattern = 100 * np.array([[1.3, 0.82, 0], [9.38, 0.83, 0], [9.33, 8.94, 0], [1.24, 8.92, 0],
                                    [50.39, 0.89, 0], [58.46, 0.93, 0], [58.41, 9.04, 0], [50.32, 8.98, 0],
                                    [50.33, 50.98, 0], [58.42, 50.92, 0], [58.48, 59.00, 0], [50.4, 59.04, 0],
                                    [1.23, 51.01, 0], [9.28, 50.96, 0], [9.29, 59.06, 0], [1.24, 59.08, 0]])
        case 'r4-pe':
            pattern = 100 * np.array([[59.08, 1.3, 0], [59.07, 9.38, 0], [50.97, 9.33, 0], [51.0, 1.24, 0],
                                    [59.0, 50.39, 0], [58.93, 58.46, 0], [50.83, 58.41, 0], [50.92, 50.32, 0],
                                    [8.9, 50.33, 0], [8.92, 58.42, 0], [0.89, 58.48, 0], [0.83, 50.4, 0],
                                    [8.97, 1.23, 0], [8.98, 9.28, 0], [0.87, 9.29, 0], [0.89, 1.24, 0]])
            

    # Import image
    image = cv2.imread("G:/data/tests/sony_hs/f_r8_d113_h40/f_r8_d113_h40_200.jpg")

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
    """ for marker in range(4):
        for corner in range(3):
#            print('warpped marker', marker, ' L', corner, ': ', linalg.norm([corners_warp[marker*4+corner+1][0] - corners_warp[marker*4+corner][0], corners_warp[marker*4+corner+1][1] - corners_warp[marker*4+corner][1]]))
#            print('pattern marker', marker, ' L', corner, ': ', 10000*linalg.norm([pattern[marker*4+corner+1][0] - pattern[marker*4+corner][0], pattern[marker*4+corner+1][1] - pattern[marker*4+corner][1]]))
            print('marker', marker+1, ' diff', corner+1, ': ', linalg.norm([corners_warp[marker*4+corner+1][0] - corners_warp[marker*4+corner][0], 
                                                                            corners_warp[marker*4+corner+1][1] - corners_warp[marker*4+corner][1]])-linalg.norm([pattern[marker*4+corner+1][0] - pattern[marker*4+corner][0], 
                                                                                                                                                                 pattern[marker*4+corner+1][1] - pattern[marker*4+corner][1]])) """

    # Threshold image
    img_thr_gb, img_thr_bf = threshold(img_warp, kernel_size, sigma_color, sigma_space, filter_threshold)

    cv2.imshow("image_thr", cv2.resize(img_thr_bf, (1080, 1080)))
    cv2.waitKey(0)

    # Measure distance
#    _, _, _, img_mes_gb = measure(img_warp, img_thr_gb, search_width)
    _, _, _, _, img_mes_bf = measure(img_warp, img_thr_bf, search_width, False)

    """ # Plot result
    fig, axis = plt.subplots(2, 2,sharex=True, sharey=True)
    axis[0, 0].set_title('Gaussian Blur Threshold')
    axis[0, 0].imshow(img_thr_gb)
    axis[0, 0].set_title('Bilateral Filter Threshold')
    axis[0, 1].imshow(img_thr_bf)
    axis[0, 0].set_title('Gaussian Blur Measurement')
    axis[1, 0].imshow(img_mes_gb)
    axis[0, 0].set_title('bilateral Filter Measurement')
    axis[1, 1].imshow(img_mes_bf)
    plt.show() """
