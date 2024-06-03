# Importing libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy import linalg

# Importing user-defined modules
from undistortImage import undistort
from detectMarkers import detect
from warpPerspective import warp
from filterImage import threshold
from alignPointcloud import export

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

    if count_top == 0:
        count_top = 1

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

def convert(img_thr):
    
    # Rotate the 2D array 90 degrees clockwise
    rotated_img_thr = np.rot90(img_thr, -1)

    img_z = np.where(rotated_img_thr == 0, np.nan, 0)

    # Define x and y edges of raster by adjustng to raster size
    img_x = np.linspace(0, 5999, num = 6000)
    img_y = np.linspace(0, 5999, num = 6000)

    return img_z, img_x, img_y

if __name__ == "__main__":
       
    # Define used camera
    camera = "sony_hs" # "sony", "sony_hs", "gopro1", "gopro2

    # Define used marker
    marker = 'DICT_4X4_1000'

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
    K = np.loadtxt(f'G:/data/pipeline_tests/camera/calibration/K.txt')  # calibration matrix[3x3]
    d = np.loadtxt(f'G:/data/pipeline_tests/camera/calibration/d.txt')  # distortion coefficients[2x1]
    
    # Define pattern as 3D coordinates in 0.1mm
    pattern = 100 * np.array([[0.9, 0.84, 0], [8.96, 0.82, 0], [8.95, 8.92, 0], [0.87, 8.93, 0],
                                     [50.87, 0.82, 0], [58.95, 0.82, 0], [58.9, 8.91, 0], [50.83, 8.9, 0],
                                     [50.74, 51.07, 0], [58.83, 51.02, 0], [58.88, 59.1, 0], [50.8, 59.14, 0],
                                     [0.86, 51.02, 0], [8.94, 51.07, 0], [8.88, 59.17, 0], [0.79, 59.11, 0]])

    # Import image
    image = cv2.imread("G:/data/pipeline_tests/camera/100_orig.PNG")

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
#    plt.show()

#    np.savetxt('G:/data/pipeline_tests/end frames/test_endframe_svtxt.asc', img_thr_bf, fmt='%d')

    # Define raster size []
    raster_size_img = 0.0001

    # Convert last frame to needed data structure for asc export
    img_z, img_x, img_y = convert(img_thr_bf)

    print('image threshold shape: ', img_thr_bf.shape)
    print('image shape: ', img_z.shape)
    print('x shape: ', img_x.shape)
    print('y shape: ', img_y.shape)

    # Export last frame as ascii file
    export(img_z, img_x, img_y, raster_size_img, 'G:/data/pipeline_tests/end frames/test_endframe.asc')  