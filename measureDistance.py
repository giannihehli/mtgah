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
    cv2.rectangle(image_windows, (horizontal_left, horizontal_y-search_width//2), 
                  (horizontal_right, horizontal_y + search_width//2), (0, 0, 255), 20)
    cv2.rectangle(image_windows, (vertical_x + search_width//2, vertical_top), 
                  (vertical_x-search_width//2, vertical_bottom), (0, 0, 255), 20)

    # Draw search direction
    cv2.arrowedLine(image_windows, (horizontal_left, horizontal_y), 
                    (horizontal_left+500, horizontal_y), (255, 0, 0), 20)
    cv2.arrowedLine(image_windows, (horizontal_right, horizontal_y), 
                    (horizontal_right-500, horizontal_y), (255, 0, 0), 20)
    cv2.arrowedLine(image_windows, (vertical_x, vertical_bottom), 
                    (vertical_x, vertical_bottom-500), (255, 0, 0), 20)

    # Get extrema of vertical search window
    top, bottom = get_extrema(image_thr[vertical_top:vertical_bottom, 
                                        vertical_x-search_width//2:vertical_x + 
                                        search_width//2])

    # Get extrema of horizontal search window
    left, right = get_extrema((image_thr[horizontal_y-search_width//2:horizontal_y + 
                                         search_width//2, 
                                         horizontal_left:horizontal_right]).T)

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
        cv2.arrowedLine(image_windows, (vertical_x, vertical_top), 
                        (vertical_x, vertical_top+500), (255, 0, 0), 20)
        cv2.circle(image_windows, (int(x_top), int(y_top)), 10, (0, 255, 0), 5)

    return x_right, x_left, y_top, y_bottom, image_windows

if __name__ == "__main__":
    ####################################################################################
    # ONLY SECTION TO ADJUST PARAMETERS
    
    # Define paths
    data_path = 'data/'
    img_name_warp = 'warp.png'
    img_name_thr = 'thr.png'
  
    # Define ROI width for measurement
    search_width = int(1000)

    ####################################################################################

    # Load image
    img_warp = cv2.imread(data_path + img_name_warp)
    img_thr = cv2.imread(data_path + img_name_thr)

    # Convert thresholded image to grayscale
    img_thr = cv2.cvtColor(img_thr, cv2.COLOR_BGR2GRAY)

    cv2.imshow("image_thr", cv2.resize(img_thr, (1080, 1080)))
    cv2.waitKey(0)

    # Measure distance
    _, _, _, _, img_mes = measure(img_warp, img_thr, search_width, False)

    # Display measured image
    cv2.imshow("image_mes", cv2.resize(img_mes, (1080, 1080)))
    cv2.waitKey(0)

    # Save measured image
    print('Save detected image: ', 'mes_' + img_name_warp)
    cv2.imwrite(data_path + 'mes_' + img_name_warp, img_mes)