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

    # Apply binary thresholding
    _, img_thr_gb = cv2.threshold(img_gb, filter_threshold, 255, cv2.THRESH_BINARY)
    _, img_thr_bf = cv2.threshold(img_bf, filter_threshold, 255, cv2.THRESH_BINARY)

    return img_thr_gb, img_thr_bf

if __name__ == "__main__":
    ####################################################################################
    # ONLY SECTION TO ADJUST PARAMETERS
    
    # Define paths
    data_path = 'data/'
    img_name = 'warp.png'

    # Define gaussian blur kernel size for Gaussian blur in/and bilateral filter (45)
    kernel_size = 35 # [px] must be positive and odd

    # Define bilateral filter parameters (200, 25)
    sigma_color = 80 # [px] Filter sigma in the color space.A larger value of the parameter 
    # means that farther colors within the pixel neighborhood (see sigmaSpace) will be 
    # mixed together, resulting in larger areas of semi-equal color.

    sigma_space = 35 # [px] Filter sigma in the coordinate space. A larger value of the 
    # parameter means that farther pixels will influence each other as long as their colors 
    # are close enough (see sigmaColor ). When d>0, it specifies the neighborhood size 
    # regardless of sigmaSpace. Otherwise, d is proportional to sigmaSpace.

    # Define filter threshold for binary thresholding (90)
    filter_threshold = 100 # [px] Threshold value for binary thresholding - the lower the 
    #number the less points will be black

    ####################################################################################

    # Load undistorted image
    img_warp = cv2.imread(data_path + img_name)

    # Threshold image
    gb, bf = threshold(img_warp, kernel_size, sigma_color, sigma_space, filter_threshold)

    # Save thresholded image
    print('Save detected image: ', 'thr_' + img_name)
    cv2.imwrite(data_path + 'thr_' + img_name, bf)

    # Display images for comparison
    fig, axis = plt.subplots(2, 2,sharex=True, sharey=True)
    axis[0, 0].set_title('Gausian Blur')
    axis[0, 0].imshow(gb)
    axis[0, 1].set_title('Bilateral Filter')
    axis[0, 1].imshow(bf)
    axis[1, 0].set_title('Original Image')
    axis[1, 0].imshow(img_warp)
    axis[1, 1].set_title('Original Image')
    axis[1, 1].imshow(img_warp)
    plt.show()