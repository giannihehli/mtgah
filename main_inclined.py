# IMPORT PACKAGES AND MODULES
import os
import cv2
import glob
import time
import numpy as np
import pandas as pd
from pyntcloud import PyntCloud


# IMPORT USER-DEFINED MODULES
from calibrateCameravideo import calibratevideo
from undistortImage import undistort
from detectMarkers import detect
from detectMarkers import sortcorners
from warpPerspective import warp
from filterImage import threshold
from measureDistance import measure
from plotParameters import plotparams
from plotExperiments import plottotal
from measureTop import measuretop
from alignPointcloud import getimage
from alignPointcloud import sortpoints
from alignPointcloud import calculate_transformation
from alignPointcloud import transform
from alignPointcloud import rasterize
from alignPointcloud import convertimage
from alignPointcloud import export

if __name__ == '__main__':
    ############################################################################################################
    # ONLY SECTION TO ADJUST PARAMETERS

    # Define path with data to be analysed
    data = 'G:/data/'
#    data = 'G:/experiments/'

    # Define day (or name) of experiments that should be analysed
    day = 'pipeline_tests'
#    day = '20240604'
#    day = 'combined'

    #######################################
    # CAMERA OPTIONS

    # Input factor for skipping frames in calibration video
    skip_frames = 10  # [] Number of frames to skip in calibration video

    # Define images for optional output of respective images
    img_out = '' # Options: '', _undst', '_det', '_warp', '_thr', '_mes'

    # Define exp_out for optional output of all respective experiment images
    exp_out = '' # Options: '', '<<experiment_name>>',

    # Define gaussian blur kernel size for Gaussian blur in/and bilateral filter (45)
    kernel_size = 35 # [px] must be positive and odd

    # Define bilateral filter parameters (200, 25)
    sigma_color = 80 # [px] Filter sigma in the color space. A larger value of the parameter means that farther colors within the pixel neighborhood (see sigmaSpace) will be mixed together, resulting in larger areas of semi-equal color.
    sigma_space = 35 # [px] Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels will influence each other as long as their colors are close enough (see sigmaColor ). When d>0, it specifies the neighborhood size regardless of sigmaSpace. Otherwise, d is proportional to sigmaSpace.

    # Define filter threshold for binary thresholding (90)
    filter_threshold = 100 # [px] Threshold value for binary thresholding - the lower the number the less points will be black

    # Define ROI width for measurement
    search_width = 1000 # [px] Width of the ROI for distance measurement

    #######################################
    # SCANNER OPTIONS

    # Define how many aruco codes are used on the scanned area
    aruco_count = 4 # [] Number of aruco codes used on the scanned area

    # Define raster size in m for rasterization of scanned pointcloud
    raster_size = 0.001 # [m] Size of one bin in the raster in x and y direction

    # Define raster min and max values in [m] for x and y direction
    raster_min_x = 0.1 # [m] Minimum value of the raster in x direction
    raster_max_x = 0.5 # [m] Maximum value of the raster in x direction
    raster_min_y = 0.1 # [m] Minimum value of the raster in y direction
    raster_max_y = 0.5 # [m] Maximum value of the raster in y direction

    # Define raster size in [m] for rasterization of last frame
    img_min_x = 0 # [m] Minimum value of the raster in x direction
    img_max_x = 0.5 # [m] Maximum value of the raster in x direction
    img_min_y = 0.2 # [m] Minimum value of the raster in y direction
    img_max_y = 0.3 # [m] Maximum value of the raster in y direction

    ############################################################################################################
    