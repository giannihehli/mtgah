# IMPORT PACKAGES AND MODULES
import os
import cv2
import glob
import numpy as np
import pandas as pd
from termcolor import colored
from pyntcloud import PyntCloud

# IMPORT USER-DEFINED MODULES
from calibrateCameravideo import calibratevideo
from undistortImage import undistort
from detectMarkers import detect
from warpPerspective import warp
from filterImage import threshold
from measureDistance import measure
from plotParameters import plotparams
from plotExperiments import plottotal
from measureTop import measuretop
from alignPointcloud import getimage
from alignPointcloud import calculate_transformation
from alignPointcloud import transform
from alignPointcloud import rasterize
from alignPointcloud import export

############################################################################################################
# ONLY SECTION TO ADJUST PARAMETERS

# Define path with data to be analysed
#data_path = 'H:/data/tests/sony_hs/'
data_path = 'G:/experiments/20240531/'

#######################################
# CAMERA OPTIONS
# Define used camera
camera = 'sony_hs' # 'sony_hs', 'sony', 'gopro1', 'gopro2

# Input factor for skipping frames in calibration video
skip_frames = 10  # [] Number of frames to skip in calibration video

# Define images for optional output of respective images
img_out = '' # Options: '', _undst', '_det', '_warp', '_thr', '_mes'

# Define exp_out for optional output of all respective experiment images
exp_out = '' # Options: '', 'f_r8_d113_h40', 

# Define gaussian blur kernel size for Gaussian blur in/and bilateral filter (45)
kernel_size = 35 # [px] must be positive and odd

# Define bilateral filter parameters (200, 25)
sigma_color = 80 # [px] Filter sigma in the color space. A larger value of the parameter means that farther colors within the pixel neighborhood (see sigmaSpace) will be mixed together, resulting in larger areas of semi-equal color.
sigma_space = 35 # [px] Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels will influence each other as long as their colors are close enough (see sigmaColor ). When d>0, it specifies the neighborhood size regardless of sigmaSpace. Otherwise, d is proportional to sigmaSpace.

# Define filter threshold for binary thresholding (90)
filter_threshold = 90 # [px] Threshold value for binary thresholding - 90 for 250fps and 110 for 500fps

# Define ROI width for measurement
search_width = 600 # [px] Width of the ROI for distance measurement

#######################################
# SCANNER OPTIONS

# Define raster size in m for rasterization of scanned pointcloud
raster_size = 0.001 # [m] Size of one bin in the raster in x and y direction

# Define raster min and max values in m
raster_min = 0 # [m] Minimum value of the raster in x and y direction
raster_max = 0.6 # [m] Maximum value of the raster in x and y direction

############################################################################################################

# Calibrate camera
if os.path.isfile(f'{data_path}camera/calibration/K.txt'):
    print(colored(f'Camera calibrated. Loading calibration parameters from {data_path}camera/calibration/.', 'green'))
    K = np.loadtxt(f'{data_path}camera/calibration/K.txt')  # calibration matrix[3x3]
    d = np.loadtxt(f'{data_path}camera/calibration/d.txt')  # distortion coefficients[5x1]
elif os.path.isfile(f'{data_path}camera/calibration.mp4'):
    print(colored(f'Camera not calibrated. Calibrating with {data_path}camera/calibration.mp4.', 'orange'))
    rep, K, d, rvec, tvec, X_W = calibratevideo(f'{data_path}camera/', skip_frames)
else:
    print(colored(f'Camera not calibrated and no calibration file found - approximated parameters used from H:/data/camera/calibration/{camera}/', 'red'))
    K = np.loadtxt(f'H:/data/camera/calibration/{camera}/K.txt')  # calibration matrix[3x3]
    d = np.loadtxt(f'H:/data/camera/calibration/{camera}/d.txt')  # distortion coefficients[5x1]

# Initialize lists for measured distances in every experiment
layout_tot = []
basis_tot = []
roughness_tot = []
direction_tot = []
diameter_tot = []
height_tot = []
attempt_tot = []
d_vertical_tot = []
d_horizontal_tot = []

# Loop through all videos in data path
for vid_path in glob.glob(data_path + 'camera/' + '*.MP4'):
        # Check if video is already processed and skip if so
    if vid_path.split('\\')[-1] == 'calibration.MP4':
        print('Calibration video - skip processing.')
        continue    

    # Load video in video capture
    cap = cv2.VideoCapture(vid_path)
    
    # Get experiment name
    exp = os.path.splitext(os.path.basename(vid_path))[0]
    
    # Print processed experiment
    print(f'Processing experiment: {exp}')

    # Get needed video information
    fps = cap.get(cv2.CAP_PROP_FPS) # [frames/s]
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # [frames]

    # Define used parameters according to experiment name
    layout = exp.split('_')[0]
    basis = exp.split('_')[1]
    roughness = basis[1]
    direction = basis.split('-')[1]
    diameter = exp.split('_')[2]
    height = exp.split('_')[3]
    attempt = exp.split('_')[4]

    # Define experiment layout
    if layout == 'f':
            layout = 'flat'
    elif layout == 'i':
            layout = 'inclined'

#    print(f'Processing frames: {exp} with layout: {layout}, basis: {basis}, diameter: {diameter}, height: {height}')
    # Define reference pattern in clockwise order in world frame (3D) in [0.1mm]
    match basis:
        case 'r0-pa':
            pattern = 100 * np.array([[0.9, 0.84, 0], [8.96, 0.82, 0], [8.95, 8.92, 0], [0.87, 8.93, 0],
                                     [50.87, 0.82, 0], [58.95, 0.82, 0], [58.9, 8.91, 0], [50.83, 8.9, 0],
                                     [50.74, 51.07, 0], [58.83, 51.02, 0], [58.88, 59.1, 0], [50.8, 59.14, 0],
                                     [0.86, 51.02, 0], [8.94, 51.07, 0], [8.88, 59.17, 0], [0.79, 59.11, 0]])
        case 'r0-pe':
            pattern = 100 * np.array([[59.07, 0.87, 0], [59.18, 8.96, 0], [51.07, 8.95, 0], [50.98, 0.86, 0],
                                     [59.18, 50.87, 0], [59.13, 58.93, 0], [51.06, 58.9, 0], [51.09, 50.82, 0],
                                     [8.91, 50.74, 0], [8.96, 58.83, 0], [0.88, 58.88, 0], [0.85, 50.8, 0],
                                     [8.89, 0.86, 0], [8.92, 8.94, 0], [0.82, 8.88, 0], [0.81, 0.8, 0]])
        case 'r4-pa': # originally 'r8' so new 48-pa needs to be measured
            pattern = 10 * np.array([[12.6, 12.8, 0], [98.5, 12.3, 0], [98.7, 98.5, 0], [13.1, 98.8, 0],
                                [499.2, 12.7, 0], [585.3, 12.8, 0], [585.1, 98.7, 0], [499.1, 98.6, 0],
                                [499.5, 501.2, 0], [585.5, 501.1, 0], [585.6, 587.1, 0], [499.6, 587.1, 0],
                                [12.7, 501.2, 0], [98.3, 501.2, 0], [98.4, 587.5, 0], [12.6, 587.3, 0]])
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
#        case 'r8-pa':
 #           pattern = 100 * np.array([[, , 0], [, , 0], [, , 0], [, , 0],
  #                                   [, , 0], [, , 0], [, , 0], [, , 0],
   #                                  [, , 0], [, , 0], [, , 0], [, , 0],
    #                                 [, , 0], [, , 0], [, , 0], [, , 0]])
     #   case 'r8-pe':
      #      pattern = 100 * np.array([[, , 0], [, , 0], [, , 0], [, , 0],
       #                              [, , 0], [, , 0], [, , 0], [, , 0],
        #                             [, , 0], [, , 0], [, , 0], [, , 0],
         #                            [, , 0], [, , 0], [, , 0], [, , 0]])

    # Define initial radius according to diameter         
    r_initial = int(diameter[1:])/2

    # Define initial height according to height
    h_initial = int(height[1:])
            
    #Define used marker type
    marker = 'DICT_4X4_1000'

    # Make directory for output images
    if img_out:
        output_folder = f'{data_path}camera/{exp}/'
        try: 
            os.mkdir(output_folder)
            print('Directory ', output_folder, ' created.')
        except FileExistsError:
            print('Directory ', output_folder, ' already exists - images saved again.')

    # Initialize lists for measured distances per experiment
    distance_right = []
    distance_left = []
    distance_bottom = []
    velocity_right = [0]
    velocity_left = [0]
    velocity_bottom = [0]
    time = []

    for frame in range(frame_count):
        # Load image
        print(colored(f'Analyse frame {frame}', 'blue'))
        ret, image = cap.read()

        # Stop video analysis if no frame is loaded anymore
        if ret == False:
            break

        # If needed save images to folder
        if exp == exp_out:
            try:
                os.mkdir(f'{data_path}camera/{exp_out}')
                print(f'Directory {exp_out} created. All images of this experiment saved there.')
            except FileExistsError:
                pass
            print(f'Saving {frame+100}_orig')
            cv2.imwrite(f'{data_path}camera/{exp_out}/{frame+100}_orig.png', image)

        # Undistort images
        img_undst = undistort(K, d, image)
#        cv2.imshow('image_undist', cv2.resize(img_undst, (1920, 1080)))
#        cv2.waitKey(0)

        # Save image of wanted experiment in folder
        if exp == exp_out:
            print(f'Saving {frame+100}_undst')
            cv2.imwrite(f'{data_path}camera/{exp_out}/{frame+100}_undst.png', img_undst)

        # Detect markers
        img_det, corners, ids = detect(img_undst, marker)
#        cv2.imshow('image_det', cv2.resize(img_det, (1920, 1080)))
#        cv2.waitKey(0)

        # Save image of wanted experiment in folder
        if exp == exp_out:
            print(f'Saving {frame+100}_det')
            cv2.imwrite(f'{data_path}camera/{exp_out}/{frame+100}_det.png', img_det)

        # Warp perspective
        img_warp, M = warp(corners, pattern, img_undst)
#        cv2.imshow('image_warp', cv2.resize(img_warp, (1080, 1080)))
#        cv2.waitKey(0)

        # Save image of wanted experiment in folder
        if exp == exp_out:
            print(f'Saving {frame+100}_warp')
            cv2.imwrite(f'{data_path}camera/{exp_out}/{frame+100}_warp.png', img_warp)

        # Threshold image
        _ , img_thr = threshold(img_warp, kernel_size, sigma_color, sigma_space, filter_threshold) # img_thr_gb, img_thr_bf
#        cv2.imshow('image_thr', cv2.resize(img_thr, (1080, 1080)))
#        cv2.waitKey(0)

        # Save image of wanted experiment in folder
        if exp == exp_out:
            print(f'Saving {frame+100}_thr')
            cv2.imwrite(f'{data_path}camera/{exp_out}/{frame+100}_thr.png', img_thr)

        # Measure distances
        x_right, x_left, y_bottom, img_mes = measure(img_warp, img_thr, search_width)
#        cv2.imshow('image_mes', cv2.resize(img_mes, (1080, 1080)))
#        cv2.waitKey(0)

        # Save image of wanted experiment in folder
        if exp == exp_out:
            print(f'Saving {frame+100}_mes')
            cv2.imwrite(f'{data_path}camera/{exp_out}/{frame+100}_mes.png', img_mes)

        # If needed save images to folder
        if img_out:
            print(f'Saving img{img_out}')
            cv2.imwrite(f'{output_folder}{frame+100}{img_out}.png', locals().get(f'img{img_out}'))

        # Append measured parameters to lists
        distance_right.append(x_right)
        distance_left.append(x_left)
        distance_bottom.append(y_bottom)
        time.append(frame/fps)

        # Calculate velocity with conversion factor to [mm/s]
        if frame >= 1:
            velocity_right.append((distance_right[frame]-distance_right[frame-1]) * 0.1 * fps)
            velocity_left.append((distance_left[frame]-distance_left[frame-1]) * -0.1 * fps)
            velocity_bottom.append((distance_bottom[frame] - distance_bottom[frame-1]) * 0.1 * fps)

        # Show and output last frame as .tiff file
        if frame == frame_count-1:
            # Show last frame
#            cv2.imshow(str(frame), cv2.resize(img_thr, (1080, 1080)))
#            cv2.waitKey(0)

            try: 
                os.mkdir(f'{data_path}end frame tiffs')
                print('Directory end frame tiffs created and last frame saved as threshold.')
                cv2.imwrite(f'{data_path}end frame tiffs/{exp}.tiff', img_thr)
            except FileExistsError:
                print('Directory end frame tiffs already exists but last frame saved as threshold.')
                cv2.imwrite(f'{data_path}end frame tiffs/{exp}.tiff', img_thr)
                
            distance_top, img_mes_top = measuretop(img_warp, img_thr, search_width)


    # Release video capture
    cap.release()

    # Get final diameter
    diameter_vertical = distance_bottom[-1] - distance_top
    diameter_horizontal = distance_right[-1] - distance_left[-1]

    # Get initial distance
    distance_right_cor = min(distance_right)
    distance_left_cor = max(distance_left)
    distance_bottom_cor = min(distance_bottom)

    # Set initial distance to zero with conversion factor to [mm]
    distance_right = [0.1 * (distance - distance_right_cor) for distance in distance_right]
    distance_left = [-0.1 * (distance - distance_left_cor) for distance in distance_left]
    distance_bottom = [0.1 * (distance - distance_bottom_cor) for distance in distance_bottom]
    radius_horizontal = [0.5 * (distance_right[i] + distance_left[i]) for i in range(len(distance_right))]
    velocity_horizontal = [0.5 * (velocity_right[i] + velocity_left[i]) for i in range(len(velocity_right))]

    # Filter velocity values with mean filter
    filter_size = 29
    velocity_right = np.convolve(velocity_right, np.ones(filter_size)/filter_size, mode='same')
    velocity_left = np.convolve(velocity_left, np.ones(filter_size)/filter_size, mode='same')
    velocity_bottom = np.convolve(velocity_bottom, np.ones(filter_size)/filter_size, mode='same')
    velocity_horizontal = np.convolve(velocity_horizontal, np.ones(filter_size)/filter_size, mode='same')
   
    # Save measured parameters to dataframe
    dict = {'time': time, 'distance_left': distance_left, 'distance_right': distance_right, 
            'velocity_left': velocity_left, 'velocity_right': velocity_right, 
            'radius_horizontal': radius_horizontal, 'velocity_horizontal': velocity_horizontal, 
            'distance_bottom': distance_bottom, 'velocity_bottom': velocity_bottom}
    df = pd.DataFrame(dict)
 
    # Plot measured parameters
    plotparams(data_path, exp, df, layout, basis, direction, r_initial, h_initial, diameter_vertical, diameter_horizontal)

    # Make folder for raw data
    try:
        os.mkdir(f'{data_path}camera/raw_data')
        print(f'Directory camera/raw_data created. All measurements of this experiment saved there.')
    except FileExistsError:
        pass

    # Save raw parameters to csv file
    df.to_csv(f'{data_path}camera/raw_data/{exp}_raw.csv')

    # Append measured parameters to total list
    layout_tot.append(layout)
    basis_tot.append(basis)
    roughness_tot.append(roughness)
    direction_tot.append(direction)
    diameter_tot.append(int(diameter[1:])/2)
    height_tot.append(int(height[1:]))
    attempt_tot.append(attempt)
    d_vertical_tot.append(diameter_vertical)
    d_horizontal_tot.append(diameter_horizontal)

    # Load the .ply file of the processed experiment
    cloud = PyntCloud.from_file(f'{data_path}scanner/{exp}.ply')

    # Get the image and indices
    img_ptc, indices = getimage(cloud)

    # Show the pointcloud image
    cv2.imshow('img_ptc', img_ptc)
    cv2.waitKey(0)

    # Save the pointcloud image
    cv2.imwrite(f'{data_path}scanner/{exp}_ptc.jpg', img_ptc)

    # Detect ArUco markers on pointcloud image
    ptc_det, ptc_corners, ptc_ids = detect(img_ptc, marker)

    # Show the image with detected markers
    cv2.imshow('ptc_det', ptc_det)
    cv2.waitKey(0)

    # Save the image with detected markers
    cv2.imwrite(f'{data_path}scanner/{exp}_det.jpg', ptc_det)

    # Define corners in 3D as source points
    source_ptc = np.hstack((ptc_corners[:16], np.zeros((16, 1), dtype=ptc_corners.dtype)))

    # Convert the target points to the correct unit [mm]
    target_ptc = 0.1 * pattern

    # Calculate the transformation matrix from the detected corners and the measured pattern
    M = calculate_transformation(source_ptc, target_ptc)

    # Transform the pointcloud to correct origin
    cloud_corr = transform(cloud, M)

    # Rasterise the corrected point cloud
    max_z, x_edges, y_edges = rasterize(cloud_corr, raster_size, raster_min, raster_max, raster_min, raster_max)

    # Define the output path
    output_path = f'{data_path}rasters/{exp}_raster.asc'

    # Try making the directory for rasters
    try:
        os.mkdir(f'{data_path}rasters')
        print(f'Directory rasters created. All rasters saved there.')
    except FileExistsError:
        pass

    # Export data as ascii file
    export(max_z, x_edges, y_edges, raster_size, output_path)

# Save total measured parameters to dataframe
dict_tot = {'layout': layout_tot, 'basis': basis_tot, 'roughness': roughness_tot, 'direction': direction_tot, 'diameter': diameter_tot, 
            'height': height_tot, 'attempt': attempt_tot, 'd_vertical': d_vertical_tot, 'd_horizontal': d_horizontal_tot}
df_tot = pd.DataFrame(dict_tot)

# Plot total measured parameters
plottotal(data_path, df_tot)

# Save total measured parameters to csv file
df_tot.to_csv(f'{data_path}camera/raw_data/total_raw.csv')