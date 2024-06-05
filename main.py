# IMPORT PACKAGES AND MODULES
import os
import cv2
import glob
import time
import numpy as np
import pandas as pd
from pyntcloud import PyntCloud
import matplotlib.pyplot as plt

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
from measureTop import convert
from alignPointcloud import getimage
from alignPointcloud import sortpoints
from alignPointcloud import calculate_transformation
from alignPointcloud import transform
from alignPointcloud import rasterize
from alignPointcloud import export

if __name__ == '__main__':
    ############################################################################################################
    # ONLY SECTION TO ADJUST PARAMETERS

    # Define path with data to be analysed
    data_path = 'G:/data/pipeline_tests/'
#    data_path = 'G:/experiments/20240604/'

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

    ############################################################################################################

    # Calibrate camera
    if os.path.isfile(f'{data_path}camera/calibration/K.txt'):
        print(f'Camera calibrated. Loading calibration parameters from {data_path}camera/calibration/.')
        K = np.loadtxt(f'{data_path}camera/calibration/K.txt')  # calibration matrix[3x3]
        d = np.loadtxt(f'{data_path}camera/calibration/d.txt')  # distortion coefficients[5x1]
    elif os.path.isfile(f'{data_path}camera/calibration.mp4'):
        print(f'Camera not calibrated. Calibrating with {data_path}camera/calibration.mp4.')
        rep, K, d, rvec, tvec, X_W = calibratevideo(f'{data_path}camera/', skip_frames)
    else:
        print(f'Camera not calibrated and no calibration file found - approximated parameters used from G:/data/pipeline_tests/camera/calibration/')
        K = np.loadtxt(f'G:/data/pipeline_tests/camera/calibration/K.txt')  # calibration matrix[3x3]
        d = np.loadtxt(f'G:/data/pipeline_tests/camera/calibration/d.txt')  # distortion coefficients[5x1]

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
        roughness = int(basis[1])
        direction = basis.split('-')[1]
        diameter = exp.split('_')[2]
        height = exp.split('_')[3]
        attempt = exp.split('_')[4]

        # Define experiment layout
        if layout == 'f':
                layout = 'flat'
        elif layout == 'i':
                layout = 'inclined'

        # Define reference pattern in clockwise order in world frame (3D) in [0.1mm]
        match basis:
            case 'r0-pe': # changed from earlier pa
                pattern = 100 * np.array([[0.9, 0.84, 0], [8.96, 0.82, 0], [8.95, 8.92, 0], [0.87, 8.93, 0],
                                        [50.87, 0.82, 0], [58.95, 0.82, 0], [58.9, 8.91, 0], [50.83, 8.9, 0],
                                        [50.74, 51.07, 0], [58.83, 51.02, 0], [58.88, 59.1, 0], [50.8, 59.14, 0],
                                        [0.86, 51.02, 0], [8.94, 51.07, 0], [8.88, 59.17, 0], [0.79, 59.11, 0]])
            case 'r0-pa': # changed from earlier pe
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
        points_right = np.empty(0)
        points_left = np.empty(0)
        points_bottom = np.empty(0)
        velocity_right = np.array(0)
        velocity_left = np.array(0)
        velocity_bottom = np.array(0)
        time_frame = np.empty(0)

        for frame in range(frame_count):
            # Start time measurement
            timestamp_start = time.time()

            # Load image
            print(f'Analyse frame {frame}')
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

            # Save image of wanted experiment in folder
            if exp == exp_out:
                print(f'Saving {frame+100}_undst')
                cv2.imwrite(f'{data_path}camera/{exp_out}/{frame+100}_undst.png', img_undst)

            # Detect markers
            img_det, corners, ids = detect(img_undst, marker)
            
            # Define used corners so that wrongly detected corners are filtered out
            corners = corners[0:16]

            # Save image of wanted experiment in folder
            if exp == exp_out:
                print(f'Saving {frame+100}_det')
                cv2.imwrite(f'{data_path}camera/{exp_out}/{frame+100}_det.png', img_det)

            # Warp perspective
            img_warp, M = warp(corners, pattern, img_undst)

            # Save image of wanted experiment in folder
            if exp == exp_out:
                print(f'Saving {frame+100}_warp')
                cv2.imwrite(f'{data_path}camera/{exp_out}/{frame+100}_warp.png', img_warp)

            # Threshold image
            _ , img_thr = threshold(img_warp, kernel_size, sigma_color, sigma_space, filter_threshold) # img_thr_gb, img_thr_bf

            # Save image of wanted experiment in folder
            if exp == exp_out:
                print(f'Saving {frame+100}_thr')
                cv2.imwrite(f'{data_path}camera/{exp_out}/{frame+100}_thr.png', img_thr)

            # Measure distances
            x_right, x_left, y_bottom, img_mes = measure(img_warp, img_thr, search_width)

            # Save image of wanted experiment in folder
            if exp == exp_out:
                print(f'Saving {frame+100}_mes')
                cv2.imwrite(f'{data_path}camera/{exp_out}/{frame+100}_mes.png', img_mes)

            # If needed save images to folder
            if img_out:
                print(f'Saving img{img_out}')
                cv2.imwrite(f'{output_folder}{frame+100}{img_out}.png', locals().get(f'img{img_out}'))

            # Append measured parameters to lists
            points_right = np.append(points_right, x_right)
            points_left = np.append(points_left, x_left)
            points_bottom = np.append(points_bottom, y_bottom)
            time_frame = np.append(time_frame, frame/fps)

            # Calculate velocity with conversion factor to [mm/s]
            if frame >= 1:
                velocity_right = np.append(velocity_right, (points_right[frame]-points_right[frame-1]) * 0.1 * fps)
                velocity_left = np.append(velocity_left, (points_left[frame]-points_left[frame-1]) * -0.1 * fps)
                velocity_bottom = np.append(velocity_bottom, (points_bottom[frame] - points_bottom[frame-1]) * 0.1 * fps)

            # Show and output last frame as .tiff file
            if frame == frame_count-1:

                # Measure the top distance in last frame    
                distance_top, img_mes_top = measuretop(img_warp, img_thr, search_width)

                # Define output path for last frame
                frame_output = f'{data_path}end frames/'

                try: 
                    os.mkdir(frame_output)
                    print('Directory end frames created and last frame saved as threshold.')
                except FileExistsError:
                    print('Directory end frames already exists but last frame saved as threshold.')
                    
                # Save last frame as .tiff file
                cv2.imwrite(f'{frame_output}{exp}_endframe.tiff', img_thr)

                # Convert last frame to needed data structure for asc export
                img_z, img_x, img_y = convert(img_thr)

                # Define image raster size in [m]
                raster_size_img = 0.0001 # [m] for image with 6000 pixels with the 0.6m basis

                # Export last frame as ascii file
                export(img_z, img_x, img_y, raster_size_img, f'{frame_output}{exp}_endframe.asc')           
            
            # Stop time measurement
            timestamp_end = time.time()

            # Calculate needed time per frame
            time_per_frame = timestamp_end - timestamp_start

            # Print time per frame
#            print(f'Time per frame: {time_per_frame} s')

        # Release video capture
        cap.release()

        print('Released video - starting with data processing.')

        # Get final diameter
        diameter_vertical = 0.1 * (points_bottom[-1] - distance_top)
        diameter_horizontal = 0.1 * (points_right[-1] - points_left[-1])

        # Get initial distance
        points_right_cor = min(points_right)
        points_left_cor = max(points_left)
        points_bottom_cor = min(points_bottom)

        # Set initial distance to zero with conversion factor to [mm]
        distance_right = 0.1 * (points_right - points_right_cor)
        distance_left = -0.1 * (points_left - points_left_cor)
        distance_bottom = 0.1 * (points_bottom - points_bottom_cor)
        distance_horizontal = 0.5 * (distance_right + distance_left)
        velocity_horizontal = 0.5 * (velocity_right + velocity_left)

        # Calculate radius from ongoing measurement or from last frame for vertical data in [mm]
        radius_horizontal = 0.1 * (points_right - points_left)/2
        radius_vertical = (distance_bottom + (diameter_vertical/2 - max(distance_bottom)))

        print(f'Mean difference between distance horizontal and radius horizontal: {np.mean(np.array(distance_horizontal) - np.array(radius_horizontal))}')
        print(f'Mean difference between distance vertical and radius vertical: {np.mean(np.array(distance_bottom) - np.array(radius_vertical))}')

        # Filter velocity values with mean filter
        filter_size = 29
        velocity_right = np.convolve(velocity_right, np.ones(filter_size)/filter_size, mode='same')
        velocity_left = np.convolve(velocity_left, np.ones(filter_size)/filter_size, mode='same')
        velocity_bottom = np.convolve(velocity_bottom, np.ones(filter_size)/filter_size, mode='same')
        velocity_horizontal = np.convolve(velocity_horizontal, np.ones(filter_size)/filter_size, mode='same')
    
        # Save measured parameters to dataframe
        dict = {'time': time_frame, 'distance_left': distance_left, 'distance_right': distance_right, 
                'velocity_left': velocity_left, 'velocity_right': velocity_right, 
                'distance_horizontal': distance_horizontal, 'velocity_horizontal': velocity_horizontal, 
                'distance_bottom': distance_bottom, 'velocity_bottom': velocity_bottom,
                'radius_horizontal': radius_horizontal, 'radius_vertical': radius_vertical}
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
        diameter_tot.append(int(diameter[1:]))
        height_tot.append(int(height[1:]))
        attempt_tot.append(attempt)
        d_vertical_tot.append(diameter_vertical)
        d_horizontal_tot.append(diameter_horizontal)

        print(f'Saved camera measurements at {data_path}camera/raw_data/{exp}_raw.csv - starting with orientation of pointcloud.')

        # Load the .ply file of the processed experiment
        cloud = PyntCloud.from_file(f'{data_path}scanner/{exp}.ply')

        # Get the image and indices
        img_ptc, indices = getimage(cloud)

        # Try making the directory for the pointcloud images
        try:
            os.mkdir(f'{data_path}scanner/images')
            print(f'Directory scanner/images created. All pointcloud images saved there.')
        except FileExistsError:
            pass

        # Save the pointcloud image
        cv2.imwrite(f'{data_path}scanner/images/{exp}_ptc.jpg', img_ptc)

        # Detect ArUco markers on pointcloud image
        ptc_det, ptc_corners, ptc_ids = detect(img_ptc, marker)

        # Save the image with detected markers
        cv2.imwrite(f'{data_path}scanner/images/{exp}_det.jpg', ptc_det)
        
        # Sort the detected corners and target points
        source_ptc, target_ptc = sortpoints(ptc_corners, ptc_ids, pattern, aruco_count)   

        # Calculate the transformation matrix from the detected corners and the measured pattern
        M = calculate_transformation(source_ptc, target_ptc)

        # Transform the pointcloud to correct origin
        cloud_corr = transform(cloud, M)

        # Rasterise the corrected point cloud
        max_z, x_edges, y_edges = rasterize(cloud_corr, raster_size, raster_min_x, raster_max_x, raster_min_y, raster_max_y)

        # Define the output path
        ptc_output = f'{data_path}rasters/{exp}_raster.asc'

        # Try making the directory for rasters
        try:
            os.mkdir(f'{data_path}rasters')
            print(f'Directory rasters created. All rasters saved there.')
        except FileExistsError:
            pass

        # Export pointcloud data as ascii file
        export(max_z, x_edges, y_edges, raster_size, ptc_output)

        print(f'Saved pointcloud raster at {ptc_output}.')

    # Save total measured parameters to dataframe
    dict_tot = {'layout': layout_tot, 'basis': basis_tot, 'roughness': roughness_tot, 'direction': direction_tot, 'diameter': diameter_tot, 
                'height': height_tot, 'attempt': attempt_tot, 'd_vertical': d_vertical_tot, 'd_horizontal': d_horizontal_tot}
    df_tot = pd.DataFrame(dict_tot)

    # Plot total measured parameters
    plottotal(data_path, df_tot)

    # Save total measured parameters to csv file
    df_tot.to_csv(f'{data_path}camera/raw_data/total_raw.csv')

    # Close all open windows
    plt.close('all')