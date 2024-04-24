# IMPORT PACKAGES AND MODULES
import sys
import os
import cv2
import glob
import numpy as np
import pandas as pd
from termcolor import colored

# IMPORT USER-DEFINED MODULES
from videostoframes import extract
import calibrateCamera
from undistortImage import undistort
from detectMarkers import detect
from locateCamera import locate
from warpPerspective import warp
from filterImage import threshold
from measureDistance import measure
from plotParameters import plot

# Define used parameters
camera = 'sony' # 'sony', 'gopro1', 'gopro2
calib_path = 'H:/data/calibration/' + camera + '/'
data_path = 'H:/data/tests/sony_hs/'

# Define used marker
marker = 'DICT_4X4_50'

# Define images for output
img_out = None # None _undst', '_det', '_warp', '_thr', '_mes'

# Calibrate camera
if os.path.isfile('calibration/' + camera + '/K.txt'):
    print(colored(f'Camera calibrated. Loading calibration parameters from {calib_path}.', 'green'))
    K = np.loadtxt('calibration/' + camera + '/K.txt')  # calibration matrix[3x3]
    d = np.loadtxt('calibration/' + camera + '/d.txt')  # distortion coefficients[5x1]
else:
    print(colored('Calibrating camera...', 'yellow'))
    rep, K, d, rvec, tvec, X_W = calibrateCamera.calibrate(camera, view=True, check=True)

''' # Extract frames from all videos
extract(data_path, '*.MP4') '''

# Loop through all videos in data path
for vid_path in glob.glob(data_path + '*.MP4'):
    # Load video in video capture
    cap = cv2.VideoCapture(vid_path)
    print(colored(f'Processing video: {vid_path}', 'blue'))

    # Get needed video information
    fps = cap.get(cv2.CAP_PROP_FPS) # [frames/s]
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # [frames]
    
    # Get video name
    vid = os.path.splitext(os.path.basename(vid_path))[0]

    # Define used parameters according to video name
    layout = vid.split('_')[0]
    basis = vid.split('_')[1]
    diameter = vid.split('_')[2]
    height = vid.split('_')[3]

    print(f'Processing frames: {vid} with layout: {layout}, basis: {basis}, diameter: {diameter}, height: {height}')
    # Define reference pattern in clockwise order in world frame (3D) in [0.1mm]
    match basis:
        case 'sm':
            pattern = 10 * np.array([[12.5, 6.2, 0], [99.5, 8, 0], [98.6, 95, 0], [11.5, 93.2, 0],
                                [498.8, 8.2, 0], [586, 8.2, 0], [586.5, 95.4, 0], [499.2, 95, 0],
                                [499.5, 503, 0], [586.7, 503.9, 0], [585.4, 591.2, 0], [498, 590.3, 0],
                                [16.2, 503.1, 0], [103.3, 503.2, 0], [102.7, 590.5, 0], [15.6, 590.5, 0]]
                                )
        case 'r8':
            pattern = 10 * np.array([[12.6, 12.8, 0], [98.5, 12.3, 0], [98.7, 98.5, 0], [13.1, 98.8, 0],
                                [499.2, 12.7, 0], [585.3, 12.8, 0], [585.1, 98.7, 0], [499.1, 98.6, 0],
                                [499.5, 501.2, 0], [585.5, 501.1, 0], [585.6, 587.1, 0], [499.6, 587.1, 0],
                                [12.7, 501.2, 0], [98.3, 501.2, 0], [98.4, 587.5, 0], [12.6, 587.3, 0]]
                                )

    # Make directory for output images
    if img_out:
        output_folder = f'{data_path}{vid}{img_out}'
        try: 
            os.mkdir(output_folder)
            print('Directory ', output_folder, ' created.')
        except FileExistsError:
            print('Directory ', output_folder, ' already exists - images saved again.')
    else:
        print('No images saved to output folder.')

    # Initialize lists for measured distances
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

        if ret == False:
            break

        # Undistort images
        img_undst = undistort(K, d, image)
#        cv2.imshow('image_undist', cv2.resize(img_undst, (1920, 1080)))
#        cv2.waitKey(0)

        # Detect markers
        img_det, corners, ids = detect(img_undst, marker)
#        cv2.imshow('image_det', cv2.resize(img_det, (1920, 1080)))
#        cv2.waitKey(0)

        # Warp perspective
        img_warp, M = warp(corners, pattern, img_undst)
#        cv2.imshow('image_warp', cv2.resize(img_warp, (1080, 1080)))
#        cv2.waitKey(0)

        # Threshold image
        _ , img_thr = threshold(img_warp)
#        cv2.imshow('image_thr', cv2.resize(img_thr, (1080, 1080)))
#        cv2.waitKey(0)

        # Measure distances
        x_right, x_left, y_bottom, img_mes = measure(img_warp, img_thr)
#         cv2.imshow('image_mes', cv2.resize(img_mes, (1080, 1080)))
#        cv2.waitKey(0)

        # If needed save images to folder
        if img_out:
            print(f'Saving img{img_out}')
            cv2.imwrite(f'{output_folder}/{frame+100}{img_out}.jpg', locals().get(f'img{img_out}'))

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

    # Release video capture
    cap.release()

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
    plot(df, vid)

    # Save parameters to csv file
    df.to_csv(f'{data_path}{vid}.csv')
