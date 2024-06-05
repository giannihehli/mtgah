# Resources: 
# - OpenCV-Python tutorial for calibration: http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
#   - Variable names were changed for clarity

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pycalib.plot import plotCamera
from termcolor import colored

def calibratevideo(data_path, skip_frames):

    # Chessboard variables for how many corners and size
    rows = 12
    cols = 8
    size = 0.0194272727272727 # [m] Physical size of a cell (the distance between neighrboring corners). Any positive number works.

    # Theoretical object points for the chessboard we're calibrating against,
    # These will come out like: 
    #     size * (0, 0, 0), (1, 0, 0), ..., 
    #     size * (rows-1, cols-1, 0)
    # Note that the Z value for all stays at 0, as this is a printed out 2D image
    # And also that the max point is -1 of the max because we're zero-indexing
    # The following line generates all the tuples needed at (0, 0, 0)
    objp = np.zeros((rows*cols,3), np.float32)
    
    # The following line fills the tuples just generated with their values size * (0, 0, 0), size * (1, 0, 0), ...
    objp[:,:2] = size*np.mgrid[0:rows,0:cols].T.reshape(-1, 2)

    # All images used should be the same size, which if taken with the same camera shouldn't be a problem
    # I'm using a set of images taken with the camera with the naming convention:
    # 'camera-pic-of-chessboard-<NUMBER>.jpg'
    imageSize = None # Determined at runtime

    # Create arrays you'll use to store object points and image points from all images processed
    objpoints = [] # 3D point in real world space where chess squares are
    imgpoints = [] # 2D point in image plane, determined by CV2

    # Count variables
    count_found = 0
    count_failed = 0

    # Load video in video capture
    cap = cv2.VideoCapture(f'{data_path}calibration.mp4')
    print(colored(f'Processing video: {data_path}calibration.mp4', 'blue'))

    # Get needed video information
    fps = cap.get(cv2.CAP_PROP_FPS) # [frames/s]
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # [frames]

    try:
        os.mkdir(f'{data_path}calibration')
        print(f'Directory {data_path}calibration created.')
    except FileExistsError:
        pass

    # Loop through frames in calibration video
    for frame in range(round(frame_count/skip_frames)):
        # Set frame position to every skip_frames frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame*skip_frames)

        # Load image
        print(f'Analyse frame {skip_frames*frame}')
        ret, img = cap.read()

        if ret == False:
            break

        # Grayscale of the image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # If our image size is unknown, set it now
        if not imageSize:
            imageSize = (gray.shape[1], gray.shape[0])
            print('Image size: ', imageSize)

        # Find chessboard in the image, setting PatternSize(2nd arg) to a tuple of (#rows, #columns)
        ret, corners = cv2.findChessboardCorners(gray, (rows,cols), None)

        # If a chessboard was found, let's collect image/corner points
        if ret == True:
            count_found += 1
            print(colored(f'Detection successful for frame {skip_frames*frame}', 'green'))

            # Add the points in 3D that we just discovered
            objpoints.append(objp)
            
            # Enhance corner accuracy with cornerSubPix
            corners_acc = cv2.cornerSubPix(
                    image=gray, 
                    corners=corners, 
                    winSize=(5, 5), 
                    zeroZone=(-1, -1),
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)) # Last parameter is about termination critera
            
            # Add the accurate points in 2D that we just discovered
            imgpoints.append(corners_acc)
            
            # Draw the corners to a new image to show whoever is performing the calibration
            # that the board was properly detected
            img = cv2.drawChessboardCorners(img, (rows, cols), corners_acc, ret)
            # Pause to display each image, waiting for key press
            #cv2.imshow('Chessboard', img)
            #cv2.waitKey(0)
            try:
                os.mkdir(f'{data_path}calibration/corners')
            except FileExistsError:
                pass
            
            cv2.imwrite(f'{data_path}calibration/corners/{skip_frames*frame+100}_cor.png', img)
        
        else:     # if not found
            count_failed += 1
            print(colored(f'Detection failed for frame {skip_frames*frame}', 'red'))
            continue 

    # Destroy any open CV windows
    cv2.destroyAllWindows()

    # Make sure we were able to calibrate on at least one chessboard by checking
    # if we ever determined the image size
    if not imageSize:
        # Calibration failed because we didn't see any chessboards of the PatternSize used
        print(f'Calibration was unsuccessful. We could not detect chessboards in any of the frames supplied. Try changing the patternSize passed into findChessboardCorners(), or re-do video of chessboard.')
        # Exit for failure
        exit()
    
    # Now that we've seen all of our images, perform the camera calibration
    # based on the set of points we've discovered
    rpe, K, d, rvec, tvec = cv2.calibrateCamera(
            objectPoints=objpoints,
            imagePoints=imgpoints,
            imageSize=imageSize,
            cameraMatrix=None,
            distCoeffs=None,
            flags=cv2.CALIB_FIX_ASPECT_RATIO)
        
    # Save values to be used where matrix+dist is required, for instance for posture estimation
    print('Calibration successful / RPE: ', rpe, ' / found: ', count_found, ' / failed: ', count_failed)
    np.savetxt(f'{data_path}calibration/K.txt', K)
    print('Saved intrinsic parameter K = ', K)
    np.savetxt(f'{data_path}calibration/d.txt', d)
    print('Saved Distortion parameters d = (k1, k2, p1, p2, k3) = ', d)
    
    # Plot calibration
    plot_calibration(rvec, tvec, objp, data_path, rpe)

    return rpe, K, d, rvec, tvec, objp

def plot_calibration(rvec, tvec, objp, data_path, rpe):
    # plotCamera() config
    plot_mode   = 0    # 0: fixed camera / moving chessboard,  1: fixed chessboard, moving camera
    plot_range  = 0.4 # target volume [-plot_range:plot_range]
    camera_size = 0.03  # size of the camera in plot

    # 3D PLOT
    fig = plt.figure()
    fig.suptitle(f'Camera calibration from {data_path}calibration.mp4\nRPE: {rpe:.5f}')
    fig.show()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    
    ax.set_xlim(-plot_range, plot_range)
    ax.set_ylim(-plot_range, plot_range)
    ax.set_zlim(0, 2*plot_range)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Position and chessboards in 3D space')

    if plot_mode == 0: # fixed camera = plot in CCS
        
        plotCamera(ax, np.eye(3), np.zeros((1,3)), color='b', scale=camera_size) # camera is at (0,0,0)
        plt.subplots_adjust(top=0.88, bottom=0.11, left=0.125, right=0.9, hspace=0.2, wspace=0.2)

        for i_ex in range(len(rvec)):
            X_C = np.zeros((objp.shape))
            for i_x in range(objp.shape[0]):
                R_w2c = cv2.Rodrigues(rvec[i_ex])[0] # convert to the rotation matrix
                t_w2c = tvec[i_ex].reshape(3)
                X_C[i_x,:] = R_w2c.dot(objp[i_x,:]) + t_w2c # Transform chess corners in WCS to CCS
                    
            ax.plot(X_C[:,0], X_C[:,1], X_C[:,2], '.') # plot chess corners in CCS

    elif plot_mode == 1: # fixed chessboard = plot in WCS
        plt.subplots_adjust(top=0.88, bottom=0.11, left=0.125, right=0.9, hspace=0.2, wspace=0.2)
        for i_ex in range(len(rvec)):
            R_c2w = np.linalg.inv(cv2.Rodrigues(rvec[i_ex])[0]) # Camera orientation in world coordinate system
            t_c2w = -R_c2w.dot(tvec[i_ex]).reshape((1,3)) # Camera position in world coordinate system
            
            plotCamera(ax, R_c2w, t_c2w, color='b', scale=camera_size)
            print('Plot camera', i_ex, 'at', t_c2w)

        ax.plot(objp[:,0], objp[:,1], objp[:,2], '.')

    ax.view_init(azim=45, elev=-160, roll=0)
    plt.savefig(f'{data_path}calibration/result.pdf')
    plt.show()

if __name__ == '__main__':
    # Camera selection
    camera = 'sony_hs' # 'sony', 'sony_hs' 'gopro1', 'gopro2

    # Define data path
    data_path = 'H:/data/tests/sony_hs/'

    # Input factor for skipping frames in calibration video
    skip_frames = 10    

    rep, K, d, rvec, tvec, objp = calibratevideo(data_path, skip_frames)
