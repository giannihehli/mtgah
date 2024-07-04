# Import needed packages
import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pycalib.plot import plotCamera

def calibrate(camera, data_path):

    # Chessboard variables for how many corners and size
    rows = 12
    cols = 8
    size = 0.0194272727272727 # [m] Physical size of a cell

    # Create object points for the chessboard
    objp = np.zeros((rows*cols,3), np.float32)
    
    # Fill the tuples just generated with their values
    objp[:,:2] = size*np.mgrid[0:rows,0:cols].T.reshape(-1, 2)

    # Initialise image size
    imageSize = None # Determined at runtime

    # Create arrays used to store object points and image points from all images processed
    objpoints = [] # 3D point in real world space where chess squares are
    imgpoints = [] # 2D point in image plane, determined by CV2

    # Count variables
    count_found = 0
    count_failed = 0


    # Define inputs files and output path
    input_files = data_path + "/*.jpg"
    output_path = data_path
    #input_files = "H:/data/calibration/" + camera + "/*.jpg"
    #output_path = "H:/data/calibration/" + camera
    
    # Get all image pathes
    images = glob.glob(input_files)

    try:
        os.mkdir(f'{data_path}calibration')
        print(f'Directory {data_path}calibration created.')
    except FileExistsError:
        pass
    

    # Loop through images glob"ed
    for image_path in images:
        # Open the image
        img = cv2.imread(image_path)
        # Grayscale of the image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find chessboard in the image, setting PatternSizeto a tuple of (#rows, #columns)
        ret, corners = cv2.findChessboardCorners(gray, (rows,cols), None)

        # If a chessboard was found, let"s collect image/corner points
        if ret == True:
            count_found += 1
            print("Detection successful : ", os.path.basename(image_path))
            # Add the points in 3D that we just discovered
            objpoints.append(objp)
            
            # Enhance corner accuracy with cornerSubPix
            corners_acc = cv2.cornerSubPix(
                    image=gray, 
                    corners=corners, 
                    winSize=(5, 5), 
                    zeroZone=(-1, -1),
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
            imgpoints.append(corners_acc)

            # If our image size is unknown, set it now
            if not imageSize:
                imageSize = (gray.shape[1], gray.shape[0])
                print("Image size: ", imageSize)
        
            # Draw the corners to a new image to show whoever is performing the calibration
            # that the board was properly detected
            img = cv2.drawChessboardCorners(img, (rows, cols), corners_acc, ret)
            # Pause to display each image, waiting for key press
            #cv2.imshow("Chessboard", img)
            #cv2.waitKey(0)
            try:
                os.mkdir(output_path + "calibration/detected")
            except FileExistsError:
                pass
            
            cv2.imwrite(output_path+"calibration/detected/det_"+os.path.basename(image_path), 
                        img)
        
        else:     # if not found
            count_failed += 1
            print("Detection failed : ", os.path.basename(image_path))
            continue 

    # Destroy any open CV windows
    cv2.destroyAllWindows()

    # Make sure at least one image was found
    if len(images) < 1:
        # Calibration failed because there were no images, warn the user
        print("Calibration was unsuccessful - no images of chessboards were found.")
        # Exit for failure
        exit()

    # Make sure we were able to calibrate on at least one chessboard by checking
    # if we ever determined the image size
    if not imageSize:
        # Calibration failed because we didn"t see any chessboards of the PatternSize used
        print(f'Calibration was unsuccessful -  could not detect chessboards in any frame.')
        # Exit for failure
        exit()
    
    # Now that we"ve seen all of our images, perform the camera calibration
    # based on the set of points we"ve discovered
    rpe, K, d, rvec, tvec = cv2.calibrateCamera(
            objectPoints=objpoints,
            imagePoints=imgpoints,
            imageSize=imageSize,
            cameraMatrix=None,
            distCoeffs=None,
            flags=cv2.CALIB_FIX_ASPECT_RATIO)
        
    # Save values to be used where matrix+dist is required
    print("Calibration successful / found: ", count_found, " / failed: ", count_failed)
    print("Reprojection error (RPE): ", rpe)
    np.savetxt(data_path + "calibration/K.txt", K)
    print("Saved intrinsic parameter K = ", K)
    np.savetxt(data_path + "calibration/d.txt", d)
    print("Saved Distortion parameters d = (k1, k2, p1, p2, k3) = ", d)
    
    # Plot calibration
    plot_calibration(rvec, tvec, objp, data_path, rpe)
    
    return rpe, K, d, rvec, tvec, objp

def plot_calibration(rvec, tvec, objp, data_path, rpe):
    # plotCamera() config
    plot_mode   = 0 # 0: fixed camera/moving chessboard, 1: fixed chessboard/moving camera
    plot_range_pos  = 0.4 # target volume [plot_range_neg:plot_range_pos]
    plot_range_neg  = -0.2 # target volume [plot_range_neg:plot_range_pos]
    camera_size = 0.03  # size of the camera in plot

    # 3D PLOT
    fig = plt.figure()
    fig.suptitle(f'Camera calibration from {data_path}*.jpg\nRPE: {rpe:.5f}')
    fig.show()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    
    ax.set_xlim(plot_range_neg, plot_range_pos)
    ax.set_ylim(plot_range_neg, plot_range_pos)
    ax.set_zlim(0, 2*plot_range_pos)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title('Camera Position and chessboards in 3D space')

    if plot_mode == 0: # fixed camera = plot in CCS
        
        plotCamera(ax, np.eye(3), np.zeros((1,3)), color="b", scale=camera_size)

        for i_ex in range(len(rvec)):
            X_C = np.zeros((objp.shape))
            for i_x in range(objp.shape[0]):
                R_w2c = cv2.Rodrigues(rvec[i_ex])[0] # convert to the rotation matrix
                t_w2c = tvec[i_ex].reshape(3)
                X_C[i_x,:] = R_w2c.dot(objp[i_x,:]) + t_w2c # Transform from WCS to CCS
                    
            ax.plot(X_C[:,0], X_C[:,1], X_C[:,2], ".") # plot chess corners in CCS

    elif plot_mode == 1: # fixed chessboard = plot in WCS
        
        for i_ex in range(len(rvec)):
            R_c2w = np.linalg.inv(cv2.Rodrigues(rvec[i_ex])[0]) # Camera orientation in WCS
            t_c2w = -R_c2w.dot(tvec[i_ex]).reshape((1,3)) # Camera position in WCS
            
            plotCamera(ax, R_c2w, t_c2w, color="b", scale=camera_size)
            print("Plot camera", i_ex, "at", t_c2w)

        ax.plot(objp[:,0], objp[:,1], objp[:,2], ".")
    
    ax.view_init(azim=45, elev=-160, roll=0)
#    plt.show()
#    cv2.waitKey(0)
    plt.savefig(data_path + "calibration/calib_result.pdf")
    
    return

if __name__ == "__main__":
    ####################################################################################
    # ONLY SECTION TO ADJUST PARAMETERS

    # Camera selection
    camera = "sony_hs" # "sony", "sony_hs" "gopro1", "gopro2

    # Define data path
    data_path = "G:/data/tests/calibration/" + camera + "/"

    ####################################################################################

    rpe, K, d, rvec, tvec, objp = calibrate(camera, data_path)