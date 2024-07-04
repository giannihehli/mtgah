# -*- coding: utf-8 -*-
"""

Created on Mon May 21 15:26:48 2018
@author: Amos
reference: Camera Calibration 
          ("http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials
          /py_calib3d/py_calibration/py_calibration.html#calibration")

"""
import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pycalib.plot import plotCamera
from termcolor import colored

def plot_calibration(rvec, tvec, X_W):
    # plotCamera() config
    plot_mode   = 0    # 0: fixed camera / moving chessboard,  1: fixed chessboard, moving camera
    plot_range  = 0.5 # target volume [-plot_range:plot_range]
    camera_size = 0.03  # size of the camera in plot

    # 3D PLOT
    fig_in = plt.figure()
    fig_in.show()
    ax_in = Axes3D(fig_in, auto_add_to_figure=False)
    fig_in.add_axes(ax_in)
    
    ax_in.set_xlim(-plot_range, plot_range)
    ax_in.set_ylim(-plot_range, plot_range)
    ax_in.set_zlim(-plot_range, plot_range)

    ax_in.set_xlabel('X')
    ax_in.set_ylabel('Y')
    ax_in.set_zlabel('Z')
    ax_in.legend()
    ax_in.set_title('Camera Position and chessboards in 3D space')

    if plot_mode == 0: # fixed camera = plot in CCS
        
        plotCamera(ax_in, np.eye(3), np.zeros((1,3)), color="b", scale=camera_size) # camera is at (0,0,0)

        for i_ex in range(len(rvec)):
            X_C = np.zeros((X_W.shape))
            for i_x in range(X_W.shape[0]):
                R_w2c = cv2.Rodrigues(rvec[i_ex])[0] # convert to the rotation matrix
                t_w2c = tvec[i_ex].reshape(3)
                X_C[i_x,:] = R_w2c.dot(X_W[i_x,:]) + t_w2c # Transform chess corners in WCS to CCS
                    
            ax_in.plot(X_C[:,0], X_C[:,1], X_C[:,2], ".") # plot chess corners in CCS

    elif plot_mode == 1: # fixed chessboard = plot in WCS
        
        for i_ex in range(len(rvec)):
            R_c2w = np.linalg.inv(cv2.Rodrigues(rvec[i_ex])[0]) # Camera orientation in world coordinate system
            t_c2w = -R_c2w.dot(tvec[i_ex]).reshape((1,3)) # Camera position in world coordinate system
            
            plotCamera(ax_in, R_c2w, t_c2w, color="b", scale=camera_size)
            print("Plot camera", i_ex, "at", t_c2w)

        ax_in.plot(X_W[:,0], X_W[:,1], X_W[:,2], ".")
    
    plt.show()
    cv2.waitKey(0)
    plt.savefig("calibration/" + camera + "/result.pdf")

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob("H:/data/calibration/sony/moving2/*.JPG") #read a series of images

camera = "sony"

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Convert the image to gray

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2=cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria) #refine the corner locations
        imgpoints.append(corners2)

        # Draw and display the corners
        
        cv2.drawChessboardCorners(img, (9,6), corners2, ret)
        IMS = cv2.resize(img, (2160, 1440))
        cv2.imshow('img', IMS)
        cv2.waitKey(500)

cv2.destroyAllWindows()

#calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

#save parameters needed in undistortion
np.savez("parameters.npz",mtx=mtx,dist=dist,rvecs=rvecs,tvecs=tvecs)
np.savez("points.npz",objpoints=objpoints,imgpoints=imgpoints)

print('intrinsic matrix=\n', mtx)
print('distortion coefficients=\n', dist)
print('rotation vector for each image=', *rvecs, sep = "\n")
print('translation vector for each image=', *tvecs, sep= "\n")

plot_calibration(rvecs, tvecs, objpoints)