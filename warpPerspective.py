# Importing libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Importing user-defined modules
from detectMarkers import detect


def warp(corners, pattern, image):
    # Find the homography matrix
    # Assume corners and pattern are arrays of points in the source and destination images
    H, _ = cv2.findHomography(corners, pattern, cv2.RANSAC, 5)

    # Wrap the image
    img_warp = cv2.warpPerspective(image, H, (int(6000), int(6000)), 
                                   cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    return img_warp, H

if __name__ == "__main__":
    ####################################################################################
    # ONLY SECTION TO ADJUST PARAMETERS

    # Define paths
    data_path = 'data/'
    img_name = 'undst.png'
    
    # Define used basis - if new basis is used, please measure and define pattern below
    basis =  'r2-pa' # 'r0-pe', 'r0-pa', 'r2-pe', 'r2-pa', 'r4-pe', 'r4-pa'

    ####################################################################################
    
    # Define reference pattern in clockwise order in world frame (3D) in [0.1mm]
    match basis:
      case 'r0-pe': # changed from earlier pa
          pattern = 100 * np.array([[0.9, 0.84, 0], [8.96, 0.82, 0], 
                                    [8.95, 8.92, 0], [0.87, 8.93, 0],
                                    [50.87, 0.82, 0], [58.95, 0.82, 0], 
                                    [58.9, 8.91, 0], [50.83, 8.9, 0],
                                    [50.74, 51.07, 0], [58.83, 51.02, 0], 
                                    [58.88, 59.1, 0], [50.8, 59.14, 0],
                                    [0.86, 51.02, 0], [8.94, 51.07, 0], 
                                    [8.88, 59.17, 0], [0.79, 59.11, 0]])
      case 'r0-pa': # changed from earlier pe
          pattern = 100 * np.array([[59.07, 0.87, 0], [59.18, 8.96, 0], 
                                    [51.07, 8.95, 0], [50.98, 0.86, 0],
                                    [59.18, 50.87, 0], [59.13, 58.93, 0], 
                                    [51.06, 58.9, 0], [51.09, 50.82, 0],
                                    [8.91, 50.74, 0], [8.96, 58.83, 0], 
                                    [0.88, 58.88, 0], [0.85, 50.8, 0],
                                    [8.89, 0.86, 0], [8.92, 8.94, 0], 
                                    [0.82, 8.88, 0], [0.81, 0.8, 0]])
      case 'r2-pa':
          pattern = 100 * np.array([[1.06, 0.82, 0], [9.15, 0.82, 0], 
                                    [9.12, 8.92, 0], [1.05, 8.92, 0],
                                    [50.7, 0.91, 0], [58.77, 0.94, 0], 
                                    [58.75, 9.04, 0], [50.66, 9, 0],
                                    [50.57, 51.09, 0], [58.63, 50.99, 0], 
                                    [58.73, 59.07, 0], [50.66, 59.14, 0],
                                    [1.08, 51.02, 0], [9.15, 51.02, 0], 
                                    [9.12, 59.13, 0], [1.04, 59.11, 0]])
      case 'r2-pe':
          pattern = 100 * np.array([[59.1, 1.06, 0], [59.12, 9.15, 0], 
                                    [51.02, 9.12, 0], [51, 1.05, 0],
                                    [59.04, 50.7, 0], [58.99, 58.77, 0], 
                                    [58.88, 58.75, 0], [50.96, 50.66, 0],
                                    [8.87, 50.57, 0], [8.91, 58.63, 0], 
                                    [0.85, 58.73, 0], [0.81, 50.66, 0],
                                    [8.9, 1.08, 0], [8.91, 9.15, 0], 
                                    [0.81, 9.12, 0], [0.86, 1.04, 0]])
      case 'r4-pa':
          pattern = 100 * np.array([[1.3, 0.82, 0], [9.38, 0.83, 0], 
                                    [9.33, 8.94, 0], [1.24, 8.92, 0],
                                    [50.39, 0.89, 0], [58.46, 0.93, 0], 
                                    [58.41, 9.04, 0], [50.32, 8.98, 0],
                                    [50.33, 50.98, 0], [58.42, 50.92, 0], 
                                    [58.48, 59.00, 0], [50.4, 59.04, 0],
                                    [1.23, 51.01, 0], [9.28, 50.96, 0], 
                                    [9.29, 59.06, 0], [1.24, 59.08, 0]])
      case 'r4-pe':
          pattern = 100 * np.array([[59.08, 1.3, 0], [59.07, 9.38, 0], 
                                    [50.97, 9.33, 0], [51.0, 1.24, 0],
                                    [59.0, 50.39, 0], [58.93, 58.46, 0], 
                                    [50.83, 58.41, 0], [50.92, 50.32, 0],
                                    [8.9, 50.33, 0], [8.92, 58.42, 0], 
                                    [0.89, 58.48, 0], [0.83, 50.4, 0],
                                    [8.97, 1.23, 0], [8.98, 9.28, 0], 
                                    [0.87, 9.29, 0], [0.89, 1.24, 0]])

    # Load undistorted image
    img_undst = cv2.imread(data_path + img_name)

    # Detect markers
    marker = "DICT_4X4_50" 
    img_det, corners, ids = detect(img_undst, marker) 
    cv2.imshow('img_det', cv2.resize(img_det, (1920, 1080))) # Detected Capture
    cv2.waitKey(0)

    # Warp the image
    img_warp, H = warp(corners, pattern, img_undst)
    cv2.imshow('img_warp', cv2.resize(img_warp, (1080, 1080))) # Transformed Capture
    cv2.waitKey(0)

    # Save warped image
    print('Save detected image: ', 'warp_' + img_name)
    cv2.imwrite(data_path + 'warp_' + img_name, img_warp)  

    # Check transformation result by comparing detected corners with transformed corners
    rpe = 0
    img_warp_det, corners_warp, ids_warp = detect(img_warp, marker)
    cv2.imshow('img_warp_det', cv2.resize(img_warp_det, (1080, 1080))) # Detected Capture
    cv2.waitKey(0)
    print("Transformation matrix: \n", H)
    for i in range(len(corners)):
        print('corners ', i, ': ', [corners[i][0], corners[i][1]])
        img_pts = H.dot([corners[i][0], corners[i][1], 1])/H[2, :].dot([corners[i][0], 
                                                                        corners[i][1], 1])
        print('img_pts ', i, ': ', img_pts[0:2])
        print('corner_warp ', i, ': ', corners_warp[i][0], corners_warp[i][1])
        rpe += np.linalg.norm([corners_warp[i][0], corners_warp[i][1]] - img_pts[0:2])
    
    # Calculate reprojection error
    rpe = rpe/len(pattern)
    print('RPE in [px]: ', rpe)
    print('RPE in [mm]: ', rpe/10)

    # Save warped and detected images
    print('Save detected image: det_warp')
    cv2.imwrite(data_path + 'det_warp.png', img_warp_det)