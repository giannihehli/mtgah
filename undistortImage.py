import numpy as np
import cv2
import os

def undistort(K, d, image):

    height,  width = image.shape[:2]

    # Get optimal camera matrix
    newcameramtx, _ = cv2.getOptimalNewCameraMatrix(K, d, (width, height), 1, (width, height))

    # Undistort image
    img_undst = cv2.undistort(image, K, d, None, newcameramtx)

    return img_undst

if __name__ == '__main__':
    ####################################################################################
    # ONLY SECTION TO ADJUST PARAMETERS

    camera = 'sony_hs' # 'sony', 'sony_hs', 'gopro1', 'gopro2'

    # Define paths
    data_path = 'data/'
    img_name = 'orig.png'

    ####################################################################################

    # Import calibration parameters
    K = np.loadtxt('calibration/' + camera + '/K.txt')  # calibration matrix[3x3]
    d = np.loadtxt('calibration/' + camera + '/d.txt')  # distortion coefficients[2x1]

    # Load image
    image = cv2.imread(data_path + img_name)

    # Undistort images
    img_undst = undistort(K, d, image)

    # Save undistorted images
    print('Save undistorted image: ', 'undst_' + img_name)
    cv2.imwrite(data_path + 'undst_' + img_name, img_undst)