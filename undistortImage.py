import numpy as np
import cv2
import os

def undistort(K, d, image):

    height,  width = image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, d, (width,height), 1, (width,height))

    # undistort
    img_undst = cv2.undistort(image, K, d, None, newcameramtx)

    return img_undst

if __name__ == "__main__":

    camera = "sony" # "sony", "gopro1", "gopro2
    # Import calibration parameters
    K = np.loadtxt("C:/Users/hehligia/OneDrive - ETH Zurich/Documents/Code/masterthesis/calibration/" + camera + "/K.txt")  # calibration matrix[3x3]
    d = np.loadtxt("C:/Users/hehligia/OneDrive - ETH Zurich/Documents/Code/masterthesis/calibration/" + camera + "/d.txt")  # distortion coefficients[2x1]

    # Define paths
    data_path = "H:/data/calibration/" + camera + "/"
    img_path = "H:/data/calibration/" + camera + "/216"

    # Load image
    image = cv2.imread(img_path)

    # Undistort images
    img_undst = undistort(K, d, image)

    # Save undistorted images
    print("Save undistorted image: ", "calib_" + os.path.basename(img_path))
    cv2.imwrite(data_path + "calib_" + os.path.basename(img_path) + ".PNG", img_undst)