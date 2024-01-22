import numpy as np
import cv2
import glob

def undistortImage(camera, K, d, data_path, img_name):

    img = cv2.imread(data_path + img_name + ".JPG")
    height,  width, channels = img.shape
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, d, (width,height), 1, (width,height))

    # undistort
    img_undst = cv2.undistort(img, K, d, None, newcameramtx)
    print("Save undistorted image: ", "calib_" + img_name)
    cv2.imwrite(data_path + "calib_" + img_name + ".PNG", img_undst)

if __name__ == "__main__":

    camera = "gopro1" # "sony", "gopro1", "gopro2
    # Import calibration parameters
    K = np.loadtxt("H:/data/calibration/" + camera + "/K.txt")  # calibration matrix[3x3]
    d = np.loadtxt("H:/data/calibration/" + camera + "/d.txt")  # distortion coefficients[2x1]

    # Undistort images
    undistortImage(camera, K, d, "H:/data/calibration/" + camera + "/", "216")