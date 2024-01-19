import numpy as np
import cv2
import glob

def undistortImage(camera, K, d, data_path, img_name):

    img = cv2.imread(data_path + img_name)
    height,  width, channels = img.shape
    new_height = height - 1000
    new_width = width - 1000
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, d, (height,width), 0, (height,width))

    # undistort
    img_undst = cv2.undistort(img, K, d, None, newcameramtx)
    print("Save undistorted image: ", img_name)
    cv2.imwrite(data_path + "/calibresult.png", img_undst)

if __name__ == "__main__":

    camera = "sony" # "sony", "gopro1", "gopro2
    # Import calibration parameters
    K = np.loadtxt("H:/data/calibration/" + camera + "/K.txt")  # calibration matrix[3x3]
    d = np.loadtxt("H:/data/calibration/" + camera + "/d.txt")  # distortion coefficients[2x1]

    # Undistort images
    undistortImage(camera, K, d, "H:/data/tests/" + camera + "/C0024/", "697.JPG")