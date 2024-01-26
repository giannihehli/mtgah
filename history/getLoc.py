# This script was supposed to detect chessboard patterns in the cornes of the basis. However, it did not work.

# import all packages and libraries
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from undistortImage import undistortImage

# define function to get location of the basis
def getLoc(img):
    print("get location of the basis...")

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points
    objp = np.zeros((2*2, 39), np.float32)
    print(objp)
    objp[:,:2] = np.mgrid[0:2, 0:2].T.reshape(-1,2)
    print(objp)

    # arrays to store object points and image points from image
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane

    # convert image
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # find the chess board corners
    ret, corners = cv.findChessboardCorners(img_gray, (2,2), None)

    # if found, add object points, image points (after refining them)
    if ret == True:
        print("corners: ", corners)
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(img_gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # draw and display the corners
        img = cv.drawChessboardCorners(img, (2,2), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    else:
        print("No corners found.")

def getHarris(img):
    # convert image
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    corners = cv.goodFeaturesToTrack(img_gray,25,0.01,10)
    corners = np.int0(corners)

    for i in corners:
        x,y = i.ravel()
        cv.circle(img,(x,y),3,255,-1)

    plt.imshow(img),plt.show()
    cv.waitKey(0)

if __name__ == "__main__":
    # Define and import image
    img_path = "H:/data/tests/C0029 - Trim/"
    img_name ="C0029 - Trim_0"
    img = cv.imread(img_path + img_name + ".jpg")

    #getLoc(img)
    getHarris(img)