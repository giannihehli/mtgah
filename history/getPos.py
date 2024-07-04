# This script was supposed to detect QR-Codes in the corner or find the corner of the basis 
# through classic corner detection such as Harris. However, it did not work.

# Import libraries and packages
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from undistortImage import undistortImage

def find_template(img, template):
    # Apply template Matching
    img2 = img.copy()
    # All the 6 methods for comparison in a list
    methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
                'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
    w, h = template.shape[::-1]

    for meth in methods:
        img = img2.copy()
        method = eval(meth)
        # Apply template Matching
        res = cv.matchTemplate(img2,template,method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc

        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv.rectangle(img,top_left, bottom_right, 255, 2)
        plt.subplot(121),plt.imshow(res,cmap = 'gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(img,cmap = 'gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle(meth)
        plt.show()
        cv.waitKey(0)


def getPos(img, img_gray):
    print("Get position of the basis...")
    img_32 = np.float32(img_gray)

    # find Harris corners
    dst = cv.cornerHarris(img_32, 11, 11, 0.04)

    # result is dilated for marking the corners, not important
    dst = cv.dilate(dst,None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst>0.1*dst.max()]=[0,0,255]
    cv.imshow('dst',img)
    cv.imwrite("H:/data/tests/position.jpg", img)


def apply_filters(img):
    # Apply different filters
    laplacian = cv.Laplacian(img,cv.CV_64F)
    sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
    sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=5)

    # Plot results
    plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
    plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == "__main__":
    # Define camera and data
    camera = "sony" # "sony", "gopro1", "gopro2
    img_path = "H:/data/tests/"

    # Import calibration parameters
    K = np.loadtxt("H:/data/calibration/" + camera + "/K.txt")  # calibration matrix[3x3]
    d = np.loadtxt("H:/data/calibration/" + camera + "/d.txt")  # distortion coefficients[2x1]

    # Undistort images
    #img_und = undistortImage(K, d, img_path, "DSC00213")
    vid_und = undistortImage(K, d, img_path, "DSC00212")

    # Read image
    img = cv.imread(img_path + vid_und)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Check if image is read
    assert img is not None, "file could not be read, check with os.path.exists()"

    # Apply template matching
    template = cv.imread(img_path + "template_tl.jpg", cv.IMREAD_GRAYSCALE)
    find_template(img_gray, template)

    # Get position
    #getPos(img_path + img_und)
    getPos(img, img_gray)

    # Apply filters
    #apply_filters(img_gray)