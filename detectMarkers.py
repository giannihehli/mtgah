# import the necessary packages
import numpy as np
import imutils
import cv2
import cv2.aruco as aruco
import sys

def detect(image, marker):

    # define names of each possible ArUco tag OpenCV supports
    ARUCO_DICT = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
        "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
        "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
        "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
        "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
        "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
        "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
        "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
        "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000
        }

    # verify that the supplied ArUCo tag exists and is supported by
    # OpenCV
    if ARUCO_DICT.get(marker, None) is None:
        print("[INFO] ArUCo tag of '{}' is not supported".format(
            marker))
        sys.exit(0)
        
    # load the ArUCo dictionary and grab the ArUCo parameters
    arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[marker])
    arucoParams = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

    # resize image
#    frame = imutils.resize(image, width=1000)
    frame = image.copy()

    # detect ArUco markers in the input frame
    (corners, ids, rejected) = detector.detectMarkers(frame)

    # do subpixel detection
    corners_sub = []
    for corner in corners:
        term = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        refined_corner = cv2.cornerSubPix(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), np.float32(corner), (5,5), (-1,-1), term)
        corners_sub.append(refined_corner)
        
#    print("corners detect = ", corners)
#    print("corners_sub detect = ", corners_sub)

    ## Display detection result
    # Draw a square around the markers
    cv2.aruco.drawDetectedMarkers(frame, corners_sub)

    # verify *at least* one ArUco marker was detected
    if len(corners_sub) > 0:
        # flatten the ArUco IDs list
        ids = ids.flatten()

        # loop over the detected ArUCo corners_sub
        for (markerCorner, markerID) in zip(corners_sub, ids):
            # extract the marker corners_sub (which are always returned
            # in top-left, top-right, bottom-right, and bottom-left
            # order)
            markerCorner = markerCorner.reshape((4, 2))

            (topLeft, topRight, bottomRight, bottomLeft) = markerCorner

            # convert each of the (x, y)-coordinate pairs to integers
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            # compute and draw the center (x, y)-coordinates of the
            # ArUco marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(frame, (cX, cY), 10, (255, 0, 0), -1)

            cX = int(topLeft[0])
            cY = int(topLeft[1])
            cv2.circle(frame, (cX, cY), 10, (255, 0, 0), -1)
            
            # draw the ArUco marker ID on the frame
            cv2.putText(frame, str(markerID),
                (topLeft[0], topLeft[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                5, (0, 255, 0), 10)

    # show the output frame
#    cv2.imshow("frame", cv2.resize(frame, (1080, 1080)))
#    cv2.waitKey(0)

#    print("ids = ", ids)
#    print("corners = ", corners_sub)

    ## Arange corners and ids in clockwise order
    # Initialize corner_sort array with needed array dimension
    corners_sort = np.ones((len(ids)*4, 2))

    # Sort corners and ids in clockwise order
    for i in range(4):
        corners_sort[ids[i]*4:ids[i]*4+4] = corners_sub[i]

#    print("corners_sort = ", corners_sort)

    return frame, corners_sort, ids

if __name__ == "__main__":

    camera = "sony" # "sony", "gopro1", "gopro2
    # Import calibration parameters
    K = np.loadtxt("calibration/" + camera + "/K.txt")  # calibration matrix[3x3]
    d = np.loadtxt("calibration/" + camera + "/d.txt")  # distortion coefficients[2x1]

    marker = "DICT_4X4_50"
    image = cv2.imread("data/DSC00532.JPG")
    img_det, corners, ids = detect(image, marker)
    cv2.imshow("image", cv2.resize(img_det, (1920, 1080)))
    cv2.waitKey(0)