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
    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    arucoParams = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

    # resize image
    frame = imutils.resize(image, width=1000)

    # detect ArUco markers in the input frame
    (corners, ids, rejected) = detector.detectMarkers(frame)

    # Draw a square around the markers
    cv2.aruco.drawDetectedMarkers(frame, corners)

    # verify *at least* one ArUco marker was detected
    if len(corners) > 0:
        # flatten the ArUco IDs list
        ids = ids.flatten()

        # loop over the detected ArUCo corners
        for (markerCorner, markerID) in zip(corners, ids):
            # extract the marker corners (which are always returned
            # in top-left, top-right, bottom-right, and bottom-left
            # order)
            markerCorner = markerCorner.reshape((4, 2))

            (topLeft, topRight, bottomRight, bottomLeft) = markerCorner

            # convert each of the (x, y)-coordinate pairs to integers
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            # draw the bounding box of the ArUCo detection
            """ cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2) """

            # compute and draw the center (x, y)-coordinates of the
            # ArUco marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(frame, (cX, cY), 1, (0, 0, 255), -1)

            cX = int(topLeft[0])
            cY = int(topLeft[1])
            cv2.circle(frame, (cX, cY), 1, (255, 0, 0), -1)
            
            # draw the ArUco marker ID on the frame
            cv2.putText(frame, str(markerID),
                (topLeft[0], topLeft[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)

    # show the output frame
    #cv2.imshow("Frame", frame)
    #cv2.waitKey(0)
    cv2.imwrite("H:/data/aruco/C0031 - Trim_0_detected.JPG", frame)

    return frame, corners, ids

if __name__ == "__main__":

    camera = "sony" # "sony", "gopro1", "gopro2
    # Import calibration parameters
    K = np.loadtxt("calibration/" + camera + "/K.txt")  # calibration matrix[3x3]
    d = np.loadtxt("calibration/" + camera + "/d.txt")  # distortion coefficients[2x1]

    marker = "DICT_4X4_50"
    image = cv2.imread("H:/data/aruco/C0031 - Trim_0.JPG")
    img_det, corners, ids = detect(image, marker)