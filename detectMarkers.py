# import the necessary packages
import numpy as np
import cv2
import sys

def detect(image, marker):

    # define names of each possible ArUco tag OpenCV supports
    ARUCO_DICT = {
        'DICT_4X4_50': cv2.aruco.DICT_4X4_50,
        'DICT_4X4_100': cv2.aruco.DICT_4X4_100,
        'DICT_4X4_250': cv2.aruco.DICT_4X4_250,
        'DICT_4X4_1000': cv2.aruco.DICT_4X4_1000,
        'DICT_5X5_50': cv2.aruco.DICT_5X5_50,
        'DICT_5X5_100': cv2.aruco.DICT_5X5_100,
        'DICT_5X5_250': cv2.aruco.DICT_5X5_250,
        'DICT_5X5_1000': cv2.aruco.DICT_5X5_1000,
        'DICT_6X6_50': cv2.aruco.DICT_6X6_50,
        'DICT_6X6_100': cv2.aruco.DICT_6X6_100,
        'DICT_6X6_250': cv2.aruco.DICT_6X6_250,
        'DICT_6X6_1000': cv2.aruco.DICT_6X6_1000,
        'DICT_7X7_50': cv2.aruco.DICT_7X7_50,
        'DICT_7X7_100': cv2.aruco.DICT_7X7_100,
        'DICT_7X7_250': cv2.aruco.DICT_7X7_250,
        'DICT_7X7_1000': cv2.aruco.DICT_7X7_1000
        }

    # verify that the supplied ArUCo tag exists and is supported by OpenCV
    if ARUCO_DICT.get(marker, None) is None:
        print("[INFO] ArUCo tag of '{}' is not supported".format(
            marker))
        sys.exit(0)
        
    # load the ArUCo dictionary and grab the ArUCo parameters
    arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[marker])
    arucoParams = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

    # copy image
    frame = image.copy()

    # detect ArUco markers in the input frame
    (corners, ids, _) = detector.detectMarkers(frame)

    # do subpixel detection
    corners_sub = []
    for corner in corners:
        term = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        refined_corner = cv2.cornerSubPix(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 
                                          np.float32(corner), (5,5), (-1,-1), term)
        corners_sub.append(refined_corner)
        
#    print('corners detect = ', corners)
#    print('corners_sub detect = ', corners_sub)

    ## Display detection result
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

            # calculate thickness for drawing
            thickness = max(abs(bottomRight[0] - topLeft[0]), 
                            abs(bottomLeft[0] - topRight[0]))
            line_thickness = max(1, thickness//50)

            # draw the bounding box of the ArUco marker
            cv2.line(frame, topLeft, topRight, (0, 255, 0), line_thickness)
            cv2.line(frame, topRight, bottomRight, (0, 255, 0), line_thickness)
            cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), line_thickness)
            cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), line_thickness)

            # draw the top left corner (x, y)-coordinates of the ArUco marker
            cX = int(topLeft[0])
            cY = int(topLeft[1])
            cv2.circle(frame, (cX, cY), line_thickness, (255, 0, 0), -1)

            # draw the ArUco marker ID on the frame
            cv2.putText(frame, str(markerID),
                (topLeft[0] + (bottomRight[0] - topLeft[0])//4, 
                 bottomRight[1] - (bottomRight[1] - topLeft[1])//4),
                cv2.FONT_HERSHEY_SIMPLEX,
                line_thickness, (0, 0, 255), line_thickness)

    # show the output frame
#    cv2.imshow('frame', cv2.resize(frame, (1080, 1080)))
#    cv2.waitKey(0)

    # Sort corners and ids in correct order
    corners_sort, ids_sort = sortcorners(corners_sub, ids)    

    return frame, corners_sort, ids_sort

def sortcorners(corners_sub, ids):
    
    # Create numpy arrays of lists of corners and ids
    corners_sub = np.array(corners_sub)
    ids = np.array(ids)

    # Get the sort indices according to the ids
    sort_indices = np.argsort(ids)

    # Sort the corners and ids so that they are in increasing order
    corners_sort = corners_sub[sort_indices]
    ids_sort = ids[sort_indices]

    # Reshape the corners_sort array to a 2D array
    corners_sort = corners_sort.reshape(4*len(ids), 2)

    return corners_sort, ids_sort

if __name__ == '__main__':
    ####################################################################################
    # ONLY SECTION TO ADJUST PARAMETERS
    
    # Define camera
    camera = 'sony_hs' # 'sony_hs', 'sony', 'gopro1', 'gopro2

    # Define marker
    marker = 'DICT_4X4_50'

    # Define paths
    data_path = 'data/'
    img_name = 'undst.png'

    ####################################################################################

    # Read image
    image = cv2.imread(data_path + img_name)

    # Detect markers
    img_det, corners, ids = detect(image, marker)

    # Show detection result
    cv2.imshow('image', cv2.resize(img_det, (1920, 1080)))
    cv2.waitKey(0)

    # Save undistorted images
    print('Save detected image: ', 'det_' + img_name)
    cv2.imwrite(data_path + 'det_' + img_name, img_det)