# Importing libraries
import numpy as np
import cv2

# Importing user-defined modules
from detectMarkers import detect

def locate(corners, pattern, K, d):
    # Get pose estimation
    ret, rvec, tvec = cv2.solvePnP(pattern, corners, K, d,  cv2.SOLVEPNP_EPNP)
    print(cv2.solvePnP(pattern, corners, K, d,  cv2.SOLVEPNP_EPNP))

    if ret:
        # Refine pose estimation with Levenber-Marquardt iterative minimization
        rvec_ref, tvec_ref = cv2.solvePnPRefineLM(pattern, corners, K, d, rvec, tvec)
        return rvec_ref, tvec_ref
    else:
        print("Pose estimation failed. Try different pattern or camera parameters.")
        exit()

if __name__ == "__main__":
    camera = "sony" # "sony", "gopro1", "gopro2
    # Import calibration parameters
    K = np.loadtxt("C:/Users/hehligia/OneDrive - ETH Zurich/Documents/Code/masterthesis/calibration/" + camera + "/K.txt")  # calibration matrix[3x3]
    d = np.loadtxt("C:/Users/hehligia/OneDrive - ETH Zurich/Documents/Code/masterthesis/calibration/" + camera + "/d.txt")  # distortion coefficients[2x1]

    # Get corners and ids of detected markers
    marker = "DICT_4X4_50"
    image = cv2.imread("H:/data/aruco/C0031 - Trim_0.JPG")
    img_det, corners, ids = detect(image, marker, K, d)

    # Arange corners and ids in clockwise order
    # Initialize corner_sort array with needed array dimension
    corners_sort = np.ones((len(ids)*4, 2))

    # Sort corners and ids in clockwise order
    for i in range(len(ids)):
        corners_sort[ids[i]*4:ids[i]*4+4] = corners[i]

    # Define reference pattern in clockwise order in world frame (3D) in [mm]
    pattern = np.array([[12.5, 6.2, 0], [99.5, 8, 0], [98.6, 95, 0], [11.5, 93.2, 0],
                        [498.8, 8.2, 0], [586, 8.2, 0], [586.5, 95.4, 0], [499.2, 95, 0],
                        [499.5, 503, 0], [586.7, 503.9, 0], [585.4, 591.2, 0], [498, 590.3, 0],
                        [16.2, 503.1, 0], [103.3, 503.2, 0], [102.7, 590.5, 0], [15.6, 590.5, 0]])

    """     # Change datatype to float 32
    corners_sort = corners_sort.astype(np.float32)
    pattern = pattern.astype(np.float32) """

    # Print shapes and types of arrays
    print("corners: ", corners_sort)
    print("pattern: ", pattern)
    print("corners shape: ", corners_sort.shape)
    print("corners type: ", corners_sort.dtype)
    print("pattern shape: ", pattern.shape)
    print("pattern type: ", pattern.dtype)

    rvec, tvec = locate(corners_sort, pattern, K, d)