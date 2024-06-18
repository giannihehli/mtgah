import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Function to generate ArUco markers
def generate_aruco(size, code, aruco_dict):
    marker = np.zeros((size, size), dtype=np.uint8)
    aruco_marker = cv2.aruco.generateImageMarker(aruco_dict, code, size, marker, 1)
    return aruco_marker

# Function to create printable image with ArUco markers arranged as specified
def create_aruco_page(marker_size, spacing, aruco_dict_name, aruco_codes):

    # Get ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, aruco_dict_name))

    # Create a blank canvas
    canvas_size = 2 * marker_size + 2*spacing + 1
    canvas = np.ones((canvas_size, canvas_size), dtype=np.uint8) * 255
    
    # Generate ArUco markers
    markers = [generate_aruco(marker_size, code, aruco_dict) for code in aruco_codes]

    # Arrange markers on the canvas
    # ArUco marker 0 (bottom right)
    canvas[int(marker_size + 1.5*spacing +1):int(2*marker_size + 1.5*spacing +1), 
           int(marker_size + 1.5*spacing +1):int(2*marker_size + 1.5*spacing +1)] = markers[0]
    # ArUco marker 1 (bottom left)
    canvas[int(marker_size + 1.5*spacing +1):int(2*marker_size + 1.5*spacing +1), 
           int(0.5*spacing):int(marker_size + 0.5*spacing)] = markers[1]
    # ArUco marker 2 (top left)
    canvas[int(0.5*spacing):int(marker_size + 0.5*spacing), 
           int(0.5*spacing):int(marker_size + 0.5*spacing)] = markers[2]
    # ArUco marker 3 (top right)
    canvas[int(0.5*spacing):int(marker_size + 0.5*spacing), 
           int(marker_size + 1.5*spacing+1):int(2*marker_size + 1.5*spacing+1)] = markers[3]

    # Add separating lines
    line_width = 1
    cross_position = marker_size + spacing
    cv2.line(canvas, (0, cross_position), (2*canvas_size, cross_position), 
             color=0, thickness=line_width)
    cv2.line(canvas, (cross_position, 0), (cross_position, 2*canvas_size), 
             color=0, thickness=line_width)

    return canvas

if __name__ == "__main__":
    ######################################################################################
    # ONLY SECTION TO ADJUST PARAMETERS

    # Size of each ArUco marker in pixels
    marker_size = 1000

    # Spacing between and around markers in pixels
    spacing = 200

    # ArUco code ids to be used. Must be four codes to be output in a 2x2 grid
    aruco_codes = [0, 1, 2, 3]

    # Default ArUco dictionary name. choose from:
    aruco_dict_name = "DICT_4X4_50"

    # "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000", 
    # "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000", 
    # "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
	# "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000", 
    # "DICT_ARUCO_ORIGINAL", "DICT_APRILTAG_16h5", "DICT_APRILTAG_25h9", 
    # "DICT_APRILTAG_36h10", "DICT_APRILTAG_36h11"

    ######################################################################################

    # Generate printable ArUco page
    aruco_page = create_aruco_page(marker_size, spacing, aruco_dict_name, aruco_codes)

    # Try making output folder
    try:
        os.mkdir('output')
    except FileExistsError:
        pass

    # Save the image as a PNG file
    save_path = 'output/' + aruco_dict_name + '.png'
    plt.imsave(save_path, aruco_page, cmap='gray', format='png')

    # Save the image as a PDF file
    save_path = 'output/' + aruco_dict_name + '.pdf'
    plt.imsave(save_path, aruco_page, cmap='gray', format='pdf')

    print(f"ArUco page saved as {save_path}")