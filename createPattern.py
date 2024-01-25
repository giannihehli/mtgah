# Import libraries and packages
import numpy as np
from PIL import Image as im

# Define pattern creation function
def createPattern(pattern, pattern_type, height, width):
    print("Pattern: ", pattern)

    # Create numpy array from scratch
    array = np.ones(width * height, dtype=np.uint8)
    array = np.reshape(array, (height, width))

    pattern_height = int(height/pattern.shape[0])
    pattern_width = int(width/pattern.shape[1])

    # Fill array with pattern
    for i in range(0, pattern.shape[0]):
        for j in range(0, pattern.shape[1]):
            array[i*pattern_height:(i+1)*pattern_height, j*pattern_width:(j+1)*pattern_width] = pattern[i, j]*255

    image = im.fromarray(array)
    image.show()
    image.save("H:/data/patterns/" + pattern_type + "_pattern.png")

if __name__ == "__main__":
    # Define size of image
    height = 1200
    width = 1600

    # Define pattern type
    pattern_type = "chessboard" # "finder", "alignment", chessboard
    
    # Define pattern
    match pattern_type:
        case "finder":
            pattern = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                [1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                                [1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
                                [1, 1, 0, 1, 0, 0, 0, 1, 0, 1],
                                [1, 1, 0, 1, 0, 0, 0, 1, 0, 1],
                                [1, 1, 0, 1, 0, 0, 0, 1, 0, 1],
                                [1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
                                [1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
                                )
        case "alignment":
            pattern = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                                [0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
                                [0, 1, 0, 1, 0, 1, 1, 1, 1, 0],
                                [0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
                                [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
                                )
        case "chessboard":
            pattern = np.array([[0, 1, 0, 1, 0, 1, 0, 1],
                                [1, 0, 1, 0, 1, 0, 1, 0],
                                [0, 1, 0, 1, 0, 1, 0, 1],
                                [1, 0, 1, 0, 1, 0, 1, 0],
                                [0, 1, 0, 1, 0, 1, 0, 1],
                                [1, 0, 1, 0, 1, 0, 1, 0]])
    createPattern(pattern, pattern_type, height, width)
