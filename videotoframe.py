import os
import cv2

# define directories and paths
dir = "H:\\data\\tests\\"
vid = "C0020"
vid_path = vid + ".MP4"

# change working directory to data storage
os.chdir(dir)

# create output folder and video path
try:
    os.mkdir(vid)
except FileExistsError:
    #directory already exists
    pass

# open video
vidcap = cv2.VideoCapture(vid_path)
success,image = vidcap.read()
count = 0

# loop for the whole video
while success:
    # save frame as JPEG file
    cv2.imwrite(vid + "\\" "%d.jpg" % count, image)
    success, image = vidcap.read()
    print("Read frame ", count, ": ", success)
    count += 1