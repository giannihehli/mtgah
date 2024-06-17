import os
import cv2
from glob import glob

def extract(dir, files):
    # define video and paths
    input_files = dir + files

    # change working directory to data storage
    os.chdir(dir)
    print("input_files: ", input_files)

    for vid_path in glob(input_files):
        # get video name
        vid = os.path.splitext(os.path.basename(vid_path))[0]

        print("Processing video: ", vid)

        # create output folder and save frames or skip if already exists
        try:
            os.mkdir(vid)
        except FileExistsError:
            #directory already exists
            print(f"Directory {vid} already exists. {vid} has already been processed.")
            continue

        # open video
        vidcap = cv2.VideoCapture(vid_path)
        success,image = vidcap.read()
        count = 100

        # loop for the whole video
        while success:
            # save frame as JPEG file
            cv2.imwrite(vid + "/" + vid + "_%d.jpg" % count, image)
            success, image = vidcap.read()
            print("Save frame ", count, ": ", success)
            count += 1
    return

if __name__ == "__main__":
    ####################################################################################
    # ONLY SECTION TO ADJUST PARAMETERS

    # Define the data path for the frame extraction
    data_path = 'G:/data/pipeline_tests/camera/'

    # Define the video file to extract frames from - "*.MP4" for all files
    file = "f_r4-pa_d114_h105_7.MP4" 

    ####################################################################################
    
    # Extract frames from video
    extract(data_path, file)