import os
import cv2
from glob import glob

def videostoframes(dir, files):
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
            # open video
            vidcap = cv2.VideoCapture(vid_path)
            success,image = vidcap.read()
            count = 100

            # loop for the whole video
            while success:
                # save frame as JPEG file
                cv2.imwrite(vid + "/" + vid + "_frame_%d.jpg" % count, image)
                success, image = vidcap.read()
                print("Save frame ", count, ": ", success)
                count += 1
        except FileExistsError:
            #directory already exists
            print("Directory ", vid, " already exists. ", vid, " has already been processed.")
            pass

if __name__ == "__main__":
    videostoframes("G:/data_lars/", "*.MP4") # "*.MP4" for all files