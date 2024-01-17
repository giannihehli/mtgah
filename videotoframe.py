import os
import cv2
from glob import glob

<<<<<<< Updated upstream
# define directories and paths
dir = "H:\\data\\tests\\"
vid = "C0020"
vid_path = vid + ".MP4"
=======
def videotoframe(dir):
    # define video and paths
    input_files = dir + "/*.MP4"
>>>>>>> Stashed changes

    # change working directory to data storage
    os.chdir(dir)

    for vid_path in glob(input_files):
        # get video name
        vid = os.path.splitext(os.path.basename(vid_path))[0]

        print("Processing video: ", vid)

<<<<<<< Updated upstream
# loop for the whole video
while success:
    # save frame as JPEG file
    cv2.imwrite(vid + "\\" "%d.jpg" % count, image)
    success, image = vidcap.read()
    print("Read frame ", count, ": ", success)
    count += 1
=======
        # create output folder and save frames or skip if already exists
        try:
            os.mkdir(vid)
            # open video
            vidcap = cv2.VideoCapture(vid_path)
            success,image = vidcap.read()
            count = 0

            # loop for the whole video
            while success:
                # save frame as JPEG file
                cv2.imwrite(vid + "/" "%d.jpg" % count, image)
                success, image = vidcap.read()
                print("Save frame ", count, ": ", success)
                count += 1
        except FileExistsError:
            #directory already exists
            print("Directory ", vid, " already exists. ", vid, " has already been processed.")
            pass

if __name__ == "__main__":
    videotoframe("H:/data/calibration/gopro1")
>>>>>>> Stashed changes
