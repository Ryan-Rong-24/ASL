import cv2
import os
import numpy as np
root = 'C:\\Users\\ryanr\\ASLtoEng\\'
MP_DATA_PATH = os.path.join(root,'MS-ASL','raw_videos_mp4')
SAVE_DATA_PATH = os.path.join(root,'MS-ASL','raw_videos_frames')
converted = np.load(os.path.join(root,'MS-ASL','converted.npy'))

os.chdir(MP_DATA_PATH)

for video in os.listdir(MP_DATA_PATH):
    if video.endswith('.mp4'):
        print(f'Converting {video} from video to frames...')

        video_name = video.split('.')[0]
        video_folder = os.path.join(SAVE_DATA_PATH,video_name)

        if os.path.exists(video_folder) or video_name in converted:
            print(f'{video} exists')
            continue

        os.makedirs(video_folder,exist_ok=False)

        vidcap = cv2.VideoCapture(video)
        success, image = vidcap.read()
        count = 0

        os.chdir(video_folder)

        while success:
            cv2.imwrite("%d.jpg" % count, image)     # save frame as JPEG file      
            success,image = vidcap.read()
            # print('Read a new frame: ', success)
            count += 1

        print('Converted.')
        
        # close the video
        vidcap.release()

        # delete the video after converting
        os.chdir(MP_DATA_PATH)

