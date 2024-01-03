import cv2
import os
root = 'C:\\Users\\ryanr\\ASLtoEng\\'
MP_DATA_PATH = os.path.join(root,'MP_data')

for label in os.listdir(MP_DATA_PATH):
    LABEL_PATH = os.path.join(MP_DATA_PATH,label)
    for filename in os.listdir(LABEL_PATH):
        # print(LABEL_PATH)
        if filename.endswith('.mp4'):
            print(f'Converting {filename} from video to frames...')

            os.chdir(LABEL_PATH)
            video_name = filename.split('.')[0]
            os.makedirs(video_name,exist_ok=True)

            vidcap = cv2.VideoCapture(filename)
            success, image = vidcap.read()
            count = 0

            os.chdir(os.path.join(LABEL_PATH,video_name))

            while success:
                cv2.imwrite("%d.jpg" % count, image)     # save frame as JPEG file      
                success,image = vidcap.read()
                # print('Read a new frame: ', success)
                count += 1

            print('Converted.')
            
            # close the video
            vidcap.release()

            # delete the video after converting
            os.chdir(LABEL_PATH)
            os.remove(filename)

