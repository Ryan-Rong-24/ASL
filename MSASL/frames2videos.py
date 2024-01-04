# preprocessing script for WLASL dataset
# 1. Convert .swf, .mkv file to mp4.
# 2. Extract YouTube frames and create video instances.

import os
import json
import cv2
import numpy as np

import shutil

VIDEO_PATH = 'raw_videos_frames'
CROPPED_VIDEO_PATH = 'cropped_videos_new'

def convert_frames_to_video(frame_array, path_out, size, fps=25):
    out = cv2.VideoWriter(path_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()

def extract_all_yt_instances(content):
    freq_per_vid = {}
    for entry in content:
        vid_id = entry['url'][-11:]
        if vid_id in freq_per_vid.keys():
            freq_per_vid[vid_id]+=1
        else:
            freq_per_vid[vid_id]=1

    # os.chdir(VIDEO_PATH)

    i=0
    while i < len(content):
        vid_id = content[i]['url'][-11:]
        FOLDER_PATH = os.path.join(VIDEO_PATH,vid_id)

        for j in range(1,freq_per_vid[vid_id]+1):
            if os.path.exists(FOLDER_PATH):

                dst_video_path = os.path.join(CROPPED_VIDEO_PATH,vid_id+str(j)+".mp4")

                if os.path.exists(dst_video_path):
                    print(f'{dst_video_path} exists')
                    i+=1
                    continue

                start_frame = content[i]['start']
                end_frame = content[i]['end'] 
                fps = content[i]['fps']

                # image array
                frames = []

                for frame_num in range(0,len(os.listdir(os.path.join(FOLDER_PATH)))):
                    if frame_num >= start_frame and frame_num <= end_frame:
                        image = str(frame_num) + ".jpg"
                        frames.append(cv2.imread(os.path.join(FOLDER_PATH,image),cv2.IMREAD_COLOR))
                
                # when OpenCV reads an image, it returns size in (h, w, c)
                # when OpenCV creates a writer, it requres size in (w, h).
                
                if len(frames) != 0:
                    size = frames[0].shape[:2][::-1]

                    
                    convert_frames_to_video(frames, dst_video_path, size, fps)

                    print(i, dst_video_path)
            i+=1

        
def main():

    content = json.load(open('MSASL_val.json'))
    extract_all_yt_instances(content)


if __name__ == "__main__":
    main()

