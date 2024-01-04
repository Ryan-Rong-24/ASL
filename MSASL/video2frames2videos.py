import cv2
import os
import numpy as np
import json
import shutil

root = 'C:\\Users\\ryanr\\ASLtoEng\\transformer' # Change to your own directory
dataset = "train"
MP_DATA_PATH = os.path.join(root,'MSASL',f'raw_videos_{dataset}set')
SAVE_DATA_PATH = os.path.join(root,'MSASL',f'cropped_videos_{dataset}set')
# converted = np.load(os.path.join(root,'MS-ASL','converted.npy'))
converted = []

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

    i=0
    while i < len(content):
        vid_id = content[i]['url'][-11:]
        print(i)

        if os.path.exists(os.path.join(MP_DATA_PATH,vid_id+'.mp4')):
            print(f'Converting {vid_id}.mp4 from video to frames...')

            # Create new folder
            video_folder = os.path.join(MP_DATA_PATH,vid_id)

            # Check if already converted this file
            if os.path.exists(video_folder) or vid_id in converted or os.path.exists(os.path.join(SAVE_DATA_PATH,vid_id+str(1)+'.mp4')):
                print(f'{vid_id}.mp4 exists')
                i+=1
                continue

            os.makedirs(video_folder,exist_ok=False)

            vidcap = cv2.VideoCapture(os.path.join(MP_DATA_PATH,vid_id+'.mp4'))
            success, image = vidcap.read()
            count = 0

            os.chdir(video_folder)

            # Convert into frames with opencv
            while success:
                cv2.imwrite("%d.jpg" % count, image)     # save frame as JPEG file      
                success,image = vidcap.read()
                # print('Read a new frame: ', success)
                count += 1

            print('Converted.')
            
            # close the video
            vidcap.release()

            # change back to raw_videos folder
            os.chdir(MP_DATA_PATH)

            for j in range(1,freq_per_vid[vid_id]+1): # Handling creating multiple videos from one video
                dst_video_path = os.path.join(SAVE_DATA_PATH,vid_id+str(j)+".mp4")

                if os.path.exists(dst_video_path) or vid_id in converted:
                    print(f'{dst_video_path} exists')
                    i+=1
                    continue

                start_frame = content[i]['start']
                end_frame = content[i]['end'] 
                fps = content[i]['fps']

                # image array
                frames = []

                for frame_num in range(0,len(os.listdir(video_folder))):
                    # Crop the videos according to the start and end frames
                    if frame_num >= start_frame and frame_num <= end_frame:
                        image = str(frame_num) + ".jpg"
                        frames.append(cv2.imread(os.path.join(video_folder,image),cv2.IMREAD_COLOR))
                
                
                if len(frames) != 0:
                    # when OpenCV reads an image, it returns size in (h, w, c)
                    # when OpenCV creates a writer, it requres size in (w, h).
                    size = frames[0].shape[:2][::-1]
                    convert_frames_to_video(frames, dst_video_path, size, fps) # Convert back to videos
                    print(f'Moving the {i}th video {vid_id}.mp4 to {dst_video_path}')

                i+=1

            # Remove frames folder when finished
            print(f'Deleting video folder: {video_folder}')
            shutil.rmtree(video_folder)
        else: # Video does not exist in current directory
            i+=1

def main():
    content = json.load(open(f'MSASL_{dataset}.json'))
    extract_all_yt_instances(content)


if __name__ == "__main__":
    main()
