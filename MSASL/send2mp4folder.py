import os
import shutil

raw = 'raw_videos'
raw_mp4 = 'raw_videos_mp4'

for vid in os.listdir(raw):
    if vid.split('.')[-1] == 'mp4':
        cur_path = os.path.join(raw,vid)
        dest_path = os.path.join(raw_mp4,vid)
        shutil.copyfile(cur_path,dest_path)
