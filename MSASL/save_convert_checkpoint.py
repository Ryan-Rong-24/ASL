import os
import numpy as np

root = 'C:\\Users\\ryanr\\ASLtoEng\\'
SAVE_DATA_PATH = os.path.join(root,'MS-ASL','raw_videos_frames')
SAVE_CONVERT_PATH =  os.path.join(root,'MS-ASL','converted.npy')

if os.path.exists(SAVE_CONVERT_PATH):
    converted = np.load(SAVE_CONVERT_PATH).tolist()
else:
    converted = []

for folder in os.listdir(SAVE_DATA_PATH):
    converted.append(folder)

converted_np = np.array(converted,dtype=str)
np.save(SAVE_CONVERT_PATH,converted_np)