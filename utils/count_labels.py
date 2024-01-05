import os

path = 'MSASL'

count = 0
count_vids = 0
count_frames = 0

label_map = {}

for folder in os.listdir(path):
    if len(os.listdir(os.path.join(path,folder)))>0:
        label_map[folder]=count
        count+=1
        count_vids+=len(os.listdir(os.path.join(path,folder)))
        for video in os.listdir(os.path.join(path,folder)):
            count_frames+=len(os.listdir(os.path.join(path,folder,video)))

print(count_vids/(count+1)) # average videos per label
print(count_frames/count_vids) # average frames per video
print(label_map)