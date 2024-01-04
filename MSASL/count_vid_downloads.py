import json
import os

data = json.load(open("MSASL_train.json"))

data_id = []
count = []
video_list = []

for dat in data:
    if dat['url'][-11:] not in data_id:
        data_id.append(dat['url'][-11:])

for video in os.listdir("raw_videos_trainset"):
    video_id = video.split(".")[0]
    if video_id in data_id:
        video_list.append(video_id)
    else:
        print(video_id)

for video in os.listdir("cropped_videos_train"):
    video_id = video.split(".")[0][:11]
    if video_id in data_id and video_id not in count and video_id not in video_list:
        count.append(video_id)
    # else:
    #     print(video_id)

print(len(count))
print(len(data_id))