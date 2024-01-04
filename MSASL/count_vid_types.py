# Script to check some information about the dataset
import json
import os
import numpy as np

destination = 'MP_Data'
msasl_train_data = 'raw_videos_trainset'
msasl_train_json = json.load(open('MSASL_test.json'))
# converted = np.load("converted.npy")

# print('jQb9NL9_S6U' in converted)

mkv_count = 0
mp4_count = 0
webm_count = 0


for vid in os.listdir(msasl_train_data):
    if vid.split('.')[-1] == 'mkv':
        mkv_count+=1
    elif vid.split('.')[-1] == 'mp4':
        mp4_count+=1
    elif vid.split('.')[-1] == 'webm':
        webm_count+=1
    else:
        print(vid)


print("MKV:",mkv_count)
print("MP4:",mp4_count)
print("WEBM:",webm_count)


# freq_per_vid = {}
# for entry in msasl_train_json:
#     vid_id = entry['url'][-11:]
#     if vid_id in freq_per_vid.keys():
#         freq_per_vid[vid_id]+=1
#     else:
#         freq_per_vid[vid_id]=1
# unique_count = len(freq_per_vid)
# print(unique_count)


# count_more_than_one = 0
# for entry in freq_per_vid:
#     if freq_per_vid[entry] > 1:
#         count_more_than_one+=1

# print(count_more_than_one)
