import os

count =  0
dir = "raw_videos_valset"
for video in os.listdir(dir):
    if os.path.exists(os.path.join(dir,video.split(".")[0]+".webm")) and os.path.exists(os.path.join(dir,video.split(".")[0]+".mp4")):
        os.remove(os.path.join(dir,video.split(".")[0]+".webm"))
        count+=1

    if os.path.exists(os.path.join(dir,video.split(".")[0]+".mkv")) and os.path.exists(os.path.join(dir,video.split(".")[0]+".mp4")):
        os.remove(os.path.join(dir,video.split(".")[0]+".mkv"))
        count+=1

print("Removed:",count)

# count =  0
# video_ids = []
# for video in os.listdir("cropped_videos"):
#     video_id = video[:11]
#     # if os.path.exists(os.path.join("raw_videos_testset",video_id+".webm")) or \
#     #     os.path.exists(os.path.join("raw_videos_testset",video_id+".mkv")) or \
#     #     os.path.exists(os.path.join("raw_videos_testset",video_id+".mp4")):
#     #     continue
#     # else:
#     if video_id not in video_ids:
#         video_ids.append(video_id)


# print(len(video_ids))