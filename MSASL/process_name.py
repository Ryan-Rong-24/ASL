import os 

dir = "raw_videos_testset"

# Rename to __ for processing
# for video in os.listdir(dir):
#     if video.endswith(".mkv") and video.startswith("-"):
#         os.rename(os.path.join(dir,video),os.path.join(dir,"__"+video))

#     if video.endswith(".webm") and video.startswith("-"):
#         os.rename(os.path.join(dir,video),os.path.join(dir,"__"+video))


# Rename back without __ after mp4 conversion

for video in os.listdir(dir):
    if video.startswith("__") and len(video.split(".")[0])==13:
        os.rename(os.path.join(dir,video),os.path.join(dir,video[2:]))
