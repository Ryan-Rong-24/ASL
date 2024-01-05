import json
import os
import shutil

root = 'C:\\Users\\ryanr\\ASLtoEng'
destination = os.path.join(root,'MP_Data_MSASL')
dataset = "test"
msasl_train_data = f'cropped_videos_{dataset}set'
msasl_train_json = json.load(open(f'MSASL_{dataset}.json'))
synonym = json.load(open("MSASL_synonym.json"))


selected_words = json.load(open("MSASL_classes.json"))

freq_per_vid = {}
freq_cur_vid = {}
for entry in msasl_train_json:
    vid_id = entry['url'][-11:]
    if vid_id in freq_per_vid.keys():
        freq_per_vid[vid_id]+=1
    else:
        freq_per_vid[vid_id]=1
        freq_cur_vid[vid_id]=1

i=0
while i < len(msasl_train_json):
    vid_id = msasl_train_json[i]['url'][-11:]
    label = msasl_train_json[i]['clean_text']

    if label not in selected_words:
        # Find synonym
        for arr in synonym:
            if label in arr:
                temp = arr[0]
                break

        print(f'{label} found synonym {temp}')
        label = temp
    
    if not os.path.exists(os.path.join(msasl_train_data,vid_id+str(freq_cur_vid[vid_id])+".mp4")):
        print(f'{vid_id} does not exist')
        i+=1
        freq_cur_vid[vid_id]+=1
        continue

    os.makedirs(os.path.join(destination,label),exist_ok=True)

    video_path = os.path.join(msasl_train_data,vid_id+str(freq_cur_vid[vid_id])+".mp4")
    copy_path = os.path.join(destination,label)
    print(video_path + " " + label + " -> " + copy_path)
    freq_cur_vid[vid_id]+=1

    shutil.copy(video_path,copy_path)
    i+=1

