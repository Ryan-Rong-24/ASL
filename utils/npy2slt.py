import numpy as np
import os
import torch
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# slt format:
# one entry is in format -> {'name':str,'signer':str,'gloss':str,'text':str,'sign':tensor([[]])}

DIR = "../MP_Data_m_noface"

data = []

for folder in tqdm(os.listdir(DIR)):
    for video in os.listdir(os.path.join(DIR,folder)):
        cur = {'name':folder+video}
        sign_stack = []
        for frame in os.listdir(os.path.join(DIR,folder,video)):
            file = np.load(os.path.join(DIR,folder,video,frame))
            print(len(file))
            cur['signer'] = folder+video
            sign_stack.append(file)

        np_sign_stack=np.array(sign_stack)
        sign_tensor = torch.tensor(np_sign_stack)
        cur['sign'] = sign_tensor
        cur['text'] = folder
        cur['gloss'] = folder

        data.append(cur)

# print(data)

# X_train, X_test = train_test_split(data, test_size=0.2, random_state=1)

# X_train, X_val = train_test_split(X_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

# # print(X_train)
# # print(X_test)
# # print(X_val)

# map = {"bye":0,"can":1,"excuse me":2,"french fries":3,"hamburger":4,"hello":5,"help":6,"me":7,"name":8,"sorry":9,"thanks":10,"what":11,"you":12}

# train_info = [0] * 13
# for data in X_train:
#     train_info[map[data['text']]] += 1

# print(train_info)

# test_info = [0] * 13
# for data in X_test:
#     test_info[map[data['text']]] += 1

# print(test_info)

# val_info = [0] * 13
# for data in X_val:
#     val_info[map[data['text']]] += 1

# print(val_info)

# with open('mp_noface.train', 'wb') as handle:
#     pickle.dump(X_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# with open('mp_noface.val', 'wb') as handle:
#     pickle.dump(X_val, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# with open('mp_noface.test', 'wb') as handle:
#     pickle.dump(X_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
