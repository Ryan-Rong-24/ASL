import numpy as np
import os
import torch
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# slt format:
# one entry is in format -> {'name':str,'signer':str,'gloss':str,'text':str,'sign':tensor([[]])}

DATA = "../cached_data/X_data_m_noface_13.npy"
LABELS = "../cached_data/y_data_m_noface_13.npy"

# DATA = "X_test.npy"
# LABELS = "y_test.npy"

# map = {"bye":0,"can":1,"excuseme":2,"frenchfries":3,"hamburger":4,"hello":5,"help":6,"me":7,"name":8,"sorry":9,"thanks":10,"what":11,"you":12}
# new_map = dict([(value, key) for key, value in map.items()])
   
file = np.load(DATA)
file2 = np.load(LABELS)



file2 = np.argmax(file2, axis=1).reshape(-1, 1)

print(file.shape)
print(file2.shape)


X_train, X_test, y_train, y_test = train_test_split(file, file2, test_size=0.1, random_state=42)

np.save("train_data.npy",X_train)
np.save("train_labels.npy",y_train)
np.save("test_data.npy",X_test)
np.save("test_labels.npy",y_test)

print(y_test)


# for i in range(len(file)):
#     cur = {'name':new_map[i//60]+str(i%60+1)}
#     np_sign_stack=np.array(file[i])
#     sign_tensor = torch.tensor(np_sign_stack)
#     cur['sign'] = sign_tensor
#     cur['signer'] = new_map[i//60]+str(i%60+1)
#     cur['text'] = new_map[i//60]
#     cur['gloss'] = new_map[i//60]

#     data.append(cur)

# print(data)

# X_train, X_test = train_test_split(data, test_size=0.2, random_state=1)

# X_train, X_val = train_test_split(X_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

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
