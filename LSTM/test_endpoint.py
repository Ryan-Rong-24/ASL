import requests
# import json
import os
# import ssl
# from PIL import Image
# import mediapipe as mp
# import cv2
# import numpy as np

# import tensorflow as tf


url = 'https://asl-demo.eastus2.inference.ml.azure.com/score'

# print(open(os.path.join('test_endpoint_data','1.jpg'),'rb'))

# actions = ['can','you','help','me','what','name','hamburger','french fries','thanks','bye','hello','excuse me','sorry']
# model_path = os.path.join("transformer_model","transformer_model")
# print(model_path)
# model = load_model(model_path) # arg: load path

# mp_hands = mp.solutions.hands
# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles


files=[]
for i in range(0,60):
    files.append((f'image{i}',(f"{i}.jpg",open(os.path.join('test_endpoint_data',f'{i}.jpg'),'rb'),'application/octet-stream')))

# files=[
#   ('image',('file',open(os.path.join('test_endpoint_data',"32.jpg"),'rb'),'application/octet-stream'))
# ]

# files = {"image":open(os.path.join('test_endpoint_data',"32.jpg"),'rb').read()}

headers = {
  'Authorization': ('Bearer '+'H0xsFsBzpEUDBzlaRZpI2of0btms1EAI'),
}
response = requests.request("POST", url, headers=headers, files=files)
# request = requests.Request("POST", url, headers=headers, files=files)

# image_list = []
# for (name, (filename, image_obj, type)) in request.files:
#     image_list.append(image_obj)

# sequence = []
# # Set mediapipe model 
# with mp_hands.Hands(min_detection_confidence=0.5,min_tracking_confidence=0.5) as hands:
#     with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
#         for image_bytes in image_list:
#             image = Image.open(image_bytes)
#             image = np.array(image) 
#             # To improve performance, optionally mark the image as not writeable to
#             # pass by reference.
#             image.flags.writeable = False
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             results = hands.process(image)
#             results_pose = pose.process(image)

#             # Draw the hand annotations on the image.
#             # image.flags.writeable = True
#             # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#             hand_np = []
#             if results.multi_hand_landmarks:
#                 for i in range(len(results.multi_hand_landmarks)):
#                     # mp_drawing.draw_landmarks(image,results.multi_hand_landmarks[i],mp_hands.HAND_CONNECTIONS,mp_drawing_styles.get_default_hand_landmarks_style(),mp_drawing_styles.get_default_hand_connections_style())
#                     hand_np.append(np.array([[res.x, res.y, res.z] for res in results.multi_hand_landmarks[i].landmark]).flatten() if results.multi_hand_landmarks[i] else np.zeros(21*3))

#             if len(hand_np) == 0:
#                 lh = np.zeros(21*3)
#                 rh = np.zeros(21*3)
#             elif len(hand_np) == 1:
#                 lh = hand_np[0]
#                 rh = np.zeros(21*3)
#             else:
#                 lh = hand_np[0]
#                 rh = hand_np[1]        

#             # Draw pose annotations        
#             # mp_drawing.draw_landmarks(image,results_pose.pose_landmarks,mp_pose.POSE_CONNECTIONS,landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

#             ps = np.array([[res.x, res.y, res.z, res.visibility] for res in results_pose.pose_landmarks.landmark]).flatten() if results_pose.pose_landmarks else np.zeros(33*4)

#             keypoints = np.concatenate([ps, lh, rh])
#             sequence.append(keypoints)

# print(np.expand_dims(sequence, axis=0).shape)
# preds = model.predict(np.expand_dims(sequence, axis=0))[0]
# print(preds)
# print(response.request)
print(response.text)