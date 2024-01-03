import requests
# import json
import os
import time
# import ssl
# from PIL import Image
# import mediapipe as mp
# import cv2
# import numpy as np

# from tensorflow.keras import layers
# from tensorflow import keras
# import tensorflow as tf


url = 'https://asl-demo.eastus2.inference.ml.azure.com/score'

# print(open(os.path.join('test_endpoint_data','1.jpg'),'rb'))

# class PositionalEmbedding(layers.Layer):
#     def __init__(self, sequence_length, output_dim, **kwargs):
#         super().__init__(**kwargs)
#         self.position_embeddings = layers.Embedding(
#             input_dim=sequence_length, output_dim=output_dim
#         )
#         self.sequence_length = sequence_length
#         self.output_dim = output_dim

#     def call(self, inputs):
#         # The inputs are of shape: `(batch_size, frames, num_features)`
#         length = tf.shape(inputs)[1]
#         positions = tf.range(start=0, limit=length, delta=1)
#         embedded_positions = self.position_embeddings(positions)
#         print("SHAPE:")
#         print(inputs.shape)
#         return inputs + embedded_positions

#     def compute_mask(self, inputs, mask=None):
#         mask = tf.reduce_any(tf.cast(inputs, "bool"), axis=-1)
#         return mask
#     def get_config(self):
#         config = super().get_config()
#         config.update({
# #             "position_embeddings": self.position_embeddings,
#             "sequence_length": self.sequence_length,
#             "output_dim": self.output_dim,
#         })
#         return config

# class TransformerEncoder(layers.Layer):
#     def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
#         super().__init__(**kwargs)
#         self.embed_dim = embed_dim
#         self.dense_dim = dense_dim
#         self.num_heads = num_heads
#         self.attention = layers.MultiHeadAttention(
#             num_heads=num_heads, key_dim=embed_dim, dropout=0.3
#         )
#         self.dense_proj = keras.Sequential(
#             [layers.Dense(dense_dim, activation=tf.nn.gelu), layers.Dense(embed_dim),]
#         )
#         self.layernorm_1 = layers.LayerNormalization()
#         self.layernorm_2 = layers.LayerNormalization()

#     def call(self, inputs, mask=None):
#         if mask is not None:
#             mask = mask[:, tf.newaxis, :]

#         attention_output = self.attention(inputs, inputs, attention_mask=mask)
#         proj_input = self.layernorm_1(inputs + attention_output)
#         proj_output = self.dense_proj(proj_input)
#         return self.layernorm_2(proj_input + proj_output)
#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "embed_dim": self.embed_dim,
#             "dense_dim": self.dense_dim,
#             "num_heads": self.num_heads,
# #             "attention": self.attention,
# #             "dense_proj": self.dense_proj,
# #             "layernorm_1": self.layernorm_1,
# #             "layernorm_2": self.layernorm_2,
#         })
#         return config

# def get_compiled_model():
#     sequence_length = MAX_SEQ_LENGTH
#     embed_dim = NUM_FEATURES
#     dense_dim = 8
#     num_heads = 6
#     classes = len(actions)

#     inputs = keras.Input(shape=(None, None))
#     x = PositionalEmbedding(
#         sequence_length, embed_dim, name="frame_position_embedding"
#     )(inputs)
#     x = TransformerEncoder(embed_dim, dense_dim, num_heads, name="transformer_layer")(x)
#     x = layers.GlobalMaxPooling1D()(x)
#     x = layers.Dropout(0.5)(x)
#     outputs = layers.Dense(classes, activation="softmax")(x)
#     model = keras.Model(inputs, outputs)

#     model.compile(
#         optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
#     )
#     return model

# def load_model(filepath):
#     model = get_compiled_model()
#     model.load_weights(filepath)
#     # _, accuracy = model.evaluate(test_data, test_labels)
#     # print(f"Test accuracy: {round(accuracy * 100, 2)}%")

#     return model

# MAX_SEQ_LENGTH = 30
# NUM_FEATURES = 258
# sequence_length = 30
# actions = ['can','you','help','me','what','name','hamburger','french fries','thanks','bye','hello','excuse me','sorry']
# model_path = os.path.join("transformer_model","transformer_model")
# print(model_path)
# model = load_model(model_path) # arg: load path

# mp_hands = mp.solutions.hands
# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

tic = time.perf_counter()


files=[]
for i in range(0,60,2):
    files.append((f'image{i}',(f"{i}.jpg",open(os.path.join(f'test_endpoint_data',f'{i}.jpg'),'rb'),'application/octet-stream')))

# print(len(files))
headers = {
  'Authorization': ('Bearer '+'H0xsFsBzpEUDBzlaRZpI2of0btms1EAI'),
}
response = requests.request("POST", url, headers=headers, files=files)
# request = requests.Request("POST", url, headers=headers, files=files)
# print(request.method)

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
# sign = actions[np.argmax(preds)]
# print(sign)
print(response.request)
print(response.text)

toc = time.perf_counter()
print(f"Inference took {toc - tic:0.4f} seconds")