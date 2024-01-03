import json
from PIL import Image
import logging

import tensorflow as tf
import numpy as np
import os
import cv2

import mediapipe as mp
 
from azureml.contrib.services.aml_request import rawhttp
from azureml.contrib.services.aml_response import AMLResponse

class InferenceModel():
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

 
    def predict(self, inputs):
        logging.info(f"Model inputs dim: {inputs.shape}")
        prediction = self.model.predict(inputs)[0]
        return prediction

def init():
    global model
    global actions
    actions = ['can','you','help','me','what','name','hamburger','french fries','thanks','bye','hello','excuse me','sorry']
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "noface_seq60_16actions.h5")
    model = InferenceModel(model_path)

    global mp_hands 
    global mp_pose
    global mp_drawing 
    global mp_drawing_styles 
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
 
@rawhttp
def run(request):
    if request.method != 'POST':
        return AMLResponse(f"Unsupported verb: {request.method}", 400)

    # Method 1:
    image_list = []
    image_list.append(request.files['image0'])
    image_list.append(request.files['image1'])
    image_list.append(request.files['image2'])
    image_list.append(request.files['image3'])
    image_list.append(request.files['image4'])
    image_list.append(request.files['image5'])
    image_list.append(request.files['image6'])
    image_list.append(request.files['image7'])
    image_list.append(request.files['image8'])
    image_list.append(request.files['image9'])
    image_list.append(request.files['image10'])
    image_list.append(request.files['image11'])
    image_list.append(request.files['image12'])
    image_list.append(request.files['image13'])
    image_list.append(request.files['image14'])
    image_list.append(request.files['image15'])
    image_list.append(request.files['image16'])
    image_list.append(request.files['image17'])
    image_list.append(request.files['image18'])
    image_list.append(request.files['image19'])
    image_list.append(request.files['image20'])
    image_list.append(request.files['image21'])
    image_list.append(request.files['image22'])
    image_list.append(request.files['image23'])
    image_list.append(request.files['image24'])
    image_list.append(request.files['image25'])
    image_list.append(request.files['image26'])
    image_list.append(request.files['image27'])
    image_list.append(request.files['image28'])
    image_list.append(request.files['image29'])
    image_list.append(request.files['image30'])
    image_list.append(request.files['image31'])
    image_list.append(request.files['image32'])
    image_list.append(request.files['image33'])
    image_list.append(request.files['image34'])
    image_list.append(request.files['image35'])
    image_list.append(request.files['image36'])
    image_list.append(request.files['image37'])
    image_list.append(request.files['image38'])
    image_list.append(request.files['image39'])
    image_list.append(request.files['image40'])
    image_list.append(request.files['image41'])
    image_list.append(request.files['image42'])
    image_list.append(request.files['image43'])
    image_list.append(request.files['image44'])
    image_list.append(request.files['image45'])
    image_list.append(request.files['image46'])
    image_list.append(request.files['image47'])
    image_list.append(request.files['image48'])
    image_list.append(request.files['image49'])
    image_list.append(request.files['image50'])
    image_list.append(request.files['image51'])
    image_list.append(request.files['image52'])
    image_list.append(request.files['image53'])
    image_list.append(request.files['image54'])
    image_list.append(request.files['image55'])
    image_list.append(request.files['image56'])
    image_list.append(request.files['image57'])
    image_list.append(request.files['image58'])
    image_list.append(request.files['image59'])

    # Method 2: (fail)
    # try:
    #     for (name, (filename, image_obj, type)) in request.files:
    #         image_list.append(image_obj)
    # except:
    #     return AMLResponse(f"Reading files error: {request.files}", 400)

    # Method 3: (fail?)
    # image_list = request.files['files']


    sequence = []
    # Set mediapipe model 
    with mp_hands.Hands(min_detection_confidence=0.5,min_tracking_confidence=0.5) as hands:
        with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
            for image_bytes in image_list:
                image = Image.open(image_bytes)
                image = np.array(image) 
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)
                results_pose = pose.process(image)

                # Draw the hand annotations on the image.
                # image.flags.writeable = True
                # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                hand_np = []
                if results.multi_hand_landmarks:
                    for i in range(len(results.multi_hand_landmarks)):
                        # mp_drawing.draw_landmarks(image,results.multi_hand_landmarks[i],mp_hands.HAND_CONNECTIONS,mp_drawing_styles.get_default_hand_landmarks_style(),mp_drawing_styles.get_default_hand_connections_style())
                        hand_np.append(np.array([[res.x, res.y, res.z] for res in results.multi_hand_landmarks[i].landmark]).flatten() if results.multi_hand_landmarks[i] else np.zeros(21*3))

                if len(hand_np) == 0:
                    lh = np.zeros(21*3)
                    rh = np.zeros(21*3)
                elif len(hand_np) == 1:
                    lh = hand_np[0]
                    rh = np.zeros(21*3)
                else:
                    lh = hand_np[0]
                    rh = hand_np[1]        

                # Draw pose annotations        
                # mp_drawing.draw_landmarks(image,results_pose.pose_landmarks,mp_pose.POSE_CONNECTIONS,landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                ps = np.array([[res.x, res.y, res.z, res.visibility] for res in results_pose.pose_landmarks.landmark]).flatten() if results_pose.pose_landmarks else np.zeros(33*4)

                keypoints = np.concatenate([ps, lh, rh])
                sequence.append(keypoints)

    preds = model.predict(np.expand_dims(sequence, axis=0))
    sign = actions[np.argmax(preds)]
    # return AMLResponse(json.dumps({"preds": preds.tolist()}), 200)
    return AMLResponse(json.dumps({"sign": sign}), 200)
