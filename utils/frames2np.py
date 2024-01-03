import cv2
import os
from PIL import Image
import numpy as np
import mediapipe as mp
root = 'C:\\Users\\ryanr\\ASLtoEng\\'
MP_DATA_PATH = os.path.join(root,'MP_data')

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

mp_holistic = mp.solutions.holistic # Holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for label in os.listdir(MP_DATA_PATH):
        LABEL_PATH = os.path.join(MP_DATA_PATH,label)
        for folder in os.listdir(LABEL_PATH):
            FOLDER_PATH = os.path.join(LABEL_PATH, folder)
            if os.path.isdir(FOLDER_PATH):
    
                for filename in os.listdir(FOLDER_PATH):
                    # print(LABEL_PATH)
                    if filename.endswith('.jpg'):
                        print(f'Converting {filename} from frames to numpy...')

                        img = Image.open(filename)
            
                        # Make detections
                        image, results = mediapipe_detection(frame, holistic)
    
                        os.chdir(LABEL_PATH)
                        video_name = filename.split('.')[0]
    
                        # Export keypoints
                        keypoints = extract_keypoints(results)
                        npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                        np.save(npy_path, keypoints)
             
                        vidcap = cv2.VideoCapture(filename)
                        success, image = vidcap.read()
                        count = 0
            
                        os.chdir(os.path.join(LABEL_PATH,video_name))
            
                        while success:
                            cv2.imwrite("%d.jpg" % count, image)     # save frame as JPEG file      
                            success,image = vidcap.read()
                            # print('Read a new frame: ', success)
                            count += 1
            
                        print('Converted.')
                        
                        # close the video
                        vidcap.release()
            
                        # delete the video after converting
                        os.chdir(LABEL_PATH)
                        os.remove(filename)

