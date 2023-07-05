
import cv2
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
import logging
from auth_test import VideoAuthApp

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
# from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
#     running_mode=VisionTaskRunningMode.VIDEO,

class MPAuthApp(VideoAuthApp):
   
    def __init__(self, input_video, initial_detect=True, crop_padding=30, target_landmarks=[50, 280, 109, 338], pattern_rel_width=.2, pattern_out_width=20, track_channels = False, colorspace='ycrbr', blur = None, log_level='INFO'):
        VideoAuthApp.__init__(self, input_video, initial_detect, crop_padding, target_landmarks, pattern_rel_width, pattern_out_width, track_channels, colorspace, blur, log_level)

        # mediapipe detector initialization
        logging.info('Setting up MediaPipe FaceMesh')
        base_options = python.BaseOptions(model_asset_path='/home/hadleigh/deepfake_detection/common/weights/face_landmarker_v2_with_blendshapes.task')
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                            num_faces=1, 
                                             min_face_detection_confidence=.25, 
                                            min_face_presence_confidence=.25, 
                                            min_tracking_confidence=.25)
        #output_face_blendshapes=True,
        #output_facial_transformation_matrixes=True,
        self.detector = vision.FaceLandmarker.create_from_options(options)
        logging.info('Done setting up MediaPipe FaceMesh')

    
    def detect(self, frame):
        # frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)   
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)        
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = self.detector.detect(mp_img)  
        face_landmarks_list = detection_result.face_landmarks

        if len(face_landmarks_list) == 0:
            return []
        
        face_landmarks = face_landmarks_list[0] 
        # scale x,y,z by image dimensions
        H, W, c = frame.shape #not the same as self.H, self.W if initial face cropping is being used!
        landmark_coords = [(landmark.x * W, landmark.y * H, landmark.z) for landmark in face_landmarks] #un-normalize
        return landmark_coords
  