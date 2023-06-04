
import cv2
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt

from feature_extractor import FeatureExtractor

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
# from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
#     running_mode=VisionTaskRunningMode.VIDEO,
import mp_alignment

class MPFeatureExtractor(FeatureExtractor):
    def __init__(self, input_video, dist_display_win_size = 60, bbox_norm = True, draw_all_landmarks = False, draw_landmark_nums=False, draw_anchor_target_connector=True, three_d_dist_dist=False, initial_detect=True, initial_bbox_padding = 30, display_dim=800):
        FeatureExtractor.__init__(self, input_video, 'mp_output', 0, [269, 267, 39, 37, 181, 314], dist_display_win_size, bbox_norm, draw_all_landmarks, draw_landmark_nums, draw_anchor_target_connector, three_d_dist_dist, 'mp',  initial_detect, initial_bbox_padding, display_dim)

        # mediapipe extractor initialization
        print('---- Setting up MediaPipe FaceMesh ----')
        base_options = python.BaseOptions(model_asset_path='../common/face_landmarker_v2_with_blendshapes.task')
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                            num_faces=1, 
                                             min_face_detection_confidence=.25, 
                                            min_face_presence_confidence=.25, 
                                            min_tracking_confidence=.25)
        #output_face_blendshapes=True,
        #output_facial_transformation_matrixes=True,
        self.extractor = vision.FaceLandmarker.create_from_options(options)
        print('---- Done setting up MediaPipe FaceMesh ----')

    
    def extract_landmarks(self, frame):
        # frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)   
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)        
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = self.extractor.detect(mp_img)  
        face_landmarks_list = detection_result.face_landmarks

        if len(face_landmarks_list) == 0:
            return []
        
        face_landmarks = face_landmarks_list[0] 
        # scale x,y,z by image dimensions
        H, W, c = frame.shape #not the same as self.input_H, self.input_W if initial face detection (and thus cropping) is being used!
        # MediaPipe by deafult returns facial landmark coordinates normalized to 0-1 based on the input frame dimensions. Here we 
        # un-normalize to get coordinates ranging from 0-W/0_H (i.e., actual pixel coordinates)
        landmark_coords = [(landmark.x * W, landmark.y * H, landmark.z) for landmark in face_landmarks] 
        return landmark_coords
    
    def align_landmarks(self, landmark_list, W, H):
        #again, must pass in W, H, which may not be the same as self.input_W, self.input_H
        landmark_coords_3d_aligned, landmark_coords_2d_aligned  = mp_alignment.align_landmarks(landmark_list, self.input_W, self.input_H, W, H)
        return landmark_coords_2d_aligned, landmark_coords_3d_aligned
    
    def track_landmarks(self, landmark_coords=None, W=None, H=None):
        #again, must pass in W, H, which may not be the same as self.input_W, self.input_H

        #if there were no detections for this frame, add None to lists to maintain alignment of list values with frame numbers
        if landmark_coords == None:
            for i in range(468):
                if i not in self.landmark_tracker:
                    self.landmark_tracker[i] = [np.nan]
                else:
                    self.landmark_tracker[i].append(np.nan)

                if i in self.target_landmarks:
                    if i not in self.dist_tracker:
                        self.dist_tracker[i] = [np.nan]
                    else:
                        self.dist_tracker[i].append(np.nan)
            return
        
        #otherwise, add appropriate coordinates and distances to landmark_tracker and dist_tracker
        anchor_coord = landmark_coords[self.anchor_landmark]
        for i, l in enumerate(landmark_coords):
            if i not in self.landmark_tracker:
                self.landmark_tracker[i] = [l]
            else:
                self.landmark_tracker[i].append(l)

            if i in self.target_landmarks:
                x_diff = (anchor_coord[0] - l[0]) 
                y_diff = (anchor_coord[1] - l[1]) 
                z_diff = (anchor_coord[2] - l[2]) 

                if self.bbox_norm:
                    bbox = self.get_mp_bbox(landmark_coords, W, H)
                    #normalize all differences by face bounding box dimensions
                    bbox_W = bbox[1][0] - bbox[0][0]
                    bbox_H = bbox[1][1] - bbox[0][1]
                    bbox_D = bbox[1][2] - bbox[0][2]
                    x_diff /= bbox_W
                    y_diff /= bbox_H
                    z_diff /= bbox_D

                if self.three_d_dist:
                    dist = np.sqrt(x_diff**2 + y_diff**2 + z_diff**2) 
                else:
                    dist = np.sqrt(x_diff**2 + y_diff**2) 

                if i not in self.dist_tracker:
                    self.dist_tracker[i] = [dist]
                else:
                    self.dist_tracker[i].append(dist)

    def get_mp_bbox(self, landmark_coords, W, H):
        #get face bounding box coordinates based on MediaPipe's extracted landmarks 
        #https://github.com/google/mediapipe/issues/1737
        cx_min=  W
        cy_min = H
        cz_min = W #z scale is roughly same as x scale, according to https://medium.com/@susanne.thierfelder/head-pose-estimation-with-mediapipe-and-opencv-in-javascript-c87980df3acb
        cx_max = cy_max = cz_max = 0
        for id, lm in enumerate(landmark_coords):
            cx, cy, cz = lm
            if cx<cx_min:
                cx_min=cx
            if cy<cy_min:
                cy_min=cy
            if cz<cz_min:
                cz_min=cz
            if cx>cx_max:
                cx_max=cx
            if cy>cy_max:
                cy_max=cy
            if cz>cz_max:
                cz_max=cz  
        bbox = [(cx_min, cy_min, cz_min), (cx_max, cy_max, cz_max)]
        return bbox
