
import cv2
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import logging.config
import os
import sys
sys.path.append('/home/hadleigh/deepfake_detection/common')
from logging_utils import generate_logging_config
from retinaface import RetinaFaceDetector
from mesh_data import MeshData 
from video_generator import VidGenerator
import mp_alignment
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

class MPFeatureExtractor(object):
    def __init__(
        self, 
        input_video, 
        target_landmarks,
        landmark_pairs=[], 
        norm_approach="first_upper_lower_bbox", 
        analysis_type="landmark_to_anchor",
        output_directory="mp_output_videos", 
        anchor_landmark=4, 
        initial_bbox_padding=30, 
        display_dim=800, 
        log_level='INFO',
        dist_display_win_size=60, 
        draw_landmark_nums=False,
        draw_all_landmarks=True, 
        generate_video=False, 
        three_d_dist=True, 
        initial_detect=True, 
        frames_to_include = ["annotated_vid", "annotated_blank", "data_plot"]):

        logging.config.dictConfig(generate_logging_config('feature extractor', log_level))

        # initialize attributes with values passed into constructor
        self.initial_bbox_padding = initial_bbox_padding
        self.anchor_landmark = anchor_landmark
        self.target_landmarks = target_landmarks
        self.landmark_pairs = landmark_pairs
        self.three_d_dist = three_d_dist
        self.draw_all_landmarks = draw_all_landmarks
        self.draw_landmark_nums = draw_landmark_nums
        self.dist_display_win_size = dist_display_win_size
        self.generate_video = generate_video
        self.norm_approach = norm_approach
        self.analysis_type = analysis_type

        # initalize trackers
        self.landmark_coord_tracker = {}
        self.landmark_data_tracker = {}
        self.landmark_group_tracker = {}

        #initialize speed benchmarking attributes
        self.curr_landmark_ext_fps = 0
        self.tot_det_time = 0
        self.tot_overall_time = 0
        self.curr_overall_fps = 0
        self.tot_frames = 0

        self.mesh_data = MeshData()
        self.bbox = {
            "upper":[(0,0,0), (0,0,0)], 
            "lower": [(0,0,0), (0,0,0)], 
            "face": [(0,0,0), (0,0,0)]
        }
        self.landmarks = {
            "upper": self.mesh_data.upper_landmarks, 
            "lower": self.mesh_data.lower_landmarks, 
            "face": self.mesh_data.all
        }
        
        logging.info('Initializing RetinaFace Detector')
        self.face_detector = RetinaFaceDetector("resnet50")
        logging.info('Done initializing RetinaFace Detector')

        self.initialize_video_capture(input_video)


        if self.generate_video: 
            input_vid_name = input_video.split('/')[-1][:-4]
            input_cap_fps = int(self.input_capture.get(cv2.CAP_PROP_FPS))

            self.vidgen = VidGenerator(
                output_directory, 
                input_vid_name, 
                display_dim, 
                input_cap_fps, 
                frames_to_include, 
                initial_bbox_padding, 
                draw_landmark_nums, 
                draw_all_landmarks, 
                self.input_H, 
                self.input_W)

        self.frame_num = 0 
        self.init_frame = None
        self.curr_face_bbox = [(0,0,0), (0,0,0)]
        
        logging.info("Setting up MediaPipe FaceMesh")
        self.init_mediapipe()
        logging.info('Done setting up MediaPipe FaceMesh')
        


    def init_mediapipe(self):
        # mediapipe extractor initialization
        base_options = python.BaseOptions(model_asset_path='../common/weights/face_landmarker_v2_with_blendshapes.task')
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                            num_faces=1, 
                                             min_face_detection_confidence=.25, 
                                            min_face_presence_confidence=.25, 
                                            min_tracking_confidence=.25)
        #output_face_blendshapes=True,
        #output_facial_transformation_matrixes=True,
        self.extractor = vision.FaceLandmarker.create_from_options(options)


    
    def initialize_video_capture(self, input_video):

        try:
            self.input_capture = cv2.VideoCapture(input_video)
        except:
            raise ValueError('Input video path %s not valid' % input_video)

        if self.input_capture:    
            self.input_W , self.input_H = int(self.input_capture.get(3)), int(self.input_capture.get(4)) #input video dimensions
            self.tot_input_vid_frames = int(self.input_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            raise ValueError("Invalid input video")
    
    def extract_landmarks(self, frame):
        """
        Extracts facial landmarks, returning a list of 3D coordinates - one for each landmark. 

        Parameters
        ----------
        frame : np array/cv2 frame
            Frame to run face landmark exctraction on

        Returns
        -------
        landmark_coords : List of 3D tuples
            3D coordinate of each face landmark
        """
        # frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)   
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)        
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = self.extractor.detect(mp_img)  
        face_landmarks_list = detection_result.face_landmarks
        if len(face_landmarks_list) == 0:
            return []
        face_landmarks = face_landmarks_list[0] 

        H, W, c = frame.shape #not the same as self.input_H, self.input_W if initial face detection (and thus cropping) is being used!
        # MediaPipe by deafult returns facial landmark coordinates normalized to 0-1 based on the input frame dimensions. Here we 
        # un-normalize to get coordinates ranging from 0-W/0_H (i.e., actual pixel coordinates)
        landmark_coords = [(landmark.x * W, landmark.y * H, landmark.z) for landmark in face_landmarks] 
        return landmark_coords

    def align_landmarks(self, landmarks, W, H):
        landmark_coords_3d_aligned, landmark_coords_2d_aligned  = mp_alignment.align_landmarks(landmarks, self.input_W, self.input_H, W, H)
        return landmark_coords_2d_aligned, landmark_coords_3d_aligned

    def get_diff(self, bbox, xdiff, ydiff, zdiff):
        bbox_W = bbox[1][0] - bbox[0][0]
        bbox_H = bbox[1][1] - bbox[0][1]
        bbox_D = bbox[1][2] - bbox[0][2]
        xdiff = xdiff / bbox_W
        ydiff = ydiff / bbox_H
        zdiff = zdiff / bbox_D

        return xdiff, ydiff, zdiff
    
    def get_first_bbox(self, region, W, H):
        bbox = self.bbox[region]

        if bbox[1][0] == bbox [0][0] == bbox[1][1] == bbox[0][1] == bbox[1][2] == bbox[0][2] == 0:
            bbox = self.get_curr_bbox(self.landmarks[region], W, H)
            self.bbox[region] = bbox 
        
        return bbox
    
    def get_curr_bbox(self, landmarks, W, H):
        cx_min=  W
        cy_min = H
        cz_min = W #z scale is roughly same as x scale, according to https://medium.com/@susanne.thierfelder/head-pose-estimation-with-mediapipe-and-opencv-in-javascript-c87980df3acb
        cx_max = cy_max = cz_max = 0
        for id, l in enumerate(landmarks):
            lm = self.curr_frame_landmarks_3d[l]

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


    def normalize(self, xdiff, ydiff, zdiff, W, H, i):
        if self.norm_approach == "face_bbox": 
            bbox = self.get_curr_bbox(self.landmarks["face"], W, H)
            #normalize all differences by face bounding box dimensions
           
        elif self.norm_approach == "first_upper_lower_bbox":
            if i in self.landmarks["upper"]:
                bbox = self.get_first_bbox("upper", W, H)
            else:
                bbox = self.get_first_bbox("lower", W, H)
        elif self.norm_approach == "first_bbox":
            bbox = self.get_first_bbox("face", W, H)
            
        xdiff, ydiff, zdiff = self.get_diff(bbox, xdiff, ydiff, zdiff)

        
        return xdiff, ydiff, zdiff

    def set_landmark_dist(self, anchor_coord, l, W, H, i):
        x_diff = (anchor_coord[0] - l[0]) 
        y_diff = (anchor_coord[1] - l[1]) 
        z_diff = (anchor_coord[2] - l[2]) 


        x_diff, y_diff, z_diff = self.normalize(x_diff, y_diff, z_diff, W, H, i)
            
        if self.three_d_dist:
            dist = np.sqrt(x_diff**2 + y_diff**2 + z_diff**2) 
        else:
            dist = np.sqrt(x_diff**2 + y_diff**2) 
        if i not in self.landmark_data_tracker:
            self.landmark_data_tracker[i] = [dist]
        else:
            self.landmark_data_tracker[i].append(dist)  
        
    
    def set_landmarks_none(self):
    #if there were no landmarks extracted for this frame, add None to lists to maintain alignment of list values with frame numbers

        for i in range(468):
            if i not in self.landmark_coord_tracker:
                self.landmark_coord_tracker[i] = [np.nan]
            else:
                self.landmark_coord_tracker[i].append(np.nan)

            if i in self.target_landmarks:
                if i not in self.landmark_data_tracker:
                    self.landmark_data_tracker[i] = [np.nan]
                else:
                    self.landmark_data_tracker[i].append(np.nan)
    
    def track_landmarks_to_anchor(self, W, H):
        #if landmarks were extracted for this fame, add appropriate coordinates and distances to landmark_tracker and dist_tracker
        anchor_coord = self.curr_frame_landmarks_3d[self.anchor_landmark]
        for i, l in enumerate(self.curr_frame_landmarks_3d):
            if i not in self.landmark_coord_tracker:
                self.landmark_coord_tracker[i] = [l]
            else:
                self.landmark_coord_tracker[i].append(l)
            if i in self.target_landmarks:
                self.set_landmark_dist(anchor_coord, l, W, H, i)

    def track_landmark_pairs(self, landmark_coords, W, H):
        for pair in self.landmark_pairs:
            pass

    
    def track_landmarks(self, W=None, H=None):
        """
        Update landmark_tracker and dist_tracker with new frame's data

        Parameters
        ----------
        landmark_coords : List of 3D tuples, optional
            3D coordinate of each face landmark, as outputted by extract_landmarks, to keep track of 
        W, H : int, optional 
            Dimensions, in pixels, of frame that facial landmark extraction was run on.
            This is not same as self.input_W, self.input_H f if initial face detection (and thus cropping) is being used!
        
        If landmark_coords, W, and H = None, no landmarks were detected in this frame. We still must appropriately update
        the trackers

        Returns
        ----------
        None
        """
        if self.curr_frame_landmarks_3d == None:
            self.set_landmarks_none()
        elif self.analysis_type == "landmark_to_anchor":
            self.track_landmarks_to_anchor(W, H)
        elif self.analysis_type == "landmark_pairs": 
            self.set_landmarks_none()

    def crop_frame(self, frame):
        if self.curr_face_bbox == None:
            return None
        else:
            # get crop of frame to pass to facial landmark extraction
            bottom = max(self.curr_face_bbox[1] - self.initial_bbox_padding, 0)
            top = min(self.curr_face_bbox[3]+1+ self.initial_bbox_padding, self.input_H)
            left = max( self.curr_face_bbox[0]-self.initial_bbox_padding, 0)
            right = min(self.curr_face_bbox[2]+1+self.initial_bbox_padding, self.input_W)
            return frame[bottom:top,left:right]
    
    def empty_frame(self):
        self.track_landmarks()
        if self.generate_video:
            self.vidgen.set_plot_frame(self.frame_num, self.target_landmarks, self.landmark_data_tracker)
            self.vidgen.set_annotated_blank()
            self.vidgen.set_annotated_frame(self.init_frame)


    def detect_landmarks(self, frame):
        start = time.time()
        landmark_list = self.extract_landmarks(frame)
        end = time.time()

        self.tot_det_time += (end - start)
        self.curr_landmark_ext_fps = self.frame_num / self.tot_det_time

        return landmark_list
    
    def plot_and_track_landmarks(self, landmarks, frame):
        if len(landmarks) == 0:
            self.empty_frame() 
        else: 
            H, W, c = frame.shape #use h, w defined here instead of self.input_W, self.input_H because they are not the same if initial face deteciton is being used
            aligned_landmark_list_2d, aligned_landmark_list_3d = self.align_landmarks(landmarks, W, H)
            self.curr_frame_landmarks_3d = aligned_landmark_list_3d
            
            self.track_landmarks(W, H)
            if self.generate_video: 
                self.vidgen.set_annotated_frame(self.init_frame, landmarks, self.curr_face_bbox, self.anchor_landmark, self.target_landmarks, self.landmark_data_tracker)
                self.vidgen.set_annotated_blank(aligned_landmark_list_2d, self.target_landmarks, self.anchor_landmark, self.landmark_data_tracker)
                self.vidgen.set_plot_frame(self.frame_num, self.target_landmarks, self.landmark_data_tracker)

    def update_fps(self, overall_start):
        overall_end = time.time()
        self.tot_overall_time += (overall_end - overall_start)
        self.curr_overall_fps = self.frame_num / self.tot_overall_time

    def analyze_single_frame(self):
        ret, frame = self.input_capture.read()
        if ret: 
            overall_start = time.time()
            self.frame_num += 1
            self.init_frame = frame.copy()

            self.curr_face_bbox = self.face_detector.detect(frame)
            frame = self.crop_frame(frame)

            if frame is None: 
                #no face was detected in frame 
                self.empty_frame()
            else: 
                landmark_list = self.detect_landmarks(frame)
                self.plot_and_track_landmarks(landmark_list, frame)

            if self.generate_video:
                self.vidgen.write_combined()

            self.update_fps(overall_start)
            return True 
       
        return False  

    
    def run_extraction(self):
        with tqdm(total=self.tot_input_vid_frames) as pbar: 
            pbar.set_description("Performing feature extraction ")
            while self.input_capture.isOpened():
                if self.analyze_single_frame(): 

                    pbar.update(1)
                else:
                    break
            
        print('Average extraction FPS: ', self.curr_landmark_ext_fps)
        print('Average overall FPS: ', self.curr_overall_fps)
        
        if self.generate_video:
            self.input_capture.release()
            self.vidgen.release_vid()

        return self.landmark_coord_tracker, self.landmark_data_tracker, self.landmark_group_tracker


    


        



        




