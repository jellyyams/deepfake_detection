
import cv2
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from scipy import signal
import sys
sys.path.append('/home/hadleigh/df_pipeline/common')
from retinaface import RetinaFaceDetector
import logging
import logging.config
from tqdm import tqdm

class VideoPatternApp(object):
   
    def __init__(self, input_video, initial_detect, crop_padding, alpha, frequency, target_landmarks, pattern_rel_width, pattern_color, log_level):
        LOGGING_CONFIG = { 
            'version':1,
            'disable_existing_loggers': True,
            'formatters': { 
                'standard': { 
                    'format': '%(levelname)s in add_patterns: %(message)s'
                },
            },
            'handlers': {
                'default': {
                    'level': 'INFO',
                    'formatter': 'standard',
                    'class': 'logging.StreamHandler',
                }
            },
            'loggers': {
                '': {
                    'handlers': ['default'],
                    'level': 'INFO',
                    'propagate': True
                },
            },
            'root': {
                'handlers': ['default'],
                'level': log_level
            }
        }
        logging.config.dictConfig(LOGGING_CONFIG)

        
        self.initial_detect = initial_detect
        self.crop_padding = crop_padding
        self.alpha = alpha
        self.pattern_rel_width = pattern_rel_width 
        self.pattern_color = pattern_color
        self.target_landmarks = target_landmarks
     
        try:
            self.input_capture = cv2.VideoCapture(input_video)
        except:
            logging.error('Input video path %s not valid' % input_video)
            raise ValueError('Input video path %s not valid' % input_video)

        if self.input_capture is not None:    
            self.W , self.H = int(self.input_capture.get(3)), int(self.input_capture.get(4)) #input video dimensions
        else:
            raise ValueError("Invalid input video")
        
        self.tot_input_vid_frames = int(self.input_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        input_cap_fps = int(self.input_capture.get(cv2.CAP_PROP_FPS))
        self.frame_period = int(input_cap_fps / frequency) #add pattern every {frame_period} frames

        logging.info('Setting up output videos')
        self.output_vid_path = 'pattern_output/{}_pattern_b{}_g{}_r{}_{}Hz_alpha{}_sz{}.mp4'.format(input_video.split('/')[-1][:-4], self.pattern_color[0], self.pattern_color[1], self.pattern_color[2], frequency, self.alpha, self.pattern_rel_width)
        self.out_vid = cv2.VideoWriter(self.output_vid_path, cv2.VideoWriter_fourcc(*'MP4V'), input_cap_fps, (self.W, self.H))
        ref_output_vid_path = 'pattern_output/{}_pattern_b{}_g{}_r{}_{}Hz_ref_sz{}.mp4'.format(input_video.split('/')[-1][:-4], self.pattern_color[0], self.pattern_color[1], self.pattern_color[2], frequency, self.pattern_rel_width)
        self.ref_out_vid = cv2.VideoWriter(ref_output_vid_path, cv2.VideoWriter_fourcc(*'MP4V'), input_cap_fps, (self.W, self.H))
        logging.info('Done setting up output videos')
        

        #set up initial face detector, if using
        if self.initial_detect == True:
            logging.info('Initializing RetinaFace Detector')
            self.initial_detector = RetinaFaceDetector("resnet50")
            logging.info('Done initializing RetinaFace Detector')

    def get_output_vid_path(self):
        return self.output_vid_path
        
    def run(self):
        """
        process the input video!
        """
        i = 0
        with tqdm(total=self.tot_input_vid_frames) as pbar:
            pbar.set_description('Running pattern addition')
            while self.input_capture.isOpened():
                ret, frame = self.input_capture.read()
                if ret:
                    i += 1
                    self.process_frame(frame, i)  
                else:
                    break
                pbar.update(1)
        
        self.input_capture.release()
        self.out_vid.release()
        self.ref_out_vid.release()
        
    def detect(self, frame):
        raise NotImplementedError("Method detect() must be implemented in child class.")

    def process_frame(self, frame, frame_num):
        if frame_num % self.frame_period == 0:
            init_frame = frame.copy()
            if self.initial_detect:
                #run initial face detection
                initial_face_bbox = self.initial_detector.detect(frame)
                if initial_face_bbox == None:
                    frame = None
                else:
                    bottom = max(initial_face_bbox[1] - self.crop_padding, 0)
                    top = min(initial_face_bbox[3]+1+ self.crop_padding, self.H)
                    left = max( initial_face_bbox[0]-self.crop_padding, 0)
                    right = min(initial_face_bbox[2]+1+self.crop_padding, self.W)
                    frame = frame[bottom:top,left:right]
            else:
                initial_face_bbox = None
            
            landmark_list = self.detect(frame)  
            annotated_frame, annotated_frame_ref = self.annotate(init_frame, landmark_list, initial_face_bbox)
            self.out_vid.write(annotated_frame)
            self.ref_out_vid.write(annotated_frame_ref)
        else:
            self.out_vid.write(frame)
            self.ref_out_vid.write(frame)


    def annotate(self, frame, landmark_list, initial_face_bbox=None):
        if initial_face_bbox is not None:       
            #incremement all landmarks according to initial_face_bbox and crop_padding values to translate to init_frame coordinate system
            init_landmark_list = []
            for coord in landmark_list:
                init_landmark_list.append([coord[0] + initial_face_bbox[0] - self.crop_padding, coord[1] + initial_face_bbox[1] - self.crop_padding])
        else:
            init_landmark_list = landmark_list
    
        shapes = np.zeros_like(frame, np.uint8)
    
        out = frame.copy()
        out_ref = frame.copy()
        for i in self.target_landmarks:
            x, y = init_landmark_list[i]   
            #normalize by initial_face_bbox size
            # initial_face_bbox_dims = (initial_face_bbox[2] - initial_face_bbox[0], initial_face_bbox[3] - initial_face_bbox[1])
            # norm_pattern_width = (self.pattern_rel_width * max(initial_face_bbox_dims))
            box_start = (int(max(0, x - (self.pattern_rel_width/2))), int(max(0, y - (self.pattern_rel_width/2))))
            box_end = (int(min(self.W, x + (self.pattern_rel_width/2))), int(min(self.H, y + (self.pattern_rel_width/2))))
            cv2.rectangle(shapes, box_start, box_end, self.pattern_color, -1) #add transparent box around landmark in shapes overlay
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, self.alpha, shapes, 1 - self.alpha, 0)[mask]
        out_ref[mask] = cv2.addWeighted(frame, 0, shapes, 1, 0)[mask]
   
        return out, out_ref

            
