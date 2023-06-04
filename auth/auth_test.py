
import cv2
import numpy as np
import pickle
import time
import json
import os 
import matplotlib.pyplot as plt
from scipy import signal
from datetime import datetime
import sys 
import logging
import logging.config
sys.path.append('/home/hadleigh/df_pipeline/common')
from retinaface import RetinaFaceDetector
from tqdm import tqdm
from vis_channels_realtime import channel_visualizer



class VideoAuthApp(object):
   
    def __init__(self, input_video, initial_detect, crop_padding, target_landmarks, pattern_rel_width, pattern_out_width, track_channels, log_level):
        LOGGING_CONFIG = { 
            'version':1,
            'disable_existing_loggers': True,
            'formatters': { 
                'standard': { 
                    'format': '%(levelname)s in auth_test: %(message)s'
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

        self.track_channels = track_channels
        self.initial_detect = initial_detect
        self.crop_padding = crop_padding
        self.pattern_rel_width = pattern_rel_width
        self.pattern_out_width = pattern_out_width
        self.target_landmarks = target_landmarks
  
        try:
            self.input_capture = cv2.VideoCapture(input_video)
        except:
            raise ValueError('Input video path %s not valid' % input_video)
        
        if self.input_capture is not None:    
            self.W , self.H = int(self.input_capture.get(3)), int(self.input_capture.get(4)) #input video dimensions
        else:
            raise ValueError("Invalid input video")
        
      
        self.output_dir = 'auth_test_output/{}'.format(input_video.split('/')[-1][:-4])
        try:
            os.makedirs(self.output_dir, exist_ok = True)
            logging.info('Directory {} created successfully'.format(self.output_dir))
        except OSError as error:
            logging.error('Directory {} already exists.'.format(self.output_dir))

        self.tot_input_vid_frames = int(self.input_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        input_cap_fps = int(self.input_capture.get(cv2.CAP_PROP_FPS))
        logging.info('Setting up output videos')
        output_vid_name = '{}/main_output_video.mp4'.format(self.output_dir)
        self.out_vid = cv2.VideoWriter(output_vid_name, cv2.VideoWriter_fourcc(*'MP4V'), input_cap_fps, (self.W, self.H))
        
        #create output video for each of target regions
        self.target_out_vids = []
        target_out_vid_names = []
        for l_num in self.target_landmarks:
            target_out_vid_name = '{}/target_region{}.mp4'.format(self.output_dir, l_num)
            target_out_vid_names.append(target_out_vid_name)
            target_out_vid = cv2.VideoWriter(target_out_vid_name, cv2.VideoWriter_fourcc(*'hvc1'), input_cap_fps, (self.pattern_out_width, self.pattern_out_width))
            self.target_out_vids.append(target_out_vid)
       
        logging.info('Done setting up output videos')
        
        #set up initial face detector, if using
        if self.initial_detect == True:
            logging.info('Initializing RetinaFace Detector')
            self.initial_detector = RetinaFaceDetector("resnet50")
            logging.info('Done initializing RetinaFace Detector')
        
        if self.track_channels:
            # visualize rgb channels over time
            self.target_channel_apps = []
            for target_lm in target_landmarks:
                target_chanel_out_vid_name = '{}/target_region{}_channel_vis.mp4'.format(self.output_dir, target_lm)
                channel_app = channel_visualizer(target_chanel_out_vid_name, fps=input_cap_fps)
                self.target_channel_apps.append(channel_app)
    
        #create output dict to store important metadata and write initial overall metadata
        self.out_json_name = '{}/data.json'.format(self.output_dir)
        self.out_data_dict = {}
        self.out_data_dict['input_video_name'] = input_video 
        now = datetime.now()
        dt_string = now.strftime("%m-%d-%Y-%H-%M-%S")
        self.out_data_dict['analysis_datetime'] = dt_string
        self.out_data_dict['use_initial_detect'] = self.initial_detect
        self.out_data_dict['target_landmarks'] = self.target_landmarks
        self.out_data_dict['main_output_video'] = output_vid_name
        self.out_data_dict['target_region_output_videos'] = target_out_vid_names
        self.out_data_dict['frames'] = {}
    
    def get_auth_output_dir(self):
        return self.output_dir

    def run(self):
        """
        process the input video!
        """
        with tqdm(total=self.tot_input_vid_frames) as pbar:
            pbar.set_description('Running authentication')
            i = 0
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
        for vid in self.target_out_vids:
            vid.release()
        
        if self.track_channels:
            for channel_app in self.target_channel_apps:
                channel_app.out_vid.release()

        json_object = json.dumps(self.out_data_dict, indent=4)
        with open(self.out_json_name, "w") as outfile:
            outfile.write(json_object)
        
    def detect(self, frame):
        raise NotImplementedError("Method detect() must be implemented in child class.")

    def process_frame(self, frame, frame_num):
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

        self.out_data_dict['frames']['frame' + str(frame_num)] = {'initial_detection_made':True}
        landmark_list = self.detect(frame)  
        self.extract_target_regions(init_frame, landmark_list, frame_num, initial_face_bbox)
        annotated_frame = self.annotate(init_frame, landmark_list, initial_face_bbox)
        self.out_vid.write(annotated_frame)
      
    def extract_target_regions(self, frame, landmark_list, frame_num, initial_face_bbox=None):
        if initial_face_bbox is not None:       
            #incremement all landmarks according to initial_face_bbox and crop_padding values to translate to init_frame system
            init_landmark_list = []
            for coord in landmark_list:
                init_landmark_list.append([coord[0] + initial_face_bbox[0] - self.crop_padding, coord[1] + initial_face_bbox[1] - self.crop_padding])
        else:
            init_landmark_list = landmark_list
        
        #extract target regions, writing region as frame to appropriate output video
        self.out_data_dict['frames']['frame' + str(frame_num)]['target_landmark_coords'] = {}
        self.out_data_dict['frames']['frame' + str(frame_num)]['mean_channel_values'] = {}
        for i, l_num in enumerate(self.target_landmarks):
            x, y  = init_landmark_list[l_num]   

            #normalize by initial_face_bbox size
            # initial_face_bbox_dims = (initial_face_bbox[2] - initial_face_bbox[0], initial_face_bbox[3] - initial_face_bbox[1])
            # norm_pattern_width = (self.pattern_rel_width * max(initial_face_bbox_dims))
            left = int(max(0, x - (self.pattern_out_width/2)))
            top = int(max(0, y - (self.pattern_out_width/2)))
            right = int(min(self.W, x + (self.pattern_out_width/2)))
            bottom = int(min(self.H, y + (self.pattern_out_width/2)))
            target_region = frame[top:bottom, left:right, :]
            #now need to resize target_region so it is correct dimensions
            # resized_target_region = cv2.resize(target_region, dsize=(self.pattern_out_width, self.pattern_out_width), interpolation=cv2.INTER_LINEAR)
            # self.target_out_vids[i].write(resized_target_region)
            self.out_data_dict['frames']['frame' + str(frame_num)]['target_landmark_coords'][l_num] = (x, y)
            self.target_out_vids[i].write(target_region)
            if self.track_channels:
                b_data, g_data, r_data = self.target_channel_apps[i].add_frame_data(target_region)
                self.out_data_dict['frames']['frame' + str(frame_num)]['mean_channel_values'][l_num] = {'b':b_data, 'g':g_data, 'r':r_data}
            
    def annotate(self, frame, landmark_list, initial_face_bbox=None):
        if initial_face_bbox is not None:       
            #incremement all landmarks according to initial_face_bbox and crop_padding values to translate to init_frame system
            init_landmark_list = []
            for coord in landmark_list:
                init_landmark_list.append([coord[0] + initial_face_bbox[0] - self.crop_padding, coord[1] + initial_face_bbox[1] - self.crop_padding])
        else:
            init_landmark_list = landmark_list
    
        # shapes = np.zeros_like(frame, np.uint8)
        # out = frame.copy()
        # alpha = 0.5
        # for i in self.target_landmarks:
        #     x, y  = init_landmark_list[i]   
        #     box_start = (int(max(0, x - (self.box_width/2))), int(max(0, y - (self.box_width/2))))
        #     box_end = (int(min(self.W, x + (self.box_width/2))), int(min(self.H, y + (self.box_width/2))))
        #     cv2.rectangle(shapes, box_start, box_end, (255, 255, 0), 2) #add transparent box around landmark in shapes overlay
        # mask = shapes.astype(bool)
        # out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]

        color = (0, 255, 0)
        for l_num in self.target_landmarks:
            x, y  = init_landmark_list[l_num]  
            # initial_face_bbox_dims = (initial_face_bbox[2] - initial_face_bbox[0], initial_face_bbox[3] - initial_face_bbox[1])
            # norm_pattern_width = (self.pattern_rel_width * max(initial_face_bbox_dims))
            box_start = (int(max(0, x - (self.pattern_out_width/2))), int(max(0, y - (self.pattern_out_width/2))))
            box_end = (int(min(self.W, x + (self.pattern_out_width/2))), int(min(self.H, y + (self.pattern_out_width/2))))
            cv2.rectangle(frame, box_start, box_end, color, 2) #draw box around target region
            cv2.circle(frame, (int(x), int(y)), 2, color=color, thickness=-1) #draw landmark in center of target region
        return frame

            
