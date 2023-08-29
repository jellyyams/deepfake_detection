
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
from video_processing.vidgenerator import VidGenerator
from video_processing import mp_alignment
from landmarks import Landmarks
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

class MPFeatureExtractor(object):
    """
    A class for extracting identity and content-related features from a video of a person speaking.

    Attributes
    ----------
    initial_bbox_padding : int
        As described above, if initial detection is used, a crop of the frame determined by the initial face detection bounding box and initial_bbox_padding 
        will be passed to facial landmark extraction. This crop will consist of the intiail face detection bounding box, padded on each
        size by initial_bbox_padding pixels.
    anchor_landmark : int
        Index of landmark to use as anchor when computing landmark distances. This index is zero-based, and the range of possible
        indices depends on the landmark extraction model used (468 for MediaPipe and 68 for FAN)
    target_landmarks : list of int
        List of indices of landmarks to treat as targets. For each frame, the distance of each target landmark to the anchor landmark 
        (in pixels) will be computed. 
    pairs_to_plot : list of int pairs
        List of index pairs that will be plotted in generated video
    three_d_dist : bool
        Whether to compute the 3D Euclidean distance between the targets and anchors or just the 2D Euclidean distance
    generate_video : bool
        Whether to generate a video or not. Set to false for faster run time
    norm_approach: string
        string set to "first_face_bbox", "curr_face_bbox", "first_upper_lower_bbox" that indicates how coords will be normalized per frame
    output_data : list of strings
        list of strings such as "anchor_distances" or "pairwise_distances" that indicates what time of data the extraction will output 
    pairs_to_analyze: list of int pairs
        list of landmark pairs that is used to calculate pairwise distances 
    landmark_coord_tracker : dict
        Holds landmark coordinates for each frame of the input video fo revery landmark in target_landmark. Keys of the landmark_tracker dictionary are landmark indices. 
        The value for each key is a list of 3D tuples corresponding to landmark {key}'s pixel coordinate at each frame. For instance, if 
        the landmark_coord_tracker is
        { 
            0 : [(0, 1, 0), (1, 1, 1)],
            1 : [(2, 0, 0), (0, 1, 1)],
            ....
        }
        it currently contains two video frames worth of data. Landmark 0 was at pixel coordinates (0, 1, 0) at frame 0 and 
        (1, 1, 1) at frame 1, etc.
    landmark_data_tracker : dict
        Similar to landmark_coord_tracker, holds the target-anchor distances for each frame of the input video for every landmark in target_landmark. Keys of the dist_tracker
        dictionary are target landmark indices. The value for each key is a list of ints corresponding to the distance between
        target landmark {key}'s distance from the anchor landmark (in pixels) at each frame. For instance, if 
        the landmark_data_tracker is
        { 
            10 : [5, 10],
            50 : [2, 4]
        }
        it currently contains two video frames worth of data. The target landmarks in use are 10 and 50. 
        Landmark 10's distance from the anchor landmark was 5 at frame 0 and 10 at frame 1, etc. 
    landmark_group_tracker: dict
        Similar to landmark_coord_tracker, holds the pairwise distances for each frame of the input video for every landmark pair in pairs_to_analyze. 
        Keys are landmark pairs in tuple format, and the value for each key is a list of ints corresponding to the distance between the first landmark in a pair
        and the second landmark in a pair. For instance, if the landmark_group_tracker is
        {
            (336, 384) : [2.3, 4.2],
            (296, 385) : [4.1, 8.6],
        }
        it currently contains two video frames worth of data. pairs_to_analyze has two landmark pairs (336, 384) and (296, 385). 
        The distance between 336 and 384 in the first frame is 2.3 and in the second frame is 4.2 etc.
    curr_landmark_ext_fps, tot_det_time, overall_time, overall_fps, curr_overall_fps, total_frames : int
        Keep track of current landmark extraction fps and overall feature extraction fps
    landmarks: Landmarks object
        A Landmarks class object that contains attributes and methods for naming, sorting, and selecting landmarks
    face_detector : RetinaFace object
        Contains all necessary methods/attributes for performing face detection.
    vidgen: VidGenerator object
        A VidGenerator class object that contains attributes and methods for generating an output video 
    frame_num: int
        int that tracks the current frame number
    init_frame: frame
        the first frame of the video
    curr_face_bbox : list of tuples
        list of tuples that stores the current frame's bounding box
    cropped_H, cropped_W : float 
        floats that track the current crop's height and width 
    
    Methods
    -------
    extract_landmarks(frame):
       Extracts facial landmarks, returning a list of 3D coordinates - one for each landmark. 
    align_landmarks(landmark_list, W, H)
        Align extracted landmarks to obtain canonical view of face landmarks
    track_landmarks(landmark_coords, W, H)
        Update landmark_tracker and dist_tracker with new frame's data
    get_output_video_path()
        Return path to generated output video 
    run()
        Perform feature extraction on each frame, managing tracker updates, output video generation, etc. 
    resize_img(img)
        Helper function to resize inputted img to be self.display_dim x self.displa_dim pixels
    plot_dists(frame_num)
        Generate frame showing current trend of landmark distances
    annotate_frame(frame, landmark_list,  initial_face_bbox)
        Annotate frame from input video with all extracted landmarks (in their original form -unaligned)
        and, if initial detection is being ran, the initial face detecting bounding box
    annotate_blank(aligned_landmark_list)
        Annotate blank canvas with aligned landmarks to show canonical view of face 
    """
    
    def __init__(
        self, 
        input_video, 
        target_landmarks,
        pairs_to_analyze,
        pairs_to_plot=[], 
        draw_anchor_target_connector=False,
        norm_approach="first_upper_lower_bbox", 
        output_data=["anchor_distances", "pairwise_distances"],
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
        self.pairs_to_plot = pairs_to_plot
        self.three_d_dist = three_d_dist
        self.generate_video = generate_video
        self.norm_approach = norm_approach
        self.output_data = output_data
        self.pairs_to_analyze = pairs_to_analyze

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

        self.landmarks = Landmarks()
        self.bbox = {
            "upper":[(0,0,0), (0,0,0)], 
            "lower": [(0,0,0), (0,0,0)], 
            "face": [(0,0,0), (0,0,0)]
        }
        self.landmarks = {
            "upper": self.landmarks.upper_landmarks, 
            "lower": self.landmarks.lower_landmarks, 
            "face": self.landmarks.all
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
                self.input_W, 
                dist_display_win_size, 
                draw_anchor_target_connector)

        self.frame_num = 0 
        self.init_frame = None
        self.curr_face_bbox = [(0,0,0), (0,0,0)]
        self.cropped_H = 0.0
        self.cropped_W = 0.0
        
        logging.info("Setting up MediaPipe FaceMesh")
        self.init_mediapipe()
        logging.info('Done setting up MediaPipe FaceMesh')


    def init_mediapipe(self):
        """
        mediapipe extractor initialization
        """
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
        """
        Initalizes video capture settings
        """

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

    def align_landmarks(self, landmarks):
        """
        Aligns landmarks. tbh i dont know what this one does lol  
        """
        landmark_coords_3d_aligned, landmark_coords_2d_aligned  = mp_alignment.align_landmarks(landmarks, self.input_W, self.input_H, self.cropped_W, self.cropped_H)
        return landmark_coords_2d_aligned, landmark_coords_3d_aligned

    def get_diff(self, bbox, xdiff, ydiff, zdiff):
        """
        helper function that finds diff between coords of bounding box to be used in normalizing coord data
        """
        bbox_W = bbox[1][0] - bbox[0][0]
        bbox_H = bbox[1][1] - bbox[0][1]
        bbox_D = bbox[1][2] - bbox[0][2]
        xdiff = xdiff / bbox_W
        ydiff = ydiff / bbox_H
        zdiff = zdiff / bbox_D

        return xdiff, ydiff, zdiff
    
    def get_first_bbox(self, region):
        """
        gets the bounding box of the first frame
        """
        bbox = self.bbox[region]

        if bbox[1][0] == bbox [0][0] == bbox[1][1] == bbox[0][1] == bbox[1][2] == bbox[0][2] == 0:
            bbox = self.get_curr_bbox(self.landmarks[region])
            self.bbox[region] = bbox 
        
        return bbox
    
    def get_curr_bbox(self, landmarks):
        """
        gets the bounding box of the current frame
        """
        cx_min=  self.cropped_W
        cy_min = self.cropped_H
        cz_min = self.cropped_W #z scale is roughly same as x scale, according to https://medium.com/@susanne.thierfelder/head-pose-estimation-with-mediapipe-and-opencv-in-javascript-c87980df3acb
        cx_max = cy_max = cz_max = 0
        for id, l in enumerate(landmarks):
            lm = self.curr_frame_aligned_landmarks_3d[l]

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


    def normalize(self, xdiff, ydiff, zdiff, i):
        """
        normalies coordinate data based on specified approach
        """
        if self.norm_approach == "curr_face_bbox": 
            #normalize all differences by current face bounding box dimensions
            bbox = self.get_curr_bbox(self.landmarks["face"])
           
        elif self.norm_approach == "first_upper_lower_bbox":
            #normalize all differences by the first frame's upper or lower bounding box dimensions
            if i in self.landmarks["upper"]:
                bbox = self.get_first_bbox("upper")
            else:
                bbox = self.get_first_bbox("lower")
        elif self.norm_approach == "first_face_bbox":
            #normalize all differences by the first frame's face bounding box dimensions
            bbox = self.get_first_bbox("face")
            
        xdiff, ydiff, zdiff = self.get_diff(bbox, xdiff, ydiff, zdiff)

        return xdiff, ydiff, zdiff

    def set_landmark_dist(self, coord1, coord2, key, tracker, normlandmark):
        """
        finds the coord distance difference between two landmarks, used in both anchor and pairwise distance tracking
        """
        x_diff = (coord1[0] - coord2[0]) 
        y_diff = (coord1[1] - coord2[1]) 
        z_diff = (coord1[2] - coord2[2]) 


        x_diff, y_diff, z_diff = self.normalize(x_diff, y_diff, z_diff, normlandmark)
            
        if self.three_d_dist:
            dist = np.sqrt(x_diff**2 + y_diff**2 + z_diff**2) 
        else:
            dist = np.sqrt(x_diff**2 + y_diff**2) 

        if key not in tracker:
            tracker[key] = [dist]
        else:
            tracker[key].append(dist)  
        
    
    def set_landmarks_none(self):
        """
        Adds None to lists to maintain alignment of list values with frame numbers if there were no landmarks extracted for this frame
        """

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
    
    def track_landmark_data(self):
        """
        Adds appropriate coordinates and distances to landmark_tracker and dist_tracker if landmarks were extracted for this frame
        """
        anchor_coord = self.curr_frame_aligned_landmarks_3d[self.anchor_landmark]
        for i, l in enumerate(self.curr_frame_aligned_landmarks_3d):
            if i not in self.landmark_coord_tracker:
                self.landmark_coord_tracker[i] = [l]
            else:
                self.landmark_coord_tracker[i].append(l)

            
            if "anchor_distances" in self.output_data and i in self.target_landmarks:
                self.set_landmark_dist(anchor_coord, l, i, self.landmark_data_tracker, i)


    def track_landmark_groups(self):
        """ 
        Tracks landmark data where two or more landmark coordinates are needed, for instance tracking pairwise distances
        """
        if "pairwise_distances" in self.output_data:
            for pair in self.pairs_to_analyze:
                coord1 = self.curr_frame_aligned_landmarks_3d[pair[0]]
                coord2 = self.curr_frame_aligned_landmarks_3d[pair[1]]
                if (pair[0] in self.landmarks["upper"] and pair[1] in self.landmarks["upper"]) or (pair[0] in self.landmarks["lower"] and pair[1] in self.landmarks["lower"]):
                    self.norm_approach = "first_upper_lower_bbox"
                else: 
                    self.norm_approach = "first_face_bbox"

                self.set_landmark_dist(coord1, coord2, pair, self.landmark_group_tracker, pair[0])

        elif "x" in self.output_data:
            #if you want to track other group data (i.e. pairwise velocities or angles), implement here
            pass 
    
    def track_landmarks(self, landmarks, frame):
        """
        Tracks landmark data for individual landmarks
        """
        if len(landmarks) != 0:

            self.cropped_H, self.cropped_W, c = frame.shape #use h, w defined here instead of self.input_W, self.input_H because they are not the same if initial face deteciton is being used

            self.curr_frame_aligned_landmarks_2d, self.curr_frame_aligned_landmarks_3d = self.align_landmarks(landmarks)

            if self.curr_frame_aligned_landmarks_3d == None:
                self.set_landmarks_none()
            else:
                self.track_landmark_data()

    
    def crop_frame(self, frame):
        """
        Crops in on speaker's face in video frame
        """
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
        """
        handles video frame generation and tracking alignment if no landmarks were detected for a frame
        """
        self.track_landmarks()
        self.vidgen.set_plot_frame(self.frame_num, self.target_landmarks, self.landmark_data_tracker)
        self.vidgen.set_annotated_blank()
        self.vidgen.set_annotated_frame(self.init_frame)


    def detect_landmarks(self, frame):
        """
        Detects landmarks for one frame and tracks how long it took
        """
        start = time.time()
        landmark_list = self.extract_landmarks(frame)
        end = time.time()

        self.tot_det_time += (end - start)
        self.curr_landmark_ext_fps = self.frame_num / self.tot_det_time

        return landmark_list
    
          
            
    def plot_landmarks(self, landmarks): 
        '''
        create plots for annotated frame, annotated blank, and graph for each frame
        '''
        if len(landmarks) != 0: 
          
            self.vidgen.set_annotated_frame(self.init_frame, landmarks, self.curr_face_bbox, self.anchor_landmark, self.target_landmarks, self.landmark_data_tracker)
            self.vidgen.set_annotated_blank(self.curr_frame_aligned_landmarks_2d, self.target_landmarks, self.anchor_landmark, self.landmark_data_tracker)
            
            # self.vidgen.set_plot_frame(self.frame_num, self.target_landmarks, self.landmark_data_tracker)
            self.vidgen.set_plot_pairs_frame(self.frame_num, self.pairs_to_plot, self.landmark_group_tracker)
        else: 
            self.empty_frame()


    def update_fps(self, overall_start):
        """
        updates frames per second
        """
        overall_end = time.time()
        self.tot_overall_time += (overall_end - overall_start)
        self.curr_overall_fps = self.frame_num / self.tot_overall_time

    def analyze_single_frame(self):
        """
        Runs mediapipe extraction and data tracking on a single frame
        """
        ret, frame = self.input_capture.read()
        if ret: 
            overall_start = time.time()
            self.frame_num += 1
            self.init_frame = frame.copy()

            self.curr_face_bbox = self.face_detector.detect(frame)
            frame = self.crop_frame(frame)
            landmark_list = []

            if frame is not None: 
                landmark_list = self.detect_landmarks(frame)

                self.track_landmarks(landmark_list, frame)
                self.track_landmark_groups()  

            if self.generate_video:
                self.plot_landmarks(landmark_list)
                self.vidgen.write_combined()

            self.update_fps(overall_start)
            return True 
       
        return False  

    
    def run(self):
        """
        main function that runs this class
        """

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


    


        



        




