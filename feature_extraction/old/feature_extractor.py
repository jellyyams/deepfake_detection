
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

class FeatureExtractor(object):
    """
    A class for extracting identity and content-related features from a video of a person speaking.

    Attributes
    ----------
    initial_detect : bool
        Whether to run initial face detection prior to facial landmark extraction. If initial detection is used, a crop of the frame
        determined by the initial face detection bounding box and initial_bbox_padding and will be passed to facial landmark extraction. Otherwise, the whole
        frame will be passed to facial landmark extraction. 
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
    three_d_dist : bool
        Whether to compute the 3D Euclidean distance between the targets and anchors or just the 2D Euclidean distance
    dist_display_win_size : int
        When displaying the animated plot of target-anchor distances, a window of size dist_display_win_size will roll
        over the plot of target-anchor distances vs. time.
    draw_all_landmarks : bool
        Whether to draw all landmarks on the (as opposed to just target and anchor landmarks) in the output video's blank canvas
    draw_landmark_nums : bool
        Whether to draw the number of the landmark next to the landmark in the output video
    draw_anchor_target_connector : bool
        Whether to draw a line connecting each target landmark to the anchor landmark in the output video
    display_dim : int
        Each frame of the outputted video will consist of the annoted input video frame, the annotated blank frame, and a distance plot frame, all side-by-side.
        Each of these sub-frames will be display_dim x display_dim (W x H) pixels, for an overall frame dimension of 3*display_dim x display_dim (W x H) pixels.


    landmark_tracker : dict
        Holds landmark coordinates for each frame of the input video. Keys of the landmark_tracker dictionary are landmark indices. 
        The value for each key is a list of 3D tuples corresponding to landmark {key}'s pixel coordinate at each frame. For instance, if 
        the landmark_tracker is
        { 
            0 : [(0, 1, 0), (1, 1, 1)],
            1 : [(2, 0, 0), (0, 1, 1)],
            ....
        }
        it currently contains two video frames worth of data. Landmark 0 was at pixel coordinates (0, 1, 0) at frame 0 and 
        (1, 1, 1) at frame 1, etc.
    dist_tracker : dict
        Similar to landmark_tracker, holds the target-anchor distances for each frame of the input video. Keys of the dist_tracker
        dictionary are target landmark indices. The value for each key is a list of ints corresponding to the distance between
        target landmark {key}'s distance from the anchor landmark (in pixels) at each frame. For instance, if 
        the dist_tracker is
        { 
            10 : [5, 10],
            50 : [2, 4]
        }
        it currently contains two video frames worth of data. The target landmarks in use are 10 and 50. 
        Landmark 10's distance from the anchor landmark was 5 at frame 0 and 10 at frame 1, etc. 
    input_capture : cv2 VideoCapture object
        cv2 VideoCapture object for input_video
    output_video_path : str
        Path to output video 
    curr_landmark_ext_fps, tot_det_time, overall_time, overall_fps, curr_overall_fps : int
        Keep track of current landmark extraction fps and overall feature extraction fps
    face_detector : RetinaFace object
        Contains all necessary methods/attributes for performing face detection.
    blank_canvas : np array/cv2 frame
        A blank black frame to draw on
    drawing_colors : list of tuples
        List of RGB tuples used for plotting distances
    
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
    run_extraction()
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
        output_directory, 
        initial_detect, 
        initial_bbox_padding, 
        anchor_landmark, 
        target_landmarks, 
        three_d_dist, 
        dist_display_win_size, 
        draw_all_landmarks, 
        draw_landmark_nums, 
        draw_anchor_target_connector, 
        display_dim, 
        log_level, 
        generate_video, 
        norm_approach, 
        tracking
        ):
        """
        Constructs all necessary attributes for the FeatureExtractor object.

        input_video : str
            Path to input video to run feature extraction on
        output_directory : str
            Path to directory to store output in
        landmark_model : str
            The facial landmark extraction landmark_model to use. Either 'mp', corresponding to Mediapipe, or 'fan', corresponding to FAN 
        log_level : str
            'INFO', 'DEBUG', 'WARNING', etc.
        The rest of the inputs initialize the attributes of the same name (described above)
        """

        logging.config.dictConfig(generate_logging_config('feature extractor', log_level))

        # initialize attributes with values passed into constructor
        self.initial_detect = initial_detect
        self.initial_bbox_padding = initial_bbox_padding
        self.anchor_landmark = anchor_landmark
        self.target_landmarks = target_landmarks
        self.landmark_pairs = landmark_pairs
        self.three_d_dist = three_d_dist
        self.draw_all_landmarks = draw_all_landmarks
        self.draw_landmark_nums = draw_landmark_nums
        self.draw_anchor_target_connector = draw_anchor_target_connector
        self.dist_display_win_size = dist_display_win_size
        self.display_dim = display_dim
        self.generate_video = generate_video
        self.norm_approach = norm_approach
        self.tracking = tracking

        # initialize trackers
        self.landmark_tracker = {}
        self.dist_tracker = {}

        #initialize speed benchmarking attributes
        self.curr_landmark_ext_fps = 0
        self.tot_det_time = 0
        self.tot_overall_time = 0
        self.curr_overall_fps = 0
        self.tot_frames = 0

        self.bbox = {
            "upper":[(0,0,0), (0,0,0)], 
            "lower": [(0,0,0), (0,0,0)], 
            "upper_right": [(0,0,0), (0,0,0)], 
            "lower_right": [(0,0,0), (0,0,0)],
            "lower_left": [(0,0,0), (0,0,0)], 
            "upper_left": [(0,0,0), (0,0,0)],  
            "face": [(0,0,0), (0,0,0)]
        }
        self.mesh_data = MeshData()
        self.landmarks = {
            "upper": self.mesh_data.upper_landmarks, 
            "lower": self.mesh_data.lower_landmarks, 
            "upper_left": self.mesh_data.upper_left_landmarks,
            "upper_right": self.mesh_data.upper_right_landmarks,
            "lower_left": self.mesh_data.lower_left_landmarks,
            "lower_right": self.mesh_data.lower_right_landmarks,
            "face": self.mesh_data.all
        }

        #set up initial face detector, if using
        if self.initial_detect == True:
            logging.info('Initializing RetinaFace Detector')
            self.face_detector = RetinaFaceDetector("resnet50")
            logging.info('Done initializing RetinaFace Detector')

        try:
            self.input_capture = cv2.VideoCapture(input_video)
        except:
            raise ValueError('Input video path %s not valid' % input_video)
        
        if self.input_capture is not None:    
            self.input_W , self.input_H = int(self.input_capture.get(3)), int(self.input_capture.get(4)) #input video dimensions
        else:
            raise ValueError("Invalid input video")

        # get output video set up
        #create output directory if it doesn't already exist
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        input_vid_name = input_video.split('/')[-1][:-4] #name of video (remove extension)
        if self.initial_detect:
            self.output_video_path = '{}/initdet_{}.mp4'.format(output_directory, input_vid_name)
        else:
            self.output_video_path = '{}/{}.mp4'.format(output_directory, input_vid_name)
        input_cap_fps = int(self.input_capture.get(cv2.CAP_PROP_FPS))
        if generate_video: 
            self.out_vid = cv2.VideoWriter(self.output_video_path, cv2.VideoWriter_fourcc(*'MP4V'), input_cap_fps, (self.display_dim*3, self.display_dim))
            logging.info('Set up output video at {}'.format(self.output_video_path))

        #drawing stuff
        self.blank_canvas = cv2.cvtColor(np.uint8(np.ones((self.input_H, self.input_W))), cv2.COLOR_GRAY2BGR)
        with open('../common/colors.pkl', 'rb') as f:
            self.drawing_colors = pickle.load(f)

    def extract_landmarks(self, frame):
        """
        Extracts facial landmarks, returning a list of 3D coordinates - one for each landmark. 
        """
        raise NotImplementedError("Method extract_landmarks() must be implemented in child class.")

    def align_landmarks(self, landmark_list, W, H):
        """
        Align extracted landmarks in landmark_list to obtain canonical view of face landmarks
        """
        raise NotImplementedError("Method align_landmarks() must be implemented in child class.")

    def track_landmarks(self, landmark_coords=None, W=None, H=None):
        """
        Update landmark_tracker and dist_tracker with new frame's data
        """
        raise NotImplementedError("Method track_landmarks() must be implemented in child class.")



    def get_frame(self, initial_face_bbox, frame):
        if initial_face_bbox == None:
            return None
        else:
            # get crop of frame to pass to facial landmark extraction
            bottom = max(initial_face_bbox[1] - self.initial_bbox_padding, 0)
            top = min(initial_face_bbox[3]+1+ self.initial_bbox_padding, self.input_H)
            left = max( initial_face_bbox[0]-self.initial_bbox_padding, 0)
            right = min(initial_face_bbox[2]+1+self.initial_bbox_padding, self.input_W)
            return frame[bottom:top,left:right]


    def run_extraction(self):
        """
        Perform feature extraction on each frame, managing tracker updates, output video generation, etc. 
        """
        frame_num = 0
        tot_input_vid_frames = int(self.input_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        with tqdm(total=tot_input_vid_frames) as pbar:
            pbar.set_description('Performing feature extraction ')
            while self.input_capture.isOpened():
                ret, frame = self.input_capture.read()
                if ret:
                    overall_start = time.time()
                    frame_num += 1
                    #make copy of initial frame fetched from input video, before potentially cropping
                    init_frame = frame.copy() 
                    #run initial face detection
                    initial_face_bbox = self.face_detector.detect(frame)
                    frame = self.get_frame(initial_face_bbox, frame)
        

                    if frame is None: 
                        #if no face was detected with initial detection, write unannotated/blank frame to output video
                        annotated_frame = self.resize_img(init_frame)
                        annotated_blank = self.resize_img(self.blank_canvas.copy())
                        self.track_landmarks() #must be called before plot_dists, because plot_dists uses the tracker variables
                        if self.generate_video: dist_plot = self.plot_dists(frame_num)
                    else:
                        #run facial landmark detection 
                        start = time.time()
                        landmark_list = self.extract_landmarks(frame)  
                        end = time.time() 

                        #update landmark extraction fps
                        self.tot_det_time += (end - start)
                        self.curr_landmark_ext_fps = frame_num / self.tot_det_time

                        if len(landmark_list) == 0:
                            annotated_frame = self.resize_img(init_frame)
                            annotated_blank = self.resize_img(self.blank_canvas.copy())
                            self.track_landmarks()
                            if self.generate_video: dist_plot = self.plot_dists(frame_num)
                        else:
                            H, W, c = frame.shape #use h, w defined here instead of self.input_W, self.input_H because they are not the same if initial face deteciton is being used
                            aligned_landmark_list_2d, aligned_landmark_list_3d = self.align_landmarks(landmark_list, W, H)
                            
                            #Question for self: should i be passing aligned_landmark_list_3d or aligned_landmark_list_2d to update landmark_tracker is the question...
                            #Answer: if 3D distances are being calculated (i.e., three_d = True), obviously aligned 3D ones should be passed in. 
                            # If 2D distances are being calculated, the only difference between the x, y coordiantes in 
                            # aligned_landmark_list_3d and aligned_landmark_list_2d would be scaling, since aligned_landmark_list_2d are derived by projecting
                            # aligned_landmark_list_3D with a camera matrix of zero translation/rotation and fixed depth...
                            self.track_landmarks(aligned_landmark_list_3d, W, H)
                            if self.generate_video: 
                                annotated_frame = self.resize_img(self.annotate_frame(init_frame, landmark_list, initial_face_bbox))
                                annotated_blank =  self.resize_img(self.annotate_blank(aligned_landmark_list_2d))
                                dist_plot = self.plot_dists(frame_num)
                    
                    if self.generate_video: 
                        #write frame to output video
                        combined = np.hstack((annotated_frame, annotated_blank, dist_plot))       
                        self.out_vid.write(combined)

                    #update overall fps
                    overall_end = time.time()
                    self.tot_overall_time += (overall_end - overall_start)
                    self.curr_overall_fps = frame_num / self.tot_overall_time      
                else:
                    break
                pbar.update(1)

        print('Average extraction FPS: ', self.curr_landmark_ext_fps)
        print('Average overall FPS: ', self.curr_overall_fps)
        
        if self.generate_video:
            self.input_capture.release()
            self.out_vid.release()

        return self.dist_tracker
         
    def plot_dists(self, frame_num):
        """
        Generate frame showing current trend of landmark distances
        """

        for t in self.target_landmarks:
            dists = self.dist_tracker[t]
            c = np.array(self.drawing_colors[t])/255
            plt.plot(dists, color=c, label=str(t))
           

        if frame_num < self.dist_display_win_size:
            plt.xlim(0, self.dist_display_win_size)
        else:
            plt.xlim(0, frame_num + int(self.dist_display_win_size/2))

        if self.norm_approach == "face bbox":
            plt.ylim(0, 1)

        plt.legend()
        figure = plt.gcf()
        # set output figure size in pixels
        # https://stackoverflow.com/questions/332289/how-do-i-change-the-size-of-figures-drawn-with-matplotlib/4306340#4306340
        #below assumies dpi=100 
        figure.set_size_inches(0.01*self.display_dim, 0.01*self.display_dim)
        figure.canvas.draw()
        fig_img = np.array(figure.canvas.buffer_rgba())
        fig_img = cv2.cvtColor(fig_img, cv2.COLOR_RGBA2BGR)
        plt.clf()
        
        return fig_img


    def annotate_frame(self, frame, landmark_list, initial_face_bbox=None):
        """
        Annotate frame from input video with all extracted landmarks (in their original form -unaligned)
        and, if initial detection is being ran, the initial face detecting bounding box

        If initial_face_bbox = None, then initial face detection is not being run. 
        """
        if initial_face_bbox is not None:
            #draw face bounding box
            cv2.rectangle(frame, (initial_face_bbox[0] - self.initial_bbox_padding, initial_face_bbox[1] - self.initial_bbox_padding), (initial_face_bbox[2] + self.initial_bbox_padding, initial_face_bbox[3] + self.initial_bbox_padding), (255, 255, 0), 2)
            #incremement all landmarks according to initial_face_bbox and initial_bbox_padding values to translate to init_frame system
            init_landmark_list = []
            for coord in landmark_list:
                init_landmark_list.append([coord[0] + initial_face_bbox[0] - self.initial_bbox_padding, coord[1] + initial_face_bbox[1] - self.initial_bbox_padding])
        else:
            init_landmark_list = landmark_list

        anchor = (int(init_landmark_list[self.anchor_landmark][0]), int(init_landmark_list[self.anchor_landmark][1]))
        for i, coord in enumerate(init_landmark_list):
            if i not in self.target_landmarks and i != self.anchor_landmark and not self.draw_all_landmarks:
                #unless draw_all_landmarks=True, only draw anchor/target landmarks 
                continue  
            x = int(coord[0])
            y = int(coord[1])
            if i == self.anchor_landmark:
                color = (255, 255, 255)
            elif i in self.target_landmarks:
                color = self.drawing_colors[i][::-1]

                if self.draw_anchor_target_connector:
                    frame = cv2.line(frame, (x, y), anchor, color, 1)
                    d = self.dist_tracker[i][-1]
                    cv2.putText(frame, '{:.2f}'.format(d), (x, y), cv2.FONT_HERSHEY_DUPLEX, .5, color)
            else:
                color = (255, 255, 255)
            frame = cv2.circle(frame, (x, y), 2, color=color, thickness=-1)
            if self.draw_landmark_nums:
                cv2.putText(frame, f'{i}', (x-5, y-5), cv2.FONT_HERSHEY_DUPLEX, 0.25, color)
        return frame
    

    def annotate_blank(self, aligned_landmark_list):
        """
        Annotate blank canvas with aligned landmarks to show canonical view of face 
        """

        blank = self.blank_canvas.copy()

        anchor = (int(aligned_landmark_list[self.anchor_landmark][0]), int(aligned_landmark_list[self.anchor_landmark][1]))
        for i, coord in enumerate(aligned_landmark_list):
            if i not in self.target_landmarks and i != self.anchor_landmark and not self.draw_all_landmarks:
                #unless draw_all_landmarks=True, only draw anchor/target landmarks 
                continue
            x = int(coord[0])
            y = int(coord[1])
            if i == self.anchor_landmark:
                color = (255, 255, 255)
            elif i in self.target_landmarks:
                color = self.drawing_colors[i][::-1]
                if self.draw_anchor_target_connector:
                    blank = cv2.line(blank, (x, y), anchor, color, 1)
                    d = self.dist_tracker[i][-1]
                    cv2.putText(blank, '{:.2f}'.format(d), (x, y), cv2.FONT_HERSHEY_DUPLEX, .5, color)
            else:
                color = (255, 255, 255)
            blank = cv2.circle(blank, (x, y), 2, color=color, thickness=-1)
            if self.draw_landmark_nums:
                cv2.putText(blank, f'{i}', (x-5, y-5), cv2.FONT_HERSHEY_DUPLEX, 0.25, color)
        return blank
    
    def resize_img(self, img):
        """
        Helper function to resize inputted img to be self.display_dim x self.display_dim pixels
        
        From https://stackoverflow.com/questions/57233910/resizing-and-padding-image-with-specific-height-and-width-in-python-opencv-gives
        """
        w, h = self.input_W, self.input_H

        pad_bottom, pad_right = 0, 0
        ratio = w / h

        if h > self.display_dim or w > self.display_dim:
            # shrinking image algorithm
            interp = cv2.INTER_AREA
        else:
            # stretching image algorithm
            interp = cv2.INTER_CUBIC

        w = self.display_dim
        h = round(w / ratio)

        if h > self.display_dim:
            h = self.display_dim
            w = round(h * ratio)
        pad_top = int(abs(self.display_dim - h)/2)

        if 2*pad_top + h != self.display_dim:
            pad_bottom = pad_top + 1
        else:
            pad_bottom = pad_top
        pad_right = int(abs(self.display_dim - w)/2)
    
        if 2*pad_right + w != self.display_dim:
            pad_left = pad_right + 1
        else:
            pad_left = pad_right

        scaled_img = cv2.resize(img, (w, h), interpolation=interp)
        padded_img = cv2.copyMakeBorder(scaled_img,pad_top,pad_bottom,pad_left,pad_right,borderType=cv2.BORDER_CONSTANT,value=[0,0,0])
        return padded_img
  

