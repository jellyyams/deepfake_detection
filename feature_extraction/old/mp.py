
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
from feature_extractor import FeatureExtractor
import mp_alignment
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2



class MPFeatureExtractor(FeatureExtractor):
    def __init__(
        self, 
        input_video, 
        output_directory, 
        initial_detect=True, 
        initial_bbox_padding = 30, 
        three_d_dist=False, 
        dist_display_win_size = 60,  
        draw_all_landmarks = False, 
        draw_landmark_nums=False, 
        draw_anchor_target_connector=True, 
        display_dim=800, 
        log_level='INFO', 
        anchor_landmark=0, 
        target_landmarks=[269, 267, 39, 37, 181, 314], 
        generate_video=True, 
        norm_approach="face bbox", 
        tracking="landmark_to_anchor", 
        landmark_pairs=[(0, 4), (0, 17)]):
        
        FeatureExtractor.__init__(
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
            tracking, 
            landmark_pairs)

        # mediapipe extractor initialization
        logging.info('Setting up MediaPipe FaceMesh')
        base_options = python.BaseOptions(model_asset_path='../common/weights/face_landmarker_v2_with_blendshapes.task')
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                            num_faces=1, 
                                             min_face_detection_confidence=.25, 
                                            min_face_presence_confidence=.25, 
                                            min_tracking_confidence=.25)
        #output_face_blendshapes=True,
        #output_facial_transformation_matrixes=True,
        self.extractor = vision.FaceLandmarker.create_from_options(options)
        logging.info('Done setting up MediaPipe FaceMesh')

    
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
    
    def align_landmarks(self, landmark_list, W, H):
        """
        Align extracted landmarks in landmark_list to obtain canonical view of face landmarks

        Parameters
        ----------
        landmark_list : List of 3D tuples
            3D coordinate of each face landmark, as outputted by extract_landmarks
        W, H : int
            Dimensions, in pixels, of frame that facial landmark extraction was run on.
            This is not same as self.input_W, self.input_H f if initial face detection (and thus cropping) is being used!

        Returns
        -------
        landmark_coords_3d_aligned, landmark_coords_2d_aligned : List of 3D tuples, list of 2D tuples
            3D face landmark coordinates in canonical view, and corresponding 2D coordinates, derived
            by projecting the aligned 3D coordinates assuming zero camera rotation/translation
        """
        landmark_coords_3d_aligned, landmark_coords_2d_aligned  = mp_alignment.align_landmarks(landmark_list, self.input_W, self.input_H, W, H)
        return landmark_coords_2d_aligned, landmark_coords_3d_aligned


    def set_landmarks_none(self):
        #if there were no landmarks extracted for this frame, add None to lists to maintain alignment of list values with frame numbers

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

    def track_landmarks_to_anchor(self, landmark_coords, W, H):
        #if landmarks were extracted for this fame, add appropriate coordinates and distances to landmark_tracker and dist_tracker
        anchor_coord = landmark_coords[self.anchor_landmark]
        for i, l in enumerate(landmark_coords):
            if i not in self.landmark_tracker:
                self.landmark_tracker[i] = [l]
            else:
                self.landmark_tracker[i].append(l)
            if i in self.target_landmarks:
                self.set_landmark_dist(anchor_coord, landmark_coords, l, W, H, i)

    def track_landmark_pairs(self, landmark_coords, W, H):
        for pair in self.landmark_pairs:
            pass

    
    def track_landmarks(self, landmark_coords=None, W=None, H=None):
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
        if landmark_coords == None:
            self.set_landmarks_none()
        elif self.tracking == "landmark_to_anchor":
            self.track_landmarks_to_anchor(landmark_coords, W, H)
        elif self.tracking == "landmark_pairs": 
            self.set_landmarks_none()
        
