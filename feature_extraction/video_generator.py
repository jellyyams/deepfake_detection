import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt

import pickle

class VidGenerator:

    def __init__(self, output_directory, filename, display_dim, input_cap_fps, frames_to_include, initial_bbox_padding, draw_landmark_nums, draw_all_landmarks, input_H, input_W, dist_display_win_size=60):

        if not os.path.exists(output_directory):
            os.makedirs(output_directory) 
        
        self.dist_display_win_size = dist_display_win_size
        self.output_path = '{}/{}.mp4'.format(output_directory, filename)
        self.frames = self.init_frames(frames_to_include)
        self.initial_bbox_padding = initial_bbox_padding
        self.display_dim = display_dim
        self.draw_landmark_nums = draw_landmark_nums
        self.draw_all_landmarks = draw_all_landmarks
        self.input_W = input_W
        self.input_H = input_H

        self.blank_canvas = cv2.cvtColor(np.uint8(np.ones((self.input_H, self.input_W))), cv2.COLOR_GRAY2BGR)

        #drawing stuff
        with open('../common/colors.pkl', 'rb') as f:
            self.drawing_colors = pickle.load(f)

        self.out_vid = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*'MP4V'), input_cap_fps, (display_dim*3, display_dim))
    
    def init_frames(self, frames_to_include):
        frames = {}
        for f in frames_to_include:
            frames[f] = None
        
        return frames

    def release_vid(self):
        self.out_vid.release()
    
    def write_combined(self):
        tup = (v for k,v in self.frames.items())
        # combined = np.hstack(tup)
        # combined = np.hstack((self.frames["annotated_vid"], self.frames["annotated_blank"], self.frames["anchor_plot"]))
        combined = np.hstack((self.frames["annotated_vid"], self.frames["annotated_blank"], self.frames["pair_plot"]))
        self.out_vid.write(combined)

    
    def set_annotated_frame(self, frame, landmarks = None, init_bbox = None, anchor_num=None, target_landmarks=None, data_tracker=None):
        if landmarks and init_bbox and anchor_num and target_landmarks and data_tracker: 
            frame = self.annotate_frame(frame, landmarks, init_bbox, anchor_num, target_landmarks, data_tracker)
    
        self.frames["annotated_vid"] = self.resize_img(frame)
    
    def set_annotated_blank(self, aligned_landmarks = None, target_landmarks=None, anchor_num=None, data_tracker=None):
        if aligned_landmarks and target_landmarks and anchor_num and data_tracker: 
            annotated = self.annotate_blank(aligned_landmarks, target_landmarks, data_tracker, anchor_num)
            self.frames["annotated_blank"] = self.resize_img(annotated)
        else: 
            self.frames["annotated_blank"] = self.resize_img(self.blank_canvas.copy())
    
    def set_plot_frame(self, frame_num, target_landmarks, data_tracker):
        self.frames["anchor_plot"] = self.plot_data(frame_num, target_landmarks, data_tracker)

    def set_plot_pairs_frame(self, frame_num, target_pairs, pair_tracker):
        self.frames["pair_plot"] = self.plot_data(frame_num, target_pairs, pair_tracker)

    
    def annotate_frame(self, frame, landmark_list, initial_face_bbox, anchor_num, target_landmarks, data_tracker):
        """
        Annotate frame from input video with all extracted landmarks (in their original form -unaligned)
        and, if initial detection is being ran, the initial face detecting bounding box

        If initial_face_bbox = None, then initial face detection is not being run. 
        """
        if initial_face_bbox:
            #draw face bounding box
            cv2.rectangle(frame, (initial_face_bbox[0] - self.initial_bbox_padding, initial_face_bbox[1] - self.initial_bbox_padding), (initial_face_bbox[2] + self.initial_bbox_padding, initial_face_bbox[3] + self.initial_bbox_padding), (255, 255, 0), 2)
            #incremement all landmarks according to initial_face_bbox and initial_bbox_padding values to translate to init_frame system
            init_landmark_list = []
            for coord in landmark_list:
                init_landmark_list.append([coord[0] + initial_face_bbox[0] - self.initial_bbox_padding, coord[1] + initial_face_bbox[1] - self.initial_bbox_padding])
        else:
            init_landmark_list = landmark_list

        anchor = (int(init_landmark_list[anchor_num][0]), int(init_landmark_list[anchor_num][1]))
        for i, coord in enumerate(init_landmark_list):
            if i not in target_landmarks and i != anchor_num and not self.draw_all_landmarks:
                #unless draw_all_landmarks=True, only draw anchor/target landmarks 
                continue  
            x = int(coord[0])
            y = int(coord[1])
            if i == anchor_num:
                color = (255, 255, 255)
            elif i in target_landmarks:
                color = self.drawing_colors[i][::-1]

                frame = cv2.line(frame, (x, y), anchor, color, 1)
                d = data_tracker[i][-1]
                cv2.putText(frame, '{:.2f}'.format(d), (x, y), cv2.FONT_HERSHEY_DUPLEX, .5, color)
            else:
                color = (255, 255, 255)
            frame = cv2.circle(frame, (x, y), 2, color=color, thickness=-1)
            if self.draw_landmark_nums:
                cv2.putText(frame, f'{i}', (x-5, y-5), cv2.FONT_HERSHEY_DUPLEX, 0.25, color)
        return frame
    

    def annotate_blank(self, aligned_landmark_list, target_landmarks, data_tracker, anchor_num):
        """
        Annotate blank canvas with aligned landmarks to show canonical view of face 
        """

        blank = self.blank_canvas.copy()

        anchor = (int(aligned_landmark_list[anchor_num][0]), int(aligned_landmark_list[anchor_num][1]))
        for i, coord in enumerate(aligned_landmark_list):
            if i not in target_landmarks and i != anchor_num and not self.draw_all_landmarks:
                #unless draw_all_landmarks=True, only draw anchor/target landmarks 
                continue
            x = int(coord[0])
            y = int(coord[1])
            if i == anchor_num:
                color = (255, 255, 255)
            elif i in target_landmarks:
                color = self.drawing_colors[i][::-1]

                blank = cv2.line(blank, (x, y), anchor, color, 1)
                d = data_tracker[i][-1]
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


    def plot_data(self, frame_num, target_landmarks, data_tracker):
        """
        Generate frame showing current trend of landmark distances
        """

        for t in target_landmarks:
            dists = data_tracker[t]
            if "(" in str(t): 
                a = int(t[0])
                b = int(t[1])
                color = a

            else:
                color = t
            c = np.array(self.drawing_colors[color])/255
            plt.plot(dists, color=c, label=str(t))

        if frame_num < self.dist_display_win_size: 
            plt.xlim(0, self.dist_display_win_size)
        else: 
            plt.xlim(frame_num - int(self.dist_display_win_size/2), frame_num + int(self.dist_display_win_size/2))


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


    
  

