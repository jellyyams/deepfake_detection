import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
from tqdm import tqdm

class channel_visualizer():
    def __init__(self, input_video_path, outpath, colorspace = 'ycrbr', win_size=60,  plot_W = 640, plot_H = 480):
        self.win_size = win_size
        self.plot_W = plot_W
        self.plot_H =  plot_H
       
        try:
            self.input_capture = cv2.VideoCapture(input_video_path)
        except:
            raise ValueError('Input video path %s not valid' % input_video_path)
       
        if self.input_capture is not None:  
            self.tot_input_vid_frames = int(self.input_capture.get(cv2.CAP_PROP_FRAME_COUNT))  
            self.input_cap_fps = int(self.input_capture.get(cv2.CAP_PROP_FPS))
            self.W , self.H = int(self.input_capture.get(3)), int(self.input_capture.get(4)) #input video dimensions
        else:
            raise ValueError("Invalid input video")
        
        self.chan1 = []
        self.chan2 = []
        self.chan3 = []

        self.chan12 = []
        self.chan13 = []
        self.chan23 = []

        self.num_pixels = self.W * self.H
        for i in range(self.num_pixels):
            self.chan1.append([])
            self.chan2.append([])
            self.chan3.append([])
            self.chan12.append([])
            self.chan23.append([])
            self.chan13.append([])

        self.colorspace = colorspace
        if colorspace == 'ycrbr':
            self.chan1_name = 'Y'
            self.chan2_name = 'Cr'
            self.chan3_name = 'Br'
            print('YCrBr analysis')
        elif colorspace == 'bgr':
            self.chan1_name = 'B'
            self.chan2_name = 'G'
            self.chan3_name = 'R'

        self.frame_num = 0

        self.outpath = outpath
        self.input_vid_name = 'mp_' + input_video_path.split('/')[-1][:-4]
    
    
    def run(self):
        """
        process the input video!
        """

        #generate data
        with tqdm(total=self.tot_input_vid_frames) as pbar:
            pbar.set_description('Generating color channel visualization data')
            while self.input_capture.isOpened():
                ret, frame = self.input_capture.read()
                if ret:
                    self.frame_num += 1
                    if self.chan1_name == 'Y':
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
                    for i in range(self.num_pixels):
                        r = int(i / self.W)
                        c = i % self.W
                        self.chan1[i].append(frame[r, c, 0])
                        self.chan2[i].append(frame[r, c, 1])
                        self.chan3[i].append(frame[r, c, 2])
                        self.chan12[i].append(frame[r, c, 0] / frame[r, c, 1])
                        self.chan13[i].append(frame[r, c, 0] / frame[r, c, 2])
                        self.chan23[i].append(frame[r, c, 1] / frame[r, c, 2])
                else:
                    break
                pbar.update(1)
        for i in range(900):
            print(self.chan1[i][10])
        #save all data
        with open(f'{self.outpath}/{self.colorspace}_{self.input_vid_name}_pixelwise_channel_data.pkl','wb') as f:
            data_dict = {
                'chan1':self.chan1, 
                'chan2':self.chan2, 
                'chan3':self.chan3, 
                'chan12':self.chan12,
                'chan13':self.chan13,
                'chan23':self.chan23,
            }
            pickle.dump(data_dict, f)

       