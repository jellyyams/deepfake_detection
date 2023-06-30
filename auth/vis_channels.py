import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class channel_visualizer():
    def __init__(self, input_video_path, outpath, colorspace = 'yiq', win_size=60,  plot_W = 640, plot_H = 480):
        self.win_size = win_size
        self.plot_W = plot_W
        self.plot_H =  plot_H
       
        try:
            self.input_capture = cv2.VideoCapture(input_video_path)
        except:
            raise ValueError('Input video path %s not valid' % input_video_path)
       
        if self.input_capture is not None:  
            self.tot_input_vid_frames = int(self.input_capture.get(cv2.CAP_PROP_FRAME_COUNT))  
            input_cap_fps = int(self.input_capture.get(cv2.CAP_PROP_FPS))
            self.W , self.H = int(self.input_capture.get(3)), int(self.input_capture.get(4)) #input video dimensions
        else:
            raise ValueError("Invalid input video")
        
        self.chan1 = []
        self.chan2 = []
        self.chan3 = []

        if colorspace == 'yiq':
            self.chan1_name = 'Y'
            self.chan2_name = 'I'
            self.chan3_name = 'Q'
            print('YIQ analysis')
        elif colorspace == 'bgr':
            self.chan1_name = 'B'
            self.chan2_name = 'G'
            self.chan3_name = 'R'

        self.frame_num = 0

        print('---- Setting up output video ----')
        input_vid_name = 'mp_' + input_video_path.split('/')[-1][:-4]
        self.out_vid = cv2.VideoWriter(f'{outpath}{input_vid_name}_channel_vis.avi', cv2.VideoWriter_fourcc(*'mp4v'), input_cap_fps, (self.plot_W, self.plot_H * 3))
        print('---- Done setting up output video ----')
        
        
    def run(self):
        """
        process the input video!
        """
        with tqdm(total=self.tot_input_vid_frames) as pbar:
            pbar.set_description('Generating color channel visualizations')
            while self.input_capture.isOpened():
                ret, frame = self.input_capture.read()
                if ret:
                    self.frame_num += 1
                    if self.chan1_name == 'Y':
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
                    self.chan1.append(np.mean(frame[:, :, 0]))
                    self.chan2.append(np.mean(frame[:, :, 1]))
                    self.chan3.append(np.mean(frame[:, :, 2]))
                    chan1_plot = self.plot_dists('chan1')
                    chan2_plot = self.plot_dists('chan2')
                    chan3_plot = self.plot_dists('chan3')
                    combined_plot = np.vstack((chan1_plot, chan2_plot, chan3_plot))
                    self.out_vid.write(combined_plot)

                else:
                    break
                pbar.update(1)
            self.input_capture.release()
            self.out_vid.release()
    

    def plot_dists(self, channel):
        """
        generate frame showing trend of channel values
        """
        if channel == 'chan1':
            plt.title(self.chan1_name)
            plt.plot(self.chan1)
        elif channel == 'chan2':
            plt.title(self.chan2_name)
            plt.plot(self.chan2)
        else:
            plt.title(self.chan3_name)
            plt.plot(self.chan3)

        if self.frame_num < self.win_size:
            plt.xlim(0, self.win_size)
        else:
            plt.xlim(self.frame_num - int(self.win_size/2), self.frame_num + int(self.win_size/2))

        plt.ylim(0, 255)
        plt.title(channel)
        figure = plt.gcf()
        figure.canvas.draw()
        fig_img = np.array(figure.canvas.buffer_rgba())
        fig_img = cv2.cvtColor(fig_img, cv2.COLOR_RGBA2BGR)
        plt.clf()
        return fig_img
