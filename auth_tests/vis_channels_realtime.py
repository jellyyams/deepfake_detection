import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class channel_visualizer():
    def __init__(self, out_video_path, colorspace = 'ycrbr', win_size=60,  plot_W = 640, plot_H = 480, fps=30):
        self.win_size = win_size
        self.plot_W = plot_W
        self.plot_H =  plot_H
           
        self.chan1 = []
        self.chan2 = []
        self.chan3 = []

        if colorspace == 'ycrbr':
            self.chan1_name = 'Y'
            self.chan2_name = 'Cr'
            self.chan3_name = 'Br'
            print('YIQ analysis')
        elif colorspace == 'bgr':
            self.chan1_name = 'B'
            self.chan2_name = 'G'
            self.chan3_name = 'R'

        self.frame_num = 0

        self.out_vid = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (self.plot_W, self.plot_H * 3))
                
    def add_frame_data(self, frame):
        """
        process the input video!
        """
        self.frame_num += 1
        if self.chan1_name == 'Y':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
        chan1_data = np.mean(frame[:, :, 0])
        chan2_data = np.mean(frame[:, :, 1])
        chan3_data = np.mean(frame[:, :, 2])
        self.chan1.append(chan1_data)
        self.chan2.append(chan2_data)
        self.chan3.append(chan3_data)
        chan1_plot = self.plot_dists('chan1')
        chan2_plot = self.plot_dists('chan2')
        chan3_plot = self.plot_dists('chan3')
        combined_plot = np.vstack((chan1_plot, chan2_plot, chan3_plot))
        self.out_vid.write(combined_plot)
        return chan1_data, chan2_data, chan3_data #is this being used anywhere
    
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
