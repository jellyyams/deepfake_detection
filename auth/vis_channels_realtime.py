import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class channel_visualizer():
    def __init__(self, out_video_path, win_size=60,  plot_W = 640, plot_H = 480, fps=30):
        self.win_size = win_size
        self.plot_W = plot_W
        self.plot_H =  plot_H
           
        self.r = []
        self.g = []
        self.b = []

        self.frame_num = 0

        self.out_vid = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (self.plot_W, self.plot_H * 3))
                
    def add_frame_data(self, frame):
        """
        process the input video!
        """
        self.frame_num += 1
        b_data = np.mean(frame[:, :, 0])
        g_data = np.mean(frame[:, :, 1])
        r_data = np.mean(frame[:, :, 2])
        self.b.append(b_data)
        self.g.append(g_data)
        self.r.append(r_data)
        b_plot = self.plot_dists('b')
        g_plot = self.plot_dists('g')
        r_plot = self.plot_dists('r')
        combined_plot = np.vstack((b_plot, g_plot, r_plot))
        self.out_vid.write(combined_plot)
        return b_data, g_data, r_data
    
    def plot_dists(self, channel):
        """
        generate frame showing trend of channel values
        """
        if channel == 'b':
            plt.plot(self.b, color='b')
        elif channel == 'g':
            plt.plot(self.g, color='g')
        else:
            plt.plot(self.r, color='r')

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
