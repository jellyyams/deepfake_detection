import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class channel_visualizer():
    def __init__(self, input_video_path, outpath, win_size=60,  plot_W = 640, plot_H = 480):
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
        
        self.r = []
        self.g = []
        self.b = []

        self.frame_num = 0

        print('---- Setting up output video ----')
        input_vid_name = 'mp_' + input_video_path.split('/')[-1][:-4]
        self.out_vid = cv2.VideoWriter(f'{outpath}{input_vid_name}_channel_vis.mp4', cv2.VideoWriter_fourcc(*'MP4V'), input_cap_fps, (self.plot_W, self.plot_H * 3))
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
                    self.b.append(np.mean(frame[:, :, 0]))
                    self.g.append(np.mean(frame[:, :, 1]))
                    self.r.append(np.mean(frame[:, :, 2]))
                    b_plot = self.plot_dists('b')
                    g_plot = self.plot_dists('g')
                    r_plot = self.plot_dists('r')
                    combined_plot = np.vstack((b_plot, g_plot, r_plot))
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
