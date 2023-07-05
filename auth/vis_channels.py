import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
from tqdm import tqdm

def idealTemporalBandpassFilter(data, fps, freq_range):
    # print('data len ', len(data))
    # print('data ', data)
    fft = np.fft.fft(data)
    # print('fft len ', len(fft))
    # print('fft ', fft)
    frequencies = np.fft.fftfreq(len(data), d=1.0/fps)
    low = (np.abs(frequencies - freq_range[0])).argmin()
    high = (np.abs(frequencies - freq_range[1])).argmin()

    fft[:low] = 0
    fft[high:] = 0
    filtered = np.fft.ifft(data).real
    # print('result len ', len(filtered))
    # print('result ', filtered)
    return filtered

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

        self.filtered_chan12 = []
        self.filtered_chan13 = []
        self.filtered_chan23 = []

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

        # print('---- Setting up output video ----')
        self.outpath = outpath
        self.input_vid_name = 'mp_' + input_video_path.split('/')[-1][:-4]
        # self.out_vid = cv2.VideoWriter(f'{outpath}/{self.input_vid_name}_channel_vis.mp4', cv2.VideoWriter_fourcc(*'mp4v'), self.input_cap_fps, (self.plot_W, self.plot_H * 3))
        # self.ratios_out_vid = cv2.VideoWriter(f'{outpath}/{self.input_vid_name}_channel_ratio_vis.mp4', cv2.VideoWriter_fourcc(*'mp4v'), self.input_cap_fps, (self.plot_W, self.plot_H * 3))
        # self.filtered_ratios_out_vid = cv2.VideoWriter(f'{outpath}/{self.input_vid_name}_filtered_channel_ratio_vis.mp4', cv2.VideoWriter_fourcc(*'mp4v'), self.input_cap_fps, (self.plot_W, self.plot_H * 3))
        # print('---- Done setting up output video ----')
        
        
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
                    self.chan1.append(np.mean(frame[:, :, 0]))
                    self.chan2.append(np.mean(frame[:, :, 1]))
                    self.chan3.append(np.mean(frame[:, :, 2]))
                    self.chan12.append(np.mean(frame[:, :, 0]) / np.mean(frame[:, :, 1]))
                    self.chan13.append(np.mean(frame[:, :, 0]) / np.mean(frame[:, :, 2]))
                    self.chan23.append(np.mean(frame[:, :, 1]) / np.mean(frame[:, :, 2]))
                else:
                    break
                pbar.update(1)
        
        #filter data
        self.filtered_chan12 = idealTemporalBandpassFilter(self.chan12, self.input_cap_fps, (0, 1))
        self.filtered_chan13 = idealTemporalBandpassFilter(self.chan13, self.input_cap_fps, (0, 1))
        self.filtered_chan23 = idealTemporalBandpassFilter(self.chan23, self.input_cap_fps, (0, 1))

        #save all data
        with open(f'{self.outpath}/{self.colorspace}_{self.input_vid_name}_channel_data.pkl','wb') as f:
            data_dict = {
                'chan1':self.chan1, 
                'chan2':self.chan2, 
                'chan3':self.chan3, 
                'chan12':self.chan12,
                'chan13':self.chan13,
                'chan23':self.chan23,
                'filtered_chan12':self.filtered_chan12,
                'filtered_chan13':self.filtered_chan13,
                'filtered_chan23':self.filtered_chan23
            }
            pickle.dump(data_dict, f)

        # #plot raw data
        # with tqdm(total=self.tot_input_vid_frames) as pbar:
        #     pbar.set_description('Generating color channel visualization videos')
        #     for i in range(self.tot_input_vid_frames):
        #         chan1_plot = self.plot_dists('chan1', i)
        #         chan2_plot = self.plot_dists('chan2', i)
        #         chan3_plot = self.plot_dists('chan3', i)
        #         combined_plot = np.vstack((chan1_plot, chan2_plot, chan3_plot))
        #         self.out_vid.write(combined_plot)

        #         chan12_plot = self.plot_ratios('chan1:2', i)
        #         chan13_plot = self.plot_ratios('chan1:3', i)
        #         chan23_plot = self.plot_ratios('chan2:3', i)
        #         ratios_combined_plot = np.vstack((chan12_plot, chan13_plot, chan23_plot))
        #         self.ratios_out_vid.write(ratios_combined_plot)

        #         filtered_chan12_plot = self.plot_ratios('chan1:2', i, filtered=True)
        #         filtered_chan13_plot = self.plot_ratios('chan1:3', i, filtered=True)
        #         filtered_chan23_plot = self.plot_ratios('chan2:3', i, filtered=True)
        #         filtered_ratios_combined_plot = np.vstack((filtered_chan12_plot, filtered_chan13_plot, filtered_chan23_plot))
        #         self.filtered_ratios_out_vid.write(filtered_ratios_combined_plot)
                
        #         pbar.update(1)
        
      
       
        self.input_capture.release()
        # self.ratios_out_vid.release()
        # self.filtered_ratios_out_vid.release()
        #self.out_vid.release()
    
    def plot_ratios(self, channel, frame_num, filtered=False):
        if filtered:
            plot1_data = self.filtered_chan12
            chan12_ylim = (min(self.filtered_chan12), max(self.filtered_chan12))
        else:
            plot1_data = self.chan12
            chan12_ylim = (min(self.chan12), max(self.chan12))
        if filtered:
            plot2_data = self.filtered_chan13
            chan13_ylim = (min(self.filtered_chan13), max(self.filtered_chan13))
        else:
            plot2_data = self.chan13
            chan13_ylim = (min(self.chan13), max(self.chan13))
        if filtered:
            plot3_data = self.filtered_chan23
            chan23_ylim = (min(self.filtered_chan23), max(self.filtered_chan23))
        else:
            plot3_data = self.chan23
            chan23_ylim = (min(self.chan23), max(self.chan23))
        """
        generate frame showing trend of channel value ratios
        """
        if channel == 'chan1:2':
            plt.ylim(chan12_ylim)
            if filtered:
                plt.title(f'Filtered {self.chan1_name}:{self.chan2_name}')
            else:
                plt.title(f'{self.chan1_name}:{self.chan2_name}')
            plt.plot(plot1_data[0:frame_num])
        elif channel == 'chan1:3':
            plt.ylim(chan13_ylim)
            if filtered:
                plt.title(f'Filtered {self.chan1_name}:{self.chan3_name}')
            else:
                plt.title(f'{self.chan1_name}:{self.chan3_name}')
            plt.plot(plot2_data[0:frame_num])
        else:
            plt.ylim(chan23_ylim)
            if filtered:
                plt.title(f'Filtered {self.chan2_name}:{self.chan3_name}')
            else:
                plt.title(f'{self.chan2_name}:{self.chan3_name}')
            plt.plot(plot3_data[0:frame_num])

        if frame_num < self.win_size:
            plt.xlim(0, self.win_size)
        else:
            plt.xlim(frame_num - int(self.win_size/2), frame_num + int(self.win_size/2))

        figure = plt.gcf()
        figure.canvas.draw()
        fig_img = np.array(figure.canvas.buffer_rgba())
        fig_img = cv2.cvtColor(fig_img, cv2.COLOR_RGBA2BGR)
        plt.clf()
        return fig_img


    def plot_dists(self, channel, frame_num):
        """
        generate frame showing trend of channel values
        """
        chan1_ylim = (min(self.chan1) - 10, max(self.chan1) + 10)
        chan2_ylim = (min(self.chan2) - 10, max(self.chan2) + 10)
        chan3_ylim = (min(self.chan3) - 10, max(self.chan3) + 10)

        if channel == 'chan1':
            plt.ylim(chan1_ylim)
            plt.title(self.chan1_name)
            plt.plot(self.chan1[0:frame_num])
        elif channel == 'chan2':
            plt.ylim(chan2_ylim)
            plt.title(self.chan2_name)
            plt.plot(self.chan2[0:frame_num])
        else:
            plt.ylim(chan3_ylim)
            plt.title(self.chan3_name)
            plt.plot(self.chan3[0:frame_num])

        if frame_num < self.win_size:
            plt.xlim(0, self.win_size)
        else:
            plt.xlim(frame_num - int(self.win_size/2), frame_num + int(self.win_size/2))

        figure = plt.gcf()
        figure.canvas.draw()
        fig_img = np.array(figure.canvas.buffer_rgba())
        fig_img = cv2.cvtColor(fig_img, cv2.COLOR_RGBA2BGR)
        plt.clf()
        return fig_img
