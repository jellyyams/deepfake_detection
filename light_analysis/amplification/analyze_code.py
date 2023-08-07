import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy
from flash_gt import gen_blink_list,gen_blink_list_from_log,  find_led_blink_start
from datetime import datetime
from code_analysis_utils import fftVideo

def pixelwise_analysis(input_vid_path):
    fftVideo(input_vid_path, gaussian_pyr=False, level=2)


def segment_analysis(input_vid_path):
    
    input_capture = cv2.VideoCapture(input_vid_path)
    W, H= input_capture.get(3), input_capture.get(4)

    #code parameters
    SLM_N = 30 #size of one cell in SLM pixels
    block_dim = 3 #dimension of one block, in num cells
    max_blocks_W = 5
    max_blocks_H = 5
    SLM_buffer_space = 30 #num pixels (SLM) between blocks
    SLM_W = 640
    SLM_H = 360

    buffer_space_horiz = SLM_buffer_space * (W/SLM_W)
    buffer_space_vert = SLM_buffer_space * (H/SLM_H)
    N_horiz = SLM_N * (W/SLM_W)
    N_vert = SLM_N * (H/SLM_H)

    vid_start_dt = datetime.strptime('2023-07-24 16:04:51.19', '%Y-%m-%d %H:%M:%S.%f')
    print(vid_start_dt)
    vid_start_millisec = vid_start_dt.timestamp() * 1000
    led_log_path = 'aug4_input_logs/r4_g0_b0_1000mson_1000msoff_rep15.txt'
    blinks = gen_blink_list_from_log(vid_start_dt, led_log_path)

    cap = cv2.VideoCapture(input_vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    yuv_chan1 = []
    yuv_chan2 = []
    yuv_chan3 = []

    while input_capture.isOpened():
        ret, frame = input_capture.read()
        if ret:
            #yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            #add values of each cell to chan value lists
            for r in range(0, max_blocks_H):
                for c in range(0, max_blocks_W):
                    row_start = int(((r + 1) * buffer_space_vert) + (r*N_vert))
                    row_end = int(row_start + N_vert)
                    col_start = int(((c + 1) * buffer_space_horiz) + (c*N_horiz))
                    col_end = int(col_start + N_horiz)
                    print(row_start, row_end, col_start, col_end)
                    cell = frame[row_start:row_end, col_start:col_end, :]
                    cv2.imshow('testwin', cell)
                    cv2.waitKey(0)
                        
                
                    # yuv_chan1_mean = np.mean(yuv_frame[:,:, 0])
                    # yuv_chan2_mean = np.mean(yuv_frame[:,:, 1])
                    # yuv_chan3_mean = np.mean(yuv_frame[:,:, 2])
                    # yuv_chan1.append(yuv_chan1_mean)
                    # yuv_chan2.append(yuv_chan2_mean)
                    # yuv_chan3.append(yuv_chan3_mean)
                
        else:
            break
        
        cv2.destroyAllWindows() # close all windows


pixelwise_analysis('aug6_cropped_videos/r60_g0_b0_1000mson_1000msoff_N30_buff30_rep15_croppedrgba.avi')