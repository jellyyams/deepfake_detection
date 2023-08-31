import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy
from flash_gt import gen_blink_list, gen_blink_list_from_log,  find_led_blink_start
from barcode_utils import find_barcode_rows, detect_pilot_cells, estimate_pilot_indices_and_corners, denoise_pilot_signal, get_homography
from datetime import datetime
import scipy.stats as stats
import pandas as pd
import itertools
import collections 

## NEED TO UPDATE CONTOUR CENTERS TO ALWAYS BE LIST OF LIST (NOT LIST OF NUMPY ARRS)

def loadVideo(video_path, colorspace='ycrcb'):
    """
    From Hussem Ben Belgacem's Eulerian Video Magnification implementation: 
    https://github.com/hbenbel/Eulerian-Video-Magnification/
    """
    image_sequence = []
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)

    while video.isOpened():
        ret, frame = video.read()
        if ret is False:
            break
        if colorspace == 'yuv':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        elif colorspace == 'ycrcb':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    
        image_sequence.append(frame[:, :, :])

    video.release()

    return np.asarray(image_sequence), fps



def nearest_on_off_windows(on_edges, frame_num):
    for i, edge_pair in enumerate(on_edges):
        if frame_num > edge_pair[0] and frame_num > edge_pair[1]:
            if frame_num < on_edges[i + 1][0]: #frame is during the  off period after current edge pair's right edge
                if np.abs(frame_num - on_edges[i + 1][0]) < np.abs(frame_num - edge_pair[1]): #the next on period is closer
                    return on_edges[i+1], [edge_pair[1]+1,on_edges[i+1][0]-1]
                else: #previous on period is closer
                    return edge_pair, [edge_pair[1]+1,on_edges[i+1][0]-1]
            else: #move on to next on edge pair
                continue
        elif frame_num >= edge_pair[0] and frame_num <= edge_pair[1]:
            #frame is during this on period
            if np.abs(frame_num - (edge_pair[1] + 1)) < np.abs(frame_num - (edge_pair[0] - 1)) or i == 0: #the next off period is closer
                # if i = 0, there is no previous off period, must use next of period
                return edge_pair, [edge_pair[1] + 1, on_edges[i+1][0] - 1]
            else: #the previous off period is closer
                return edge_pair, [on_edges[i-1][1] + 1, edge_pair[0] - 1]     
        else:
            continue

def get_mean_cell_value(image_sequence, H, frame_num, cell_left, cell_right, cell_top, cell_bottom, analysis_channel):
    if type(frame_num) == int:
        #get mean of cell value just at this frame
        img = image_sequence[frame_num, :,:,:]
        img = cv2.warpPerspective(img, H, (640, 360))
        cell = img[cell_top:cell_bottom, cell_left:cell_right, :]
        #vis_img = cv2.cvtColor(img, cv2.COLOR_YCR_CB2BGR)
        # vis_img[cell_top:cell_bottom, cell_left:cell_right, :] = 0
        # cv2.imshow('test', vis_img)
        # cv2.waitKey(0)
        return np.mean(cell[:,:,analysis_channel])
    else:
        #get mean of cell value within frame number range
        vals = []
        for i in range(frame_num[0], frame_num[1]):
            img = image_sequence[i, :, :, :]
            img = cv2.warpPerspective(img, H, (640, 360))
            cell = img[cell_top:cell_bottom, cell_left:cell_right, :]
            vals.append(np.mean(cell[:,:,analysis_channel]))
        vals = np.array(vals)
        return np.mean(vals)


def get_block_center(block_num, corner_markers, N, block_dim, buffer_space, max_blocks_W):
    block_row = int(block_num / max_blocks_W)
    block_col = int(block_num % max_blocks_W)

    block_pix_row_start = int(block_row * N * block_dim)
    block_pix_row_end = int((block_row + 1) * N * block_dim)
    block_pix_col_start = int(block_col * N * block_dim)
    block_pix_col_end = int((block_col + 1) * N * block_dim)

    if corner_markers:
        block_pix_row_start += N + buffer_space * block_row
        block_pix_row_end += N + buffer_space * block_row 
        block_pix_col_start += N + buffer_space * block_col
        block_pix_col_end += N + buffer_space * block_col
    else:
        block_pix_row_start += buffer_space * block_row
        block_pix_row_end += buffer_space * block_row 
        block_pix_col_start += buffer_space * block_col
        block_pix_col_end += buffer_space * block_col 

    block_center_coords = np.array([block_pix_col_start + ((N*block_dim)/2), block_pix_row_start + ((N*block_dim)/2)])
    return block_center_coords


def generate_pilot_reference_centers(encoding_params, sorted_pilot_indices):
    W = 640 #DLP resolution
    H = 360

    #unpack encoding params
    N = encoding_params['N']
    buffer_space = encoding_params['buffer_space']
    block_dim = encoding_params['block_dim']
    corner_markers = encoding_params['corner_markers']
    max_blocks_H = encoding_params['max_blocks_H']
    max_blocks_W = encoding_params['max_blocks_W']

    if max_blocks_H == None:
        max_blocks_H = int((H - 2*N + buffer_space) / (N*block_dim + buffer_space))
    if max_blocks_W == None:
        max_blocks_W = int((W - 2*N + buffer_space) / (N*block_dim + buffer_space))

    
    reference_img = np.zeros((H, W)).astype(np.float32)
    reference_img = cv2.cvtColor(reference_img, cv2.COLOR_GRAY2BGR)
    reference_centers = []
    for i in sorted_pilot_indices:
        block_center = get_block_center(i, corner_markers, N, block_dim, buffer_space, max_blocks_W).astype(int).tolist()
        reference_centers.append(block_center)
   
        pilot_left = block_center[0] - int(N/2)
        pilot_right = block_center[0] + int(N/2)
        pilot_top = block_center[1] - int(N/2)
        pilot_bottom = block_center[1] + int(N/2)
    
        reference_img[pilot_top:pilot_bottom+1, pilot_left:pilot_right+1] = 255

        cv2.circle(reference_img, block_center, 2, (0, 0, 255), -1) 
        cv2.putText(reference_img, str(i), block_center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow('Reference pilot image', reference_img)
    cv2.waitKey(0)

    return reference_img, reference_centers




def decode_cells(encoding_params, on_edges, image_sequence, Hom, frame_num, analysis_channel, top_pilot_cell_signal, epsilon = 3):
    img = image_sequence[frame_num, :, :,:]
    img = cv2.warpPerspective(img, Hom, (640, 360))
    
    W = 640 #DLP resolution
    H = 360

    #unpack encoding params
    N = encoding_params['N']
    buffer_space = encoding_params['buffer_space']
    block_dim = encoding_params['block_dim']
    corner_markers = encoding_params['corner_markers']
    max_blocks_H = encoding_params['max_blocks_H']
    max_blocks_W = encoding_params['max_blocks_W']

    if max_blocks_H == None:
        max_blocks_H = int((H - 2*N + buffer_space) / (N*block_dim + buffer_space))
    if max_blocks_W == None:
        max_blocks_W = int((W - 2*N + buffer_space) / (N*block_dim + buffer_space))

    pilot_index = int(block_dim/2)
    
    nearest_on, nearest_off = nearest_on_off_windows(on_edges, frame_num)
    plt.plot(top_pilot_cell_signal)
    ymin = np.min(top_pilot_cell_signal)
    ymax = np.max(top_pilot_cell_signal)
    plt.vlines(nearest_on, ymin, ymax, colors='r', linestyles='dashed')
    plt.vlines(nearest_off, ymin, ymax, colors='b', linestyles='dashed')
    plt.vlines([frame_num], ymin, ymax, colors='g')
    plt.title("Nearest on/off periods")
    plt.show()

    vis_img = cv2.cvtColor(img, cv2.COLOR_YCR_CB2BGR)
    vis_img2 = vis_img.copy()
    for i in range(max_blocks_H*max_blocks_W):
        block_center = get_block_center(i, corner_markers, N, block_dim, buffer_space, max_blocks_W)
        block_left = block_center[0] - ((N*block_dim)/2)
        block_top = block_center[1] - ((N*block_dim)/2)

        #handle pilot cell
        cell_left = int(block_left + pilot_index*N)
        cell_right = int(block_left + (pilot_index+1)*N)
        cell_top = int(block_top + (pilot_index*N))
        cell_bottom = int(block_top+ (pilot_index+1)*N)
        cv2.rectangle(vis_img, (cell_left, cell_top), (cell_right, cell_bottom), (255, 0, 0), 2)
        cv2.rectangle(vis_img2, (cell_left, cell_top), (cell_right, cell_bottom), (255, 0, 0),2)
        pilot_cell_mean_on_value = get_mean_cell_value(image_sequence, Hom, nearest_on, cell_left, cell_right, cell_top, cell_bottom, analysis_channel)
        pilot_cell_mean_off_value = get_mean_cell_value(image_sequence, Hom, nearest_off, cell_left, cell_right, cell_top, cell_bottom, analysis_channel)
        cv2.putText(vis_img, "%.1f" % pilot_cell_mean_on_value, (cell_left+int(N/2)-10, cell_top+int(N/2)-2), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(vis_img, "%.1f" % pilot_cell_mean_off_value, (cell_left+int(N/2)-10, cell_top+int(N/2)+6), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), 1, cv2.LINE_AA)
        
        #segment info cells
        for j in range(block_dim):
            for k in range(block_dim):
                if j == pilot_index and k == pilot_index:
                    continue
                cell_left = int(block_left + k*N)
                cell_right = int(block_left + (k+1)*N)
                cell_top = int(block_top + (j*N))
                cell_bottom = int(block_top+ (j+1)*N)
                cv2.rectangle(vis_img, (cell_left, cell_top), (cell_right, cell_bottom), (0, 0, 255), 1)
                cv2.rectangle(vis_img2, (cell_left, cell_top), (cell_right, cell_bottom), (0, 0, 255), 1)
                info_cell_mean_value = get_mean_cell_value(image_sequence, Hom, frame_num, cell_left, cell_right, cell_top, cell_bottom, analysis_channel)
                if np.abs(info_cell_mean_value - pilot_cell_mean_on_value) < np.abs(info_cell_mean_value - pilot_cell_mean_off_value):
                    cell_value = 1
                else:
                    cell_value = 0
                # print(cell_value)

                #simplistic instantaneous comparison
                # if pilot_on:
                #     if info_cell_mean_value < pilot_cell_mean_value - epsilon:
                #         cell_value = 0
                #     else:
                #         cell_value = 1
                # else:
                #     if info_cell_mean_value >= pilot_cell_mean_value:
                #         cell_value = 1
                #     else:
                #         cell_value = 0
             
                cv2.putText(vis_img, "%.1f" % info_cell_mean_value, (cell_left+int(N/2)-10, cell_top+int(N/2)+2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(vis_img2, str(cell_value), (cell_left+int(N/2)-4, cell_top+int(N/2)+2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
   
    cv2.imshow('Cell channel vals', vis_img)
    cv2.imshow('Inferred bit', vis_img2)
    cv2.waitKey(0)
    cv2.imwrite('cellmeans.png', vis_img)
    # cv2.imwrite('cellvals.png', vis_img2)
    
    return

def decode_test(input_vid_path):
    #create heatmap
    #for now, just use a saved heatmap for testing
    heatmap = cv2.imread('boundary_tests/heatmap_chan2_r60_g0_b0_1000mson_1000msoff_N15_buff15_rep15_croppedrgba_ycrcb.png', cv2.IMREAD_GRAYSCALE) #for now
    #heatmap = cv2.imread('boundary_tests/rot_test.jpg')
    #heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
    contour_centers, brightest_contour, contour_bboxes  = detect_pilot_cells(heatmap)

     #assumed simple encoding parameters, for now
    encoding_params = {
        'corner_markers' : True,
        'N':15,
        'buffer_space':15,
        'block_dim':3,
        'max_blocks_W':None,
        'max_blocks_H':None
    }

    sorted_pilot_indices, sorted_contour_centers = estimate_pilot_indices_and_corners(contour_centers, heatmap, slope_epsilon=slope_epsilon)

    #generate reference image with only pilot cells (in white) from encoding params
    reference_img, reference_centers = generate_pilot_reference_centers(encoding_params, sorted_pilot_indices)

    H = get_homography(sorted_contour_centers, reference_centers, heatmap, reference_img)


    # vis_img =  cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)
    # cv2.drawContours(vis_img, [brightest_contour], -1, (0,255,0), 1)
    # cv2.imshow("Brightest pilot cell", vis_img)
    # cv2.waitKey(0)

   
    # reference_img_path = 'test_vids/aug6_codes/on_frame_r60_g0_b0_N30_buff30.bmp' #for visualization purposes
    # reference_img = cv2.imread(reference_img_path)
    # reference_img = cv2.flip(reference_img, 0) #add when testing on codes that are reflected correctly consistently
    # #reference_img = cv2.flip(reference_img, 1) #add when testing on codes that are reflected correctly consistently
    # H = get_homography_from_corners(contour_centers, encoding_params, heatmap, reference_img)

    
    ########## DECODE ##########
    # image_sequence, fps = loadVideo(input_vid_path)
    # top_pilot_cell_signal = []
    # analysis_channel = 1
    # for i in range(image_sequence.shape[0]):
    #     img = image_sequence[i, :,:, :]
    #     mask = np.zeros(img.shape[:2], np.uint8)
    #     cv2.drawContours(mask, brightest_contour, -1, 255, -1)
    #     mean_cell_brightness = cv2.mean(img[:,:,analysis_channel], mask=mask)[0]
    #     top_pilot_cell_signal.append(mean_cell_brightness)
    #     vis_img = cv2.cvtColor(img, cv2.COLOR_YCR_CB2BGR)

    # top_pilot_cell_signal = np.array(top_pilot_cell_signal)

    # plt.plot(top_pilot_cell_signal)
    # plt.show()
    
    # _, on_edges = get_pilot_blink_times(top_pilot_cell_signal, fps, 1, 1)

    # # for each frame in image sequence, determine values of each cell
    # # by comparing to reference value of pilot, which can be
    # # inferred using the blink frame nums
    # for i in range(image_sequence.shape[0]):
    #     if i < 10 or i >= 11:
    #         continue
    #     img = image_sequence[i, :,:, :]
    #     #apply homography
    #     img = cv2.warpPerspective(img, H, (640, 360))
    #     decode_cells(encoding_params, on_edges, image_sequence, H, i, analysis_channel, top_pilot_cell_signal, )
        



input_vid_path = 'test_vids/aug6_cropped_videos/r60_g0_b0_1000mson_1000msoff_N30_buff30_rep15_croppedrgba.avi'
decode_test(input_vid_path)