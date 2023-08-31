import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy
from barcode_utils import loadVideo, detect_pilot_cells, estimate_pilot_indices_and_corners, get_homography, get_pilot_blink_times
import scipy.stats as stats
from sklearn.preprocessing import minmax_scale
import pandas as pd
from generate_heatmap import generate_heatmap
import sys


def get_cell_center(i, max_border_cells_W, N, buffer_space, offset):
    cell_row = int(i / max_border_cells_W)
    cell_col = int(i % max_border_cells_W)

    cell_top = offset + cell_row*(N+buffer_space) 
    cell_left = offset + cell_col*(N+buffer_space) 

    cell_center = [cell_left+int(N/2), cell_top+int(N/2)]

    return cell_center

def generate_reference_points(W, H, N, buffer_space, corner_markers, max_border_cells_W, sorted_center_indices):
   
    if corner_markers:
        offset = N 
    else:
        offset = 0
     
    reference_img = np.zeros((H, W)).astype(np.float32)
    reference_img = cv2.cvtColor(reference_img, cv2.COLOR_GRAY2BGR)
    reference_centers = []

    for i in sorted_center_indices:   
        cell_center = get_cell_center(i, max_border_cells_W, N, buffer_space, offset)

        reference_centers.append(cell_center)

        cell_top = cell_center[1] - int(N/2)
        cell_bottom = cell_center[1] + int(N/2)
        cell_left = cell_center[0] - int(N/2)
        cell_right = cell_center[0] + int(N/2)

        reference_img[cell_top:cell_bottom+1, cell_left:cell_right+1] = 255

        cv2.circle(reference_img, cell_center, 2, (0, 0, 255), -1) 
        cv2.putText(reference_img, str(i), cell_center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow('Reference cell image', reference_img)
    cv2.waitKey(0)

    return reference_img, reference_centers

def get_cell_signal(N, buffer_space, corner_markers, image_sequence, H, cell_row, cell_col, target_channel):
    if corner_markers:
        offset = N 
    else:
        offset = 0
    signal = []
    cell_top = offset + cell_row*(N+buffer_space) 
    cell_left = offset + cell_col*(N+buffer_space) 
    cell_bottom = cell_top + N
    cell_right = cell_left + N
    mask = np.zeros((360, 640), np.uint8)
    cv2.rectangle(mask, (cell_left, cell_top), (cell_right, cell_bottom), 255, -1)
    for i in range(image_sequence.shape[0]):
        img = image_sequence[i, :,:, :]
        img = cv2.warpPerspective(img, H, (640, 360))
        mean_cell_brightness = cv2.mean(img[:,:,target_channel], mask=mask)[0]
        signal.append(mean_cell_brightness)
    return signal

def get_mag_at_freq(signal, target_freq):
    signal = np.array(signal)
    signal -= np.mean(signal)
    signal = minmax_scale(signal)
    ps =  np.abs(np.fft.fft(signal))**2
    freqs = np.fft.fftfreq(len(signal), 1/fps)
    target_freq_index = (np.abs(freqs - target_freq)).argmin()
    magnitude = np.abs(ps[target_freq_index])
    return magnitude

def get_rise_edges(signal, on, off, fps):
    signal = np.array(signal)
    tail = np.array([min(signal) for i in range(int(on * fps))])
    concat_signal = np.concatenate((tail,signal))
    concat_signal -= np.average(concat_signal)

    neg_ones = np.array([-1 for i in range(int(off*fps))])
    ones = np.array([1 for i in range(int(on*fps))])
    filter = np.hstack((neg_ones, ones))
    
    filtered_concat_signal = np.convolve(concat_signal, filter, mode='valid') 
    peaks, _ = scipy.signal.find_peaks(filtered_concat_signal) 
    
    # plt.plot(concat_signal)
    # plt.plot(filtered_concat_signal)
    # plt.show()

    peaks -= int(on*fps)

    ymin = np.min(signal)
    ymax = np.max(signal)
    # plt.plot(signal)
    # plt.vlines(peaks, ymin, ymax, colors='grey', linestyles='dashed')
    # plt.show()

    return peaks


def get_phase_diff(border_signal, info_signal, border_peaks, info_peaks, fps):
    border_signal = np.array(border_signal)
    info_signal = np.array(info_signal)

    if len(border_peaks) != len(info_peaks):
        b_peaks = np.array(border_peaks)
        i_peaks = np.array(info_peaks)

        indexes = abs(b_peaks[:, None] - i_peaks).argmin(axis=1)
        matches = list(enumerate(indexes))
        matched_border_peaks = []
        matched_info_peaks = []
        for m in matches:
            matched_border_peaks.append(border_peaks[m[0]])
            matched_info_peaks.append(info_peaks[m[1]])
    else:
        matched_border_peaks = border_peaks
        matched_info_peaks = info_peaks

    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    # ax1.plot(border_signal)
    # ax1.plot(matched_border_peaks, border_signal[matched_border_peaks], "x")
    # ax1.set_title("Border")

    # ax2.plot(info_signal)
    # ax2.plot(matched_info_peaks, info_signal[matched_info_peaks], "x")
    # ax2.set_title('Info')
    # ax2.set_xlabel('Frames]')
    # plt.show()

    phase_diff_estimates = []
    for i in range(len(matched_info_peaks) - 1):
        border_peak_diff = (matched_border_peaks[i + 1] - matched_border_peaks[i]) 
        info_peak_diff = (matched_info_peaks[i + 1] - matched_border_peaks[i]) 
        # print("B: ", border_peak_diff)
        # print("I: ", info_peak_diff)
        # print("hi ", info_peak_diff / border_peak_diff)
        # print()
        phase_diff = ((info_peak_diff / border_peak_diff) * 360) % 360
        phase_diff_estimates.append(phase_diff)
    
    # print("Phase diff estimates: ", phase_diff_estimates)

    mean_phase_diff = np.mean(np.array(phase_diff_estimates))
    return mean_phase_diff


input_vid_path = 'test_vids/aug27_input_videos/r140_g0_b0_N15_b15_s2_05Hz.MP4'
target_channel = 2
target_freq = 0.5
on_time = off_time = 1/(target_freq * 2)
colorspace = 'bgr'

W = 640 #DLP resolution
H = 360

corner_markers =  False
N = buffer_space = border_width = border_buffer_space = 15
if corner_markers:
    offset = N
else:
    offset = 0


if corner_markers:
    max_info_cells_W = int((W - 2*N - 2*border_width - border_buffer_space) / (N + buffer_space))
    max_info_cells_H = int((H - 2*N - 2*border_width - border_buffer_space) / (N + buffer_space))

    max_border_cells_W = int((W -2*N) / (border_width + border_buffer_space))
    max_border_cells_H = int((H -2*N + border_buffer_space) / (border_width + border_buffer_space))

else:
    max_info_cells_W = int((W - 2*border_width - border_buffer_space) / (N + buffer_space))
    max_info_cells_H = int((H - 2*border_width - border_buffer_space) / (N + buffer_space))

    max_border_cells_W = int((W + border_buffer_space) / (border_width + border_buffer_space))
    max_border_cells_H = int((H + border_buffer_space)/ (border_width + border_buffer_space))
    

# heatmap_chan1, heatmap_chan2, heatmap_chan3, crop_coords = generate_heatmap(input_vid_path, .45, .55, target_channel=target_channel, padding=30, colorspace=colorspace)

heatmap_chan1 = None
heatmap_chan2 = None
heatmap_chan3 = cv2.imread('aug28_heatmap_temp/heatmap_chan3_r140_g0_b0_N15_b15_s2_05Hz_bgr.png', cv2.IMREAD_GRAYSCALE)
crop_coords =  [110, 538, 442, 712]
#crop_coords = [114, 544, 446, 702]

if target_channel == 0:
    heatmap = heatmap_chan1
elif target_channel == 1:
    heatmap = heatmap_chan2
elif target_channel == 2:
    heatmap = heatmap_chan3
else:
    print("Invalid target channel")
    sys.exit(0)

contour_centers, brightest_contour, contour_bboxes  = detect_pilot_cells(heatmap)

sorted_center_indices, sorted_contour_centers = estimate_pilot_indices_and_corners(contour_centers, heatmap, slope_epsilon=0.07)

#generate reference image with only pilot cells (in white) from encoding params
reference_img, reference_centers = generate_reference_points(W, H, N, buffer_space, corner_markers, max_border_cells_W, sorted_center_indices)

Hom = get_homography(sorted_contour_centers, reference_centers, heatmap, reference_img)

######### DECODE ##########
image_sequence, fps = loadVideo(input_vid_path, colorspace=colorspace, crop_coords=crop_coords)

# border_signals = []
# print("Plotting all border signals")
# fig, axes = plt.subplots(nrows=max_border_cells_H , ncols=max_border_cells_W, figsize=(12, 8))
# for r in range(max_border_cells_H):
#     for c in range(max_border_cells_W):
#         border_signal = get_cell_signal(N, buffer_space, corner_markers, image_sequence, Hom, r, c, target_channel)
#         axes[r, c].plot(border_signal)
#         border_signals.append(border_signal)
# plt.show()


# main_border_signal = border_signals[0] #for now
# np.save('temp_border.npy', np.array(main_border_signal))

main_border_signal = list(np.load('temp_border.npy'))
plt.plot(main_border_signal)
plt.show()

border_mag = get_mag_at_freq(main_border_signal, target_freq)

border_rise_edges = get_rise_edges(main_border_signal, on_time, off_time, fps)
inferred_img = np.zeros((H, W)).astype(np.float32)
inferred_img = cv2.cvtColor(inferred_img, cv2.COLOR_GRAY2BGR)
output_bitstring = ''
for r in range(max_info_cells_H):
    for c in range(max_info_cells_W):
        info_signal = get_cell_signal( N, buffer_space, corner_markers, image_sequence, Hom, r+1, c+1, target_channel)
        info_mag = get_mag_at_freq(info_signal, target_freq)
        confidence = np.clip(info_mag / border_mag, 0, 1) * 100
        info_rise_edges = get_rise_edges(info_signal, on_time, off_time, fps)
        phase_diff = get_phase_diff(main_border_signal, info_signal, border_rise_edges, info_rise_edges, fps)

        if phase_diff >= 90 and phase_diff <= 270: #closer to 1 than 0
            cell_value = 1
        else:
            cell_value = 0

        cell_top = offset + (r+1)*(N+buffer_space) 
        cell_left = offset + (c+1)*(N+buffer_space) 
        cell_bottom = cell_top + N
        cell_right = cell_left + N
        if cell_value == 1:
            cv2.rectangle(inferred_img, (cell_left, cell_top), (cell_right, cell_bottom), (0, 0, 255), -1)
        else:
            cv2.rectangle(inferred_img, (cell_left, cell_top), (cell_right, cell_bottom), (0, 0, 255), 1)
        cv2.putText(inferred_img, str(cell_value), (cell_left+int(N/2)-4, cell_top+int(N/2)+2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(inferred_img,  "%.1f" % confidence, (cell_left+int(N/2)-11, cell_top+int(N/2)+9), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
        output_bitstring += str(cell_value)

cv2.imshow("Inferred code", inferred_img)
cv2.imwrite("code.png", inferred_img)
cv2.waitKey(0)

print(output_bitstring)