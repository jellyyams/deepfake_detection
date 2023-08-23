import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy
from flash_gt import gen_blink_list, gen_blink_list_from_log,  find_led_blink_start
from datetime import datetime
import scipy.stats as stats
from boundary_tests.modwt import modwt, modwtmra 
import pandas as pd
import itertools
import collections 


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


def detect_pilot_cells(heatmap):
    """
    Detect all possible pilot cells in heatmap
    Also return the brightest pilot cell (i.e., 'most trustworthy' indicator of blinking)
    """
    #threshold 
    blur = cv2.GaussianBlur(heatmap,(5,5),0)
    otsu_ret, th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # cv2.imshow('Thresh results', th)
    # cv2.waitKey(0)

    kernel = np.ones((5, 5), np.uint8) #must use odd kernel to avoid shifting
    th = cv2.dilate(th, kernel, iterations=3)
    th = cv2.erode(th, kernel, iterations=3)

    cv2.imshow('Dilate + Eroded', th)
    cv2.waitKey(0)

    #detect squares/recetangles
    contours,hierarchy = cv2.findContours(th, 1, 2)
    contour_area_zscrore_thresh = 1

    contour_centers = []
    contour_areas = []
    contour_bboxes = []
    vis_img = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)
    for cnt in contours:
        x1,y1 = cnt[0][0]
        approx__vertices = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        x, y, w, h = cv2.boundingRect(cnt)
        contour_bboxes.append([x, y, w, h])
        contour_center = (int(x+(w/2)), int(y+(h/2)))
        contour_area = w*h
        contour_areas.append(contour_area)
        contour_centers.append(contour_center)

        vis_img = cv2.drawContours(vis_img, [cnt], -1, (0,255,0), 1)
        vis_img = cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 0, 255), 1)
        vis_img = cv2.circle(vis_img, contour_center, 2 , (0, 0, 255), -1)
        # vis_img = cv2.putText(vis_img, "%.1f" % contour_mean_brightness, (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow("Unfiltered countour detections", vis_img)
    cv2.waitKey(0)
   
    contour_centers = np.array(contour_centers)
    contour_areas = np.array(contour_areas)
    contour_bboxes = np.array(contour_bboxes)
    zs = np.abs(stats.zscore(contour_areas))

    # fig, ax = plt.subplots(figsize =(10, 7))
    # ax.hist(contour_areas)
    # plt.title("Contour Areas Distribution")
    # plt.show()

    #find brightest pilot cell, excluding contours that are very small in area
    contour_min_area = np.quantile(contour_areas, .25)
    max_contour_brightness = float('-inf')
    brightest_contour = None
    for i, cnt in enumerate(contours):
        x1,y1 = cnt[0][0]
        if contour_areas[i] < contour_min_area:
            continue
        mask = np.zeros(heatmap.shape, np.uint8)
        cv2.drawContours(mask, cnt, -1, 255, -1)
        contour_mean_brightness = cv2.mean(heatmap, mask=mask)[0]
        if contour_mean_brightness > max_contour_brightness:
            max_contour_brightness = contour_mean_brightness
            brightest_contour = cnt
        #vis_img = cv2.putText(vis_img, "%.1f" % contour_mean_brightness, (x1, y1),cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), 1, cv2.LINE_AA)
    # cv2.imshow("Contour brightnesses", vis_img)
    # cv2.waitKey(0)

    filtered_contour_centers = contour_centers[np.where(zs<contour_area_zscrore_thresh)[0]]
    filtered_contour_bboxes = contour_bboxes[np.where(zs<contour_area_zscrore_thresh)[0]]
    
    vis_img = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)
    for bbox in filtered_contour_bboxes:
        cv2.rectangle(vis_img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 0, 255), 1)
    cv2.imshow("Filtered countour bboxes", vis_img)
    cv2.waitKey(0)


    return filtered_contour_centers, brightest_contour, filtered_contour_bboxes

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

def shi_tomasi(img, num_corners, corner_qual, min_dist):
    """
    From https://blog.ekbana.com/skew-correction-using-corner-detectors-and-homography-fda345e42e65
   
    Use Shi-Tomasi algorithm to detect corners
    Args:
        image: np.array
        num_corners: int - numbers to corners to detect
        corner_qual: float - quality of corners (a value between 0–1, below which all possible corners are rejected) 
        min_dist: int -minimum euclidean distance between two corners.
    Returns:
        corners: list
    """
    corners = cv2.goodFeaturesToTrack(img, num_corners, corner_qual, min_dist)
    corners = np.int0(corners)
    corners = sorted(np.concatenate(corners).tolist())
    print('\nThe corner points are...\n')

    im = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for index, c in enumerate(corners):
        x, y = c
        cv2.circle(im, (x, y), 3, 255, -1)

    plt.imshow(im)
    plt.title('Corner Detection: Shi-Tomasi')
    plt.show()
    return corners

def sort_contour_bboxes(contour_bboxes):
    """
    Slightly modified from version at 
    how-can-i-sort-contours-from-left-to-right-and-top-to-bottom/38693156#38693156
    """
    bboxes=sorted(contour_bboxes, key=lambda x: x[1])
    df=pd.DataFrame(bboxes, columns=['x','y','w', 'h'], dtype=int)
    df["y2"] = df["y"]+df["h"] # adding column for x on the right side
    df = df.sort_values(["y", "x", "y2"]) # sorting

    for i in range(2): # change rows between each other by their coordinates several times 
    # to sort them completely 
        for ind in range(len(df)-1):
            if df.iloc[ind][4] > df.iloc[ind+1][1] and df.iloc[ind][0]> df.iloc[ind+1][0]:
                df.iloc[ind], df.iloc[ind+1] = df.iloc[ind+1].copy(), df.iloc[ind].copy()
    
    sorted_contour_bboxes = df.values.tolist()
    return sorted_contour_bboxes



def estimate_pilot_indices_and_corners(contour_centers, heatmap):
    """"
    
    
    """ 
    slope, rows= find_barcode_rows(contour_centers, heatmap, epsilon=.1)

    theta = np.degrees(np.arcsin(slope))
    M = cv2.getRotationMatrix2D((int(heatmap.shape[1]/2), int(heatmap.shape[0]/2)), theta, 1)
    M_inv =  cv2.getRotationMatrix2D((int(heatmap.shape[1]/2), int(heatmap.shape[0]/2)), -theta, 1)
    
    vis_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)
    vis_heatmap = cv2.warpAffine(src=vis_heatmap, M=M, dsize=(heatmap.shape[1], heatmap.shape[0]))
    rot_rows = []
    for line in rows:
        rot_line = []
        for c in line:
            c = list(c)
            c.append(1)
            cprime = np.array(c).T
            c_rot =  M@cprime
            c_rot = c_rot[:2].astype(int)
            cv2.circle(vis_heatmap, c_rot, 1, (0, 0, 255), -1)
            rot_line.append(c_rot.tolist())
        rot_rows.append(rot_line)
    
    cv2.imshow("Rows made horizontal", vis_heatmap)
    cv2.waitKey(0)
    
    #sort now horizontal rows top to bottom to identify row indices
    avg_ys = []
    for rot_line in rot_rows:
        ys = []
        for c in rot_line:
            ys.append(c[1])
        avg_y = np.mean(np.array(ys))
        avg_ys.append(avg_y)

    line_sorting_indices = np.argsort(np.array(avg_ys))
    sorted_rot_rows = [rot_rows[i] for i in line_sorting_indices]
    
    sorted_cols = find_barcode_columns(rot_rows)

    nums = {}
    for i in range(len(rot_rows)):
        rot_line = rot_rows[i]
        for j in range(len(rot_line)):
            p = rot_line[j]
            for r in range(len(sorted_rot_rows)):
                for c in range(len(sorted_cols)):
                    if p in sorted_rot_rows[r] and p in sorted_cols[c]:
                        nums[r*len(sorted_cols)+c] = rows[i][j]
                        break
  

    vis_heatmap = heatmap.copy()
    vis_heatmap = cv2.cvtColor(vis_heatmap, cv2.COLOR_GRAY2BGR)
    for key, value in nums.items():
        cv2.circle(vis_heatmap, value, 2, (0, 0, 255), -1)
        cv2.putText(vis_heatmap, str(key), value, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    cv2.imshow("Inferred pilot nums", vis_heatmap)
    cv2.waitKey(0)

    pilot_indices = nums.keys()
    sorted_pilot_indices = sorted(pilot_indices)
    sorted_contour_centers = []
    for i in sorted_pilot_indices:
        sorted_contour_centers.append(nums[i])
    
    return sorted_pilot_indices, sorted_contour_centers
      

def find_barcode_columns(rows, epsilon=5):
    """
    Given a set of barcode pilot rows, group all points in those rows into columns, under potential 
    perspective projection 

    ASSUMPTION: There is at least one contour in every column
    """
    cols = []
    for row in rows:
        for p in row:
            added = False
            for i, col in enumerate(cols):
                if np.abs(p[0]-col[0][0]) < epsilon:
                    cols[i].append(p)
                    added=True
                    break
            if not added:
                cols.append([p])

    #sort cols from left to right
    avg_xs = []
    for col in cols:
        xs = []
        for c in col:
            xs.append(c[0])
        avg_x = np.mean(np.array(xs))
        avg_xs.append(avg_x)
    
    col_sorting_indices = np.argsort(np.array(avg_xs))
    sorted_cols = [cols[i] for i in col_sorting_indices]

    return sorted_cols


def find_barcode_rows(contour_centers, heatmap, epsilon=0.05):
    """
    Assuming the pilot cells form a rectangular grid with width larger than height,
    the rows of the barcode (under potential perspective projection) can be found 
    by finding the minimum number of parallel rows that contain all contour centers
    (with some tolerance epsilon to account for the fact that the center detections are
    somewhat imperfect)

    This function achieves this task, using a greedy algorithm that finds the most popular
    slope between all pairs of centers.
    """

    pairs = list(itertools.combinations(contour_centers, 2))
    
    slopes = {}
    for pair in pairs:
        pair = [pair[0].tolist(), pair[1].tolist()]
        if pair[0][0] - pair[1][0] == 0:
            slope = float('inf')
        else:
            slope = (pair[0][1] - pair[1][1]) / (pair[0][0] - pair[1][0])
        added = False
        for key in slopes.keys():
            if np.abs(slope - key) < epsilon:
                slopes[key].append(pair)
                added = True
                break
        if not added:
            slopes[slope] = [pair]
        
    sorted(slopes, key=lambda k: len(slopes[k]), reverse=True)

    most_popular_slope = list(slopes.keys())[0]
    most_popular_slope_pairs = list(slopes[most_popular_slope])
    most_popular_slope_points = set() #use set to guarantee uniqueness
    for pair in most_popular_slope_pairs:
        most_popular_slope_points.add(tuple(pair[0])) #have to use tuple here so that it can go into set
        most_popular_slope_points.add(tuple(pair[1]))
    most_popular_slope_points = list(most_popular_slope_points)

    # arbitrarilry choose firt member of most_popular_slope_points to seed
    # and start creating separate parallel rows from it
    rows = [[contour_centers[0].tolist()]]
    for p in contour_centers[1:]:
        added = False
        for i, line in enumerate(rows):
            p0 = line[0]
            if p0[0] - p[0] == 0:
                slope = float('inf')
            else:
                slope = (p0[1] - p[1]) / (p0[0] - p[0])
            if np.abs(slope - most_popular_slope) < epsilon:
                rows[i].append(p.tolist())
                added = True
                break
        if not added:
            rows.append([p.tolist()])

    vis_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255), (0, 255, 255), (0, 255, 255), (200, 0, 100), (255, 100, 140), (70, 100, 60), (200, 100, 50)]
    for l_num, line in enumerate(rows):
        if len(line) == 1:
            p1 = (line[0][0]-10, int(line[0][1] -  10*most_popular_slope))
            p2 = (line[0][0]+10, int(line[0][1] + 10*most_popular_slope))
            cv2.line(vis_heatmap, p1, p2, colors[l_num], 1)
            continue
        for i in range(0, len(line)):
            if i + 1 >= len(line):
                continue
            cv2.line(vis_heatmap, line[i], line[i+1], colors[l_num], 1)
    cv2.imshow('Detected barcode rows', vis_heatmap)
    cv2.waitKey(0)

    return most_popular_slope, rows


def get_homography_all_correspondences(encoding_params, contour_centers, heatmap):
    
    sorted_pilot_indices, sorted_contour_centers = estimate_pilot_indices_and_corners(contour_centers, heatmap)

    #generate reference image with only pilot cells (in white) from encoding params
    reference_img, reference_centers = generate_pilot_reference_centers(encoding_params, sorted_pilot_indices)

  
    # contour_centers = contour_centers.tolist()
    # contour_centers.sort(key = lambda x: x[0])
    
    # sorted_contour_bboxes = sort_contour_bboxes(contour_bboxes)
    # sorted_contour_centers = []
    # for i, bbox in enumerate(sorted_contour_bboxes):
    #     pt = [int(bbox[0]+bbox[2]/2), int(bbox[1]+bbox[2]/2)]
    #     sorted_contour_centers.append(pt)
    #     cv2.circle(vis_heatmap, pt, 2, (0, 0, 255), -1)
    #     cv2.putText(vis_heatmap, str(i), pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    # # cv2.imshow("Reference centers", vis_reference_img)
    # cv2.imshow("Heatmap pilot centers", vis_heatmap)
    # cv2.waitKey(0)
   
    sorted_contour_centers = np.array(sorted_contour_centers)
    reference_centers = np.array(reference_centers)
    H, status = cv2.findHomography(sorted_contour_centers, reference_centers)

  
    # Warp source image to destination based on homography to visualize success
    vis_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)
    for i in range(sorted_contour_centers.shape[0]):
        cv2.putText(vis_heatmap, str(i), sorted_contour_centers[i], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0, 255), 1, cv2.LINE_AA)
        #cv2.putText(reference_img, str(i), reference_centers[i]-4, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 1, cv2.LINE_AA)
    img_out = cv2.warpPerspective(vis_heatmap, H, (640, 360))
    cv2.imshow("Src image", vis_heatmap)
    cv2.imshow("Dst image", reference_img)
    cv2.imshow("Result", img_out)
    cv2.waitKey(0)
    return H

def estimate_corner_pilots(contour_centers):
    min_x = float('inf')
    max_x = float('-inf')
    min_y = float('inf')
    max_y = float('-inf')

    for c in contour_centers.tolist():
        if c[0] < min_x:
            min_x = c[0]
        elif c[0] > max_x:
            max_x = c[0]
        
        if c[1] < min_y:
            min_y = c[1]
        elif c[1] > max_y:
            max_y = c[1]

    ### simplistic min/max - only works if there is little camera pitch/yaw
    upper_left_vid = [min_x, min_y]
    upper_right_vid = [max_x, min_y]
    lower_left_vid = [min_x, max_y]
    lower_right_vid = [max_x, max_y]
    return upper_left_vid, upper_right_vid, lower_left_vid, lower_right_vid


def get_homography_from_corners(contour_centers, encoding_params, heatmap, reference_img):
    upper_left_vid, upper_right_vid, lower_left_vid, lower_right_vid = estimate_corner_pilots(contour_centers)

    vis_img = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)
    vis_img = cv2.circle(vis_img, upper_left_vid, 3, (0, 255, 0), -1)
    vis_img = cv2.circle(vis_img, upper_right_vid, 3, (0, 255, 0), -1)
    vis_img = cv2.circle(vis_img, lower_left_vid, 3, (0, 255, 0), -1)
    vis_img = cv2.circle(vis_img, lower_right_vid, 3, (0, 255, 0), -1)
    # cv2.imshow("Detected corners", vis_img)
    # cv2.waitKey(0)

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

    upper_left_ref = get_block_center(0, corner_markers, N, block_dim, buffer_space, max_blocks_W)
    upper_right_ref = get_block_center(max_blocks_W-1, corner_markers, N, block_dim, buffer_space, max_blocks_W)
    lower_left_ref = get_block_center(((max_blocks_H-1)*max_blocks_W), corner_markers, N, block_dim, buffer_space, max_blocks_W)
    lower_right_ref = get_block_center((max_blocks_H*max_blocks_W)-1, corner_markers, N, block_dim, buffer_space, max_blocks_W)

    canvas = np.zeros((H, W)).astype(np.float32)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    canvas = cv2.circle(canvas, upper_left_ref.astype(int), 2, (0, 0, 255), -1)
    canvas = cv2.circle(canvas, upper_right_ref.astype(int), 2, (0, 0, 255), -1)
    canvas = cv2.circle(canvas, lower_left_ref.astype(int), 2, (0, 0, 255), -1)
    canvas = cv2.circle(canvas, lower_right_ref.astype(int), 2, (0, 0, 255), -1)
    # cv2.imshow('Reference corners', canvas)
    # cv2.waitKey(0)

    vid_corners = np.array([upper_left_vid, upper_right_vid, lower_left_vid, lower_right_vid])
    ref_corners = np.array([upper_left_ref, upper_right_ref, lower_left_ref, lower_right_ref])
    print(ref_corners)
    H, status = cv2.findHomography(vid_corners, ref_corners)

    for c in ref_corners.astype(int):
        print(c)
        cv2.circle(reference_img, c, 2, (0, 0, 255), -1)
    # Warp source image to destination based on homography to visualize success
    img_out = cv2.warpPerspective(vis_img, H, (640, 360))
    cv2.imshow("Src image", vis_img)
    cv2.imshow("Dst image", reference_img)
    cv2.imshow("Result", img_out)
    cv2.waitKey(0)

    return H

def denoise_pilot_signal(pilot_signal, denoise_method, denoise_options):
    if denoise_method == 'dwt':
        level = denoise_options['level']
        wavelet = denoise_options['wavelet']
        w = modwt(pilot_signal, wavelet, level)
        return w[level]
    elif denoise_method == 'lowpass':
        cutoff_freq = denoise_options['cutoff_frequency']
        order = denoise_options['order']
        fps = denoise_options['fps']
        normed_pilot_signal = pilot_signal - np.mean(pilot_signal)
        sos = scipy.signal.butter(order, cutoff_freq, 'lp', fs=fps, output='sos')
        filtered = scipy.signal.sosfilt(sos, normed_pilot_signal)
        return filtered
    elif denoise_method == 'bandpass':
        low_freq = denoise_options['low_frequency']
        high_freq = denoise_options['high_frequency']
        order = denoise_options['order']
        fps = denoise_options['fps']
        normed_pilot_signal = pilot_signal - np.mean(pilot_signal)
        b, a = scipy.signal.butter(order, [low_freq, high_freq], fs=fps, btype='band')
        filtered = scipy.signal.lfilter(b, a, normed_pilot_signal)
        return filtered


def get_pilot_blink_times(pilot_signal, fps, on_time, off_time):
    """
    Given a signal corresponding to the values of blinking pilot cell in video,
    the fps of the video, and on/off time of blinks, return the approximate frame nums
    of starts and ends of each blink
    """   
    peaks, _ = scipy.signal.find_peaks(pilot_signal) #also consider plateau size
    denoise_method = 'dwt'
    dwt_options = {
        'level':3,
        'wavelet':'haar'
    }
    freq = on_time / (on_time + off_time)
    # bandpass_options = {
    #     'order':3,
    #     'low_frequency': freq - .4,
    #     'high_frequency': freq + .4,
    #     'fps': fps
    # }
    # lowpass_options = {
    #     'order':3,
    #     'cutoff_frequency':1,
    #     'fps':fps
    # }
    denoised = denoise_pilot_signal(pilot_signal, denoise_method, dwt_options)
    peak_dist = (off_time * fps) - 10
    width = [(on_time * fps) - 10, (on_time * fps) + 10]
    #concatenate min to start and end to allow for peak detection at start/stop of signal
    denoised_concat = np.concatenate(([min(denoised)],denoised,[min(denoised)]))
    denoised_concat_peaks, _  = scipy.signal.find_peaks(denoised_concat,distance=peak_dist, prominence=0.5) #consider width, dist, plateau size params
    denoised_peaks = denoised_concat_peaks - 1 #subtract 1 to account for above concatenation
    denoised_concat_peak_widths = scipy.signal.peak_widths(denoised_concat, denoised_concat_peaks)
    denoised_peak_widths = []
    #subtract 1 from edge starts/stops to account for above concatenation
    for i, el in enumerate(denoised_concat_peak_widths):
        if i == 1:
            denoised_peak_widths.append(el)
        else:
            denoised_peak_widths.append((el-1).astype(int)) #convert to int so that edges correspond to frame nums
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(pilot_signal)
    ax1.plot(peaks, pilot_signal[peaks], "x")
    ax1.set_title('Raw signal')
    ax2.plot(denoised)
    ymax = np.max(denoised)
    ymin = np.min(denoised)
    ax2.hlines(*denoised_peak_widths[1:], color="C3")
    ax2.set_title('After denoising with {}'.format(denoise_method))
    ax2.set_xlabel('Frames]')

    edge_pairs = list(zip(denoised_peak_widths[2].tolist(),denoised_peak_widths[3].tolist()))
    for edge_pair in edge_pairs:
        ax2.vlines([edge_pair[0]], ymin, ymax, colors='gray', linestyles='dashed')
        ax2.vlines([edge_pair[1]], ymin, ymax, colors='gray', linestyles='dashed')
    plt.tight_layout()
    plt.show()
    return edge_pairs

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

    
    H = get_homography_all_correspondences(encoding_params, contour_centers, heatmap)
    
    
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
    
    # on_edges = get_pilot_blink_times(top_pilot_cell_signal, fps, 1, 1)

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