import cv2
import numpy as np
import matplotlib.pyplot as plt
import itertools
from color_utils import colors_list_rgb
import scipy
import scipy.stats as stats
from modwt import modwt, modwtmra 

def loadVideo(video_path, colorspace='ycrcb', downsamples=0, crop_coords=None):
    """
    From Hussem Ben Belgacem's Eulerian Video Magnification implementation: 
    https://github.com/hbenbel/Eulerian-Video-Magnification/
    """
    if crop_coords is not None and downsamples > 0:
        print("CAREFUL, you are both cropping and downsampling. Do you really want to do this?")

    image_sequence = []
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)

    while video.isOpened():
        ret, frame = video.read()
        if ret is False:
            break
        if downsamples > 0:
            for i in range(downsamples):
                frame = cv2.pyrDown(frame)

        if crop_coords is not None:
            left = crop_coords[0]
            right = crop_coords[1]
            top = crop_coords[2]
            bottom = crop_coords[3]
            frame = frame[top:bottom+1, left:right+1]
        
        if colorspace == 'yuv':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        elif colorspace == 'ycrcb':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
  
        image_sequence.append(frame[:, :, :])

    video.release()

    return np.asarray(image_sequence), fps

def detect_pilot_cells(heatmap):
    """
    Detect all possible pilot cells in heatmap
    Also return the brightest pilot cell (i.e., 'most trustworthy' indicator of blinking)
    """
    #threshold 
    blur = cv2.GaussianBlur(heatmap,(5,5),0)
    otsu_ret, _ = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, th = cv2.threshold(blur,otsu_ret,255,cv2.THRESH_BINARY)
    cv2.imshow('Thresh results', th)
    cv2.waitKey(0)

    # kernel = np.ones((5, 5), np.uint8) #must use odd kernel to avoid shifting
    # th = cv2.dilate(th, kernel, iterations=3)
    # th = cv2.erode(th, kernel, iterations=3)

    # cv2.imshow('Dilate + Eroded', th)
    # cv2.waitKey(0)

    #detect squares/recetangles
    contours,hierarchy = cv2.findContours(th, 1, 2)
    contour_area_zscrore_thresh = 4

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
    #     vis_img = cv2.putText(vis_img, "%.1f" % contour_mean_brightness, (x1, y1),cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), 1, cv2.LINE_AA)
    # cv2.imshow("Contour brightnesses", vis_img)
    # cv2.waitKey(0)

    filtered_contour_centers = list(contour_centers[np.where(zs<contour_area_zscrore_thresh)[0]])
    filtered_contour_bboxes = list(contour_bboxes[np.where(zs<contour_area_zscrore_thresh)[0]])
    
    vis_img = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)
    for bbox in filtered_contour_bboxes:
        cv2.rectangle(vis_img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 0, 255), 1)
    cv2.imshow("Filtered countour bboxes", vis_img)
    cv2.waitKey(0)


    return filtered_contour_centers, brightest_contour, filtered_contour_bboxes


def order_points_closest(points):
    N = len(points)
    sorted_points = [points[0]]
    prev_p = points[0]
    del points[0]
    for i in range(N-1):
        curr_p = points[0]
        points.sort(key = lambda p: (p[0] - prev_p[0])**2 + (p[1] - prev_p[1])**2)
        closest_p = points[0]
        del points[0]
        sorted_points.append(closest_p)
        prev_p = curr_p
    return sorted_points

# print(order_points_closest([[0, 0], [1, 1], [2, 2], [0,0.5], [1.5, 2], [5, 5]]))

def find_barcode_rows_simple(contour_centers, heatmap, slope_epsilon=0.05, y_epsilon=5):
    pairs = list(itertools.combinations(contour_centers, 2))
    
    slopes = {}
    slope_updater = {}
    for pair in pairs:
        pair = [pair[0], pair[1]]
        if pair[0][0] - pair[1][0] == 0:
            slope = float('inf')
        else:
            slope = (pair[0][1] - pair[1][1]) / (pair[0][0] - pair[1][0])
        added = False
        for key in slopes.keys():
            updated_slope = np.mean(np.array(slope_updater[key]))
            if np.abs(slope - updated_slope) < slope_epsilon:
                slopes[key].append(pair)
                added = True
                #update slope in slope updater
                slope_updater[key].append(slope)
                break
        if not added:
            slopes[slope] = [pair]
            slope_updater[slope] = [slope]

    #reassign slopes keys to be the average of all pairwise slopes in that dict entry
    slopes_temp = {}
    for key, value in slopes.items():
        updated_slope = np.mean(np.array(slope_updater[key]))
        slopes_temp[updated_slope] = value
    slopes = slopes_temp

    sorted(slopes, key=lambda k: len(slopes[k]), reverse=True)

    most_popular_slope = list(slopes.keys())[0]
    print("Estimated slope of rows is {}".format(most_popular_slope))
    
    theta = np.degrees(np.arcsin(most_popular_slope))
    M = cv2.getRotationMatrix2D((int(heatmap.shape[1]/2), int(heatmap.shape[0]/2)), theta, 1)
    M_inv =  cv2.getRotationMatrix2D((int(heatmap.shape[1]/2), int(heatmap.shape[0]/2)), -theta, 1)

    vis_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)
    vis_heatmap_rot = cv2.warpAffine(src=vis_heatmap, M=M, dsize=(heatmap.shape[1], heatmap.shape[0]))
    rot_cs = []
    for c in contour_centers:
        c = list(c)
        c.append(1)
        cprime = np.array(c).T
        c_rot =  M@cprime
        c_rot = c_rot[:2].astype(int)
        cv2.circle(vis_heatmap_rot, c_rot, 1, (0, 0, 255), -1)
        rot_cs.append(c_rot.tolist())
    
    cv2.imshow("Temp rot", vis_heatmap_rot)
    cv2.waitKey(0)

    rot_rows = []
    rows = []
    for cnum, rot_c in enumerate(rot_cs):
        added = False
        for i in range(len(rows)):
            rot_row = rot_rows[i]
       
            curr_row_y_avg = np.mean(np.array(rot_row)[:,1])
            if np.abs(rot_c[1]-curr_row_y_avg) < y_epsilon:
                rows[i].append(contour_centers[cnum])
                rot_rows[i].append(rot_c)
                added=True
                break
        if not added:
            rows.append([contour_centers[cnum]])
            rot_rows.append([rot_c])
    
    for l_num, line in enumerate(rows):
        if len(line) == 1:
            p1 = (line[0][0]-10, int(line[0][1] -  10*most_popular_slope))
            p2 = (line[0][0]+10, int(line[0][1] + 10*most_popular_slope))
            cv2.line(vis_heatmap, p1, p2, colors_list_rgb[l_num], 2)
            continue
        for i in range(0, len(line)):
            if i + 1 >= len(line):
                continue
            cv2.line(vis_heatmap, line[i], line[i+1], colors_list_rgb[l_num], 2)
    
    cv2.imshow('Detected barcode rows', vis_heatmap)
    cv2.waitKey(0)

    return most_popular_slope, rows

def find_barcode_rows(contour_centers, heatmap, epsilon=0.05):
    """
    Assuming the pilot cells form a rectangular grid with width larger than height,
    the rows of the barcode (under potential perspective projection) can be found 
    by finding the minimum number of parallel rows that contain all contour centers
    (with some tolerance epsilon to account for the fact that the center detections are
    somewhat imperfect)s

    This function achieves this task, using a greedy algorithm that finds the most popular
    slope between all pairs of centers.

    Contour centers should be a list of lists, where each sublist is a coordinate
    """
    print("Finding barcode rows with epilson=", epsilon)
    pairs = list(itertools.combinations(contour_centers, 2))
    
    slopes = {}
    slope_updater = {}
    for pair in pairs:
        pair = [pair[0], pair[1]]
        if pair[0][0] - pair[1][0] == 0:
            slope = float('inf')
        else:
            slope = (pair[0][1] - pair[1][1]) / (pair[0][0] - pair[1][0])
        added = False
        for key in slopes.keys():
            updated_slope = np.mean(np.array(slope_updater[key]))
            if np.abs(slope - updated_slope) < epsilon:
                slopes[key].append(pair)
                added = True
                #update slope in slope updater
                slope_updater[key].append(slope)
                break
        if not added:
            slopes[slope] = [pair]
            slope_updater[slope] = [slope]

    #reassign slopes keys to be the average of all pairwise slopes in that dict entry
    slopes_temp = {}
    for key, value in slopes.items():
        updated_slope = np.mean(np.array(slope_updater[key]))
        slopes_temp[updated_slope] = value
    slopes = slopes_temp

    sorted(slopes, key=lambda k: len(slopes[k]), reverse=True)

    most_popular_slope = list(slopes.keys())[0]
    most_popular_slope_pairs = list(slopes[most_popular_slope])
    most_popular_slope_points = set() #use set to guarantee uniqueness
    for pair in most_popular_slope_pairs:
        most_popular_slope_points.add(tuple(pair[0])) #have to use tuple here so that it can go into set
        most_popular_slope_points.add(tuple(pair[1]))
    most_popular_slope_points = list(most_popular_slope_points)
    print("Estimated slope of rows is {}".format(most_popular_slope))

    vis_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)
    
    # arbitrarilry choose firt member of most_popular_slope_points to seed
    # and start creating separate parallel rows from it
    bfirst_contour_centers = order_points_closest(contour_centers)
    rows = [[bfirst_contour_centers[0]]]
    neighbor_row = 0
    #NEED TO HAVE THIS BREADTH-FIRST-ESQUE SORTING 
    for pnum in range(1, len(bfirst_contour_centers)):
        #see if this point belongs with its nearest neighbor. If not, find it a home in another row
        neighbor = bfirst_contour_centers[pnum - 1]
        p = bfirst_contour_centers[pnum]
        if neighbor[0] - p[0] == 0:
            slope = float('inf')
        else:
            slope = (neighbor[1] - p[1]) / (neighbor[0] - p[0])

        if np.abs(slope - most_popular_slope) < epsilon:
            cv2.line(vis_heatmap, neighbor, p, (0, 255, 0), 2)
            cv2.circle(vis_heatmap, neighbor, 2, (255, 255, 255), -1)
            cv2.circle(vis_heatmap, p, 2, (255, 255, 255), -1)
            cv2.imshow('temp', vis_heatmap)
            cv2.waitKey(0)
            rows[neighbor_row].append(p)
        else:
            added = False
            for i, line in enumerate(rows):
                p0 = line[0]
                if p0[0] - p[0] == 0:
                    slope = float('inf')
                else:
                    slope = (p0[1] - p[1]) / (p0[0] - p[0])
                if np.abs(slope - most_popular_slope) < epsilon:
                    rows[i].append(p)
                    added = True
                    neighbor_row = i
                    cv2.line(vis_heatmap, p0, p, (0, 255, 0), 2)
                    cv2.imshow('temp', vis_heatmap)
                    cv2.waitKey(0)
                    break
            if not added:
                cv2.circle(vis_heatmap, p, 2, (0, 0, 255), -1)
                cv2.imshow('temp', vis_heatmap)
                cv2.waitKey(0)
                rows.append([p])
                neighbor_row = len(rows) - 1
            
     
    for l_num, line in enumerate(rows):
        if len(line) == 1:
            p1 = (line[0][0]-10, int(line[0][1] -  10*most_popular_slope))
            p2 = (line[0][0]+10, int(line[0][1] + 10*most_popular_slope))
            cv2.line(vis_heatmap, p1, p2, colors_list_rgb[l_num], 2)
            continue
        for i in range(0, len(line)):
            if i + 1 >= len(line):
                continue
            cv2.line(vis_heatmap, line[i], line[i+1], colors_list_rgb[l_num], 2)
    
    cv2.imshow('Detected barcode rows', vis_heatmap)
    cv2.waitKey(0)

    return most_popular_slope, rows


def estimate_pilot_indices_and_corners(contour_centers, heatmap, slope_epsilon=0.05):
    """"
    
    
    """ 
    slope, rows= find_barcode_rows_simple(contour_centers, heatmap, slope_epsilon=slope_epsilon, y_epsilon=5)

    theta = np.degrees(np.arcsin(slope))
    M = cv2.getRotationMatrix2D((int(heatmap.shape[1]/2), int(heatmap.shape[0]/2)), theta, 1)
    M_inv =  cv2.getRotationMatrix2D((int(heatmap.shape[1]/2), int(heatmap.shape[0]/2)), -theta, 1)
    
    vis_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)
    vis_heatmap_rot = cv2.warpAffine(src=vis_heatmap, M=M, dsize=(heatmap.shape[1], heatmap.shape[0]))
    rot_rows = []
    for line in rows:
        rot_line = []
        for c in line:
            c = list(c)
            c.append(1)
            cprime = np.array(c).T
            c_rot =  M@cprime
            c_rot = c_rot[:2].astype(int)
            cv2.circle(vis_heatmap_rot, c_rot, 1, (0, 0, 255), -1)
            rot_line.append(c_rot.tolist())
        rot_rows.append(rot_line)
    
    cv2.imshow("Rows made horizontal", vis_heatmap_rot)
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
    
    sorted_cols = find_barcode_columns(rot_rows, vis_heatmap_rot)

    #use fact that rows, rot_rows, and sorted_cols have the same ordering of the contour centers
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
      

def find_barcode_columns(rows, heatmap, epsilon=5):
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
                curr_col_x_avg = np.mean(np.array(col)[:,0])
                if np.abs(p[0]-curr_col_x_avg) < epsilon:
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

    for l_num, line in enumerate(sorted_cols):
        for i in range(0, len(line)):
            if i + 1 == len(line):
                continue
            cv2.line(heatmap, line[i], line[i+1], colors_list_rgb[-l_num], 2)
   
    cv2.imshow('Detected barcode cols', heatmap)
    cv2.waitKey(0)

    return sorted_cols


def get_homography(sorted_contour_centers, reference_centers, heatmap, reference_img):
    sorted_contour_centers = np.array(sorted_contour_centers)
    reference_centers = np.array(reference_centers)
    H, status = cv2.findHomography(sorted_contour_centers, reference_centers)

  
    # Warp source image to destination based on homography to visualize success
    vis_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)
    img_out = cv2.warpPerspective(vis_heatmap, H, (640, 360))
    cv2.imshow("Src image", vis_heatmap)
    cv2.imshow("Dst image", reference_img)
    cv2.imshow("Result", img_out)
    cv2.waitKey(0)
    return H

# def get_homography_all_correspondences(encoding_params, contour_centers, heatmap, slope_epsilon=0.05):
    
#     sorted_pilot_indices, sorted_contour_centers = estimate_pilot_indices_and_corners(contour_centers, heatmap, slope_epsilon=slope_epsilon)

#     #generate reference image with only pilot cells (in white) from encoding params
#     reference_img, reference_centers = generate_pilot_reference_centers(encoding_params, sorted_pilot_indices)

  
#     # contour_centers = contour_centers.tolist()
#     # contour_centers.sort(key = lambda x: x[0])
    
#     # sorted_contour_bboxes = sort_contour_bboxes(contour_bboxes)
#     # sorted_contour_centers = []
#     # for i, bbox in enumerate(sorted_contour_bboxes):
#     #     pt = [int(bbox[0]+bbox[2]/2), int(bbox[1]+bbox[2]/2)]
#     #     sorted_contour_centers.append(pt)
#     #     cv2.circle(vis_heatmap, pt, 2, (0, 0, 255), -1)
#     #     cv2.putText(vis_heatmap, str(i), pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
#     # # cv2.imshow("Reference centers", vis_reference_img)
#     # cv2.imshow("Heatmap pilot centers", vis_heatmap)
#     # cv2.waitKey(0)
   
#     sorted_contour_centers = np.array(sorted_contour_centers)
#     reference_centers = np.array(reference_centers)
#     H, status = cv2.findHomography(sorted_contour_centers, reference_centers)

  
#     # Warp source image to destination based on homography to visualize success
#     vis_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)
#     for i in range(sorted_contour_centers.shape[0]):
#         cv2.putText(vis_heatmap, str(i), sorted_contour_centers[i], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0, 255), 1, cv2.LINE_AA)
#         #cv2.putText(reference_img, str(i), reference_centers[i]-4, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 1, cv2.LINE_AA)
#     img_out = cv2.warpPerspective(vis_heatmap, H, (640, 360))
#     cv2.imshow("Src image", vis_heatmap)
#     cv2.imshow("Dst image", reference_img)
#     cv2.imshow("Result", img_out)
#     cv2.waitKey(0)
#     return H

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


# def get_homography_from_corners(contour_centers, encoding_params, heatmap, reference_img):
#     upper_left_vid, upper_right_vid, lower_left_vid, lower_right_vid = estimate_corner_pilots(contour_centers)

#     vis_img = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)
#     vis_img = cv2.circle(vis_img, upper_left_vid, 3, (0, 255, 0), -1)
#     vis_img = cv2.circle(vis_img, upper_right_vid, 3, (0, 255, 0), -1)
#     vis_img = cv2.circle(vis_img, lower_left_vid, 3, (0, 255, 0), -1)
#     vis_img = cv2.circle(vis_img, lower_right_vid, 3, (0, 255, 0), -1)
#     # cv2.imshow("Detected corners", vis_img)
#     # cv2.waitKey(0)

#     W = 640 #DLP resolution
#     H = 360

#     #unpack encoding params
#     N = encoding_params['N']
#     buffer_space = encoding_params['buffer_space']
#     block_dim = encoding_params['block_dim']
#     corner_markers = encoding_params['corner_markers']
#     max_blocks_H = encoding_params['max_blocks_H']
#     max_blocks_W = encoding_params['max_blocks_W']

#     if max_blocks_H == None:
#         max_blocks_H = int((H - 2*N + buffer_space) / (N*block_dim + buffer_space))
#     if max_blocks_W == None:
#         max_blocks_W = int((W - 2*N + buffer_space) / (N*block_dim + buffer_space))

#     upper_left_ref = get_block_center(0, corner_markers, N, block_dim, buffer_space, max_blocks_W)
#     upper_right_ref = get_block_center(max_blocks_W-1, corner_markers, N, block_dim, buffer_space, max_blocks_W)
#     lower_left_ref = get_block_center(((max_blocks_H-1)*max_blocks_W), corner_markers, N, block_dim, buffer_space, max_blocks_W)
#     lower_right_ref = get_block_center((max_blocks_H*max_blocks_W)-1, corner_markers, N, block_dim, buffer_space, max_blocks_W)

#     canvas = np.zeros((H, W)).astype(np.float32)
#     canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
#     canvas = cv2.circle(canvas, upper_left_ref.astype(int), 2, (0, 0, 255), -1)
#     canvas = cv2.circle(canvas, upper_right_ref.astype(int), 2, (0, 0, 255), -1)
#     canvas = cv2.circle(canvas, lower_left_ref.astype(int), 2, (0, 0, 255), -1)
#     canvas = cv2.circle(canvas, lower_right_ref.astype(int), 2, (0, 0, 255), -1)
#     # cv2.imshow('Reference corners', canvas)
#     # cv2.waitKey(0)

#     vid_corners = np.array([upper_left_vid, upper_right_vid, lower_left_vid, lower_right_vid])
#     ref_corners = np.array([upper_left_ref, upper_right_ref, lower_left_ref, lower_right_ref])
#     print(ref_corners)
#     H, status = cv2.findHomography(vid_corners, ref_corners)

#     for c in ref_corners.astype(int):
#         print(c)
#         cv2.circle(reference_img, c, 2, (0, 0, 255), -1)
#     # Warp source image to destination based on homography to visualize success
#     img_out = cv2.warpPerspective(vis_img, H, (640, 360))
#     cv2.imshow("Src image", vis_img)
#     cv2.imshow("Dst image", reference_img)
#     cv2.imshow("Result", img_out)
#     cv2.waitKey(0)

#     return H

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
    pilot_signal = np.array(pilot_signal)
    peaks, _ = scipy.signal.find_peaks(pilot_signal)
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

    #concatenate min to start and end to allow for peak detection at start/stop of signal
    denoised_concat = np.concatenate(([min(denoised)],denoised,[min(denoised)]))
    denoised_concat_peaks, _  = scipy.signal.find_peaks(denoised_concat, distance=peak_dist, prominence=0.5) #consider width, dist, plateau size params
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
    ax2.plot(denoised_peaks, denoised[denoised_peaks], "x")
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
    return denoised_peaks, edge_pairs


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
