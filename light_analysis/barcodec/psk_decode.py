import cv2
import numpy as np

def detect_pilot_cells(heatmap):
    """
    Detect all possible pilot cells in heatmap
    Also return the brightest pilot cell (i.e., 'most trustworthy' indicator of blinking)
    """
    #threshold 
    blur = cv2.GaussianBlur(heatmap,(5,5),0)
    otsu_ret, _ = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, th = cv2.threshold(blur,otsu_ret - 15,255,cv2.THRESH_BINARY)
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
    # zs = np.abs(stats.zscore(contour_areas))

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

    # filtered_contour_centers = contour_centers[np.where(zs<contour_area_zscrore_thresh)[0]]
    # filtered_contour_bboxes = contour_bboxes[np.where(zs<contour_area_zscrore_thresh)[0]]
    
    # vis_img = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)
    # for bbox in filtered_contour_bboxes:
    #     cv2.rectangle(vis_img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 0, 255), 1)
    # cv2.imshow("Filtered countour bboxes", vis_img)
    # cv2.waitKey(0)


    # return filtered_contour_centers, brightest_contour, filtered_contour_bboxes

heatmap = cv2.imread('heatmap_chan2_r60_g0_b0_N30_b30_s2_ycrcb.png', cv2.IMREAD_GRAYSCALE) #for now
contour_centers, brightest_contour, contour_bboxes  = detect_pilot_cells(heatmap)
