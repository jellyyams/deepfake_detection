import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm 
from sklearn.preprocessing import minmax_scale
import sys

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

gaussian_kernel = (
    np.array(
        [
            [1,  4,  6,  4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1,  4,  6,  4, 1]
        ]
    )
    / 256
)

def pyrDown(image, kernel):
    return cv2.filter2D(image, -1, kernel)[::2, ::2]

def pyrUp(image, kernel, dst_shape=None):
    dst_height = image.shape[0] + 1
    dst_width = image.shape[1] + 1

    if dst_shape is not None:
        dst_height -= (dst_shape[0] % image.shape[0] != 0)
        dst_width -= (dst_shape[1] % image.shape[1] != 0)

    height_indexes = np.arange(1, dst_height)
    width_indexes = np.arange(1, dst_width)

    upsampled_image = np.insert(image, height_indexes, 0, axis=0)
    upsampled_image = np.insert(upsampled_image, width_indexes, 0, axis=1)

    return cv2.filter2D(upsampled_image, -1, 4 * kernel)


def generateGaussianPyramid(image, level):
    image_shape = [image.shape[:2]]
    downsamplesd_image = image.copy()

    for _ in range(level):
        downsamplesd_image = pyrDown(image=downsamplesd_image, kernel=gaussian_kernel)
        image_shape.append(downsamplesd_image.shape[:2])

    gaussian_pyramid = downsamplesd_image
    for curr_level in range(level):
        gaussian_pyramid = pyrUp(
                            image=gaussian_pyramid,
                            kernel=gaussian_kernel,
                            dst_shape=image_shape[level - curr_level - 1]
                        )

    return gaussian_pyramid


def getGaussianPyramids(images, level):
    gaussian_pyramids = np.zeros_like(images, dtype=np.float32)

    for i in tqdm.tqdm(range(images.shape[0]),
                       ascii=True,
                       desc='Gaussian Pyramids Generation'):

        gaussian_pyramids[i] = generateGaussianPyramid(
                                    image=images[i],
                                    level=level
                        )

    return gaussian_pyramids

def moving_average(a, axis, n=3):
    """
    https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy
    """
    ret = np.cumsum(a, axis=axis, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def fftVideo(images, fps, low_freq, high_freq, colorspace = 'ycrcb', heatmap_norm=False, output_suffix=None, gaussian_pyr_level=0):
    if gaussian_pyr_level > 0:
        images = getGaussianPyramids(
                    images=images,
                    level=gaussian_pyr_level
                )

    axis = 0
    norm_images = images.copy()
    if colorspace == 'gnorm':
        norm_images[:,:,:,0] = norm_images[:,:,:,0] / norm_images[:,:,:,1]
        norm_images[:,:,:,1] = norm_images[:,:,:,1] / norm_images[:,:,:,1]
        norm_images[:,:,:,2] = norm_images[:,:,:,2] / norm_images[:,:,:,1]
    norm_images = np.where(((np.max(norm_images, axis=axis) - np.min(norm_images, axis=axis)) == 0), 0.5, (norm_images - np.min(norm_images, axis=axis)) / (np.max(norm_images, axis=axis) - np.min(norm_images, axis=axis)))
    norm_images = norm_images - np.mean(norm_images, axis=axis)
   
    ps = np.abs(np.fft.fft(norm_images, axis=axis))**2
    freqs = np.fft.fftfreq(images.shape[0], 1/fps)
    idx = np.argsort(freqs)
    target_freqs = [low_freq, high_freq]
    low_freq_index = (np.abs(freqs - target_freqs[0])).argmin()
    high_freq_index = (np.abs(freqs - target_freqs[1])).argmin()
    
    heatmaps = np.sum(ps[low_freq_index:high_freq_index,:,:,:], axis=axis)
    heatmap_chan1 = heatmaps[:,:,0]
    heatmap_chan2 = heatmaps[:,:,1]
    heatmap_chan3 = heatmaps[:,:,2]
    if heatmap_norm:
        heatmap_chan1 = (heatmap_chan1 - np.min(heatmap_chan1))/(np.max(heatmap_chan1) - np.min(heatmap_chan1))
        heatmap_chan2 = (heatmap_chan2 - np.min(heatmap_chan2))/(np.max(heatmap_chan2) - np.min(heatmap_chan2))
        heatmap_chan3 = (heatmap_chan3 - np.min(heatmap_chan3))/(np.max(heatmap_chan3) - np.min(heatmap_chan3))
    
    ax = sns.heatmap(heatmap_chan1, linewidth=0)
    plt.title('Channel 1')
    plt.show()

    ax = sns.heatmap(heatmap_chan2, linewidth=0)
    plt.title('Channel 2')
    plt.show()

    ax = sns.heatmap(heatmap_chan3, linewidth=0) 
    plt.title('Channel 3')
    plt.show()

    png_heatmap_chan1 = minmax_scale(heatmap_chan1.flatten()).reshape(heatmap_chan1.shape) * 255
    png_heatmap_chan1 = png_heatmap_chan1.astype("uint8")
    png_heatmap_chan2 = minmax_scale(heatmap_chan2.flatten()).reshape(heatmap_chan2.shape) * 255
    png_heatmap_chan2 = png_heatmap_chan2.astype("uint8")
    png_heatmap_chan3 = minmax_scale(heatmap_chan3.flatten()).reshape(heatmap_chan3.shape) * 255
    png_heatmap_chan3 = png_heatmap_chan3.astype("uint8")

    if output_suffix:
        cv2.imwrite(f'heatmap_chan1_{output_suffix}_{colorspace}.png', png_heatmap_chan1)
        cv2.imwrite(f'heatmap_chan2_{output_suffix}_{colorspace}.png', png_heatmap_chan2)
        cv2.imwrite(f'heatmap_chan3_{output_suffix}_{colorspace}.png', png_heatmap_chan3)
        
    return png_heatmap_chan1, png_heatmap_chan2, png_heatmap_chan3


def downsamplesd_to_og_coords(coord, downsamples):
    """
    convert coordinate in a downampled image obtained by pyramidal downsampling
    of original image <downsamples> times, to corresponding coordinates in 
    original image
    """
    og = coord
    for i in range(downsamples):
        og *= 2
    return og


def get_crop_boundaries(video_path, colorspace, low_freq, high_freq, downsamples, target_channel, padding):
    """
    Perform rough boundary analysis on downsampled version of video in order to identify region to run
    analysis on with full resolution
    """
    min_contour_area = 100

    images, fps =  loadVideo(video_path, colorspace=colorspace, downsamples=downsamples)
    chan1, chan2, chan3  = fftVideo(images, fps, low_freq, high_freq, colorspace=colorspace)
    if target_channel == 0:
        target_chan = chan1
    elif target_channel == 1:
        target_chan = chan2
    elif target_channel == 2:
        target_chan = chan3
    else: 
        print("Invalid target channel")
        sys.exit(0)

    #perform analysis of downsampled heatmap corresponding to desired channel
    blur_target_chan = cv2.GaussianBlur(target_chan,(5,5),0)
    otsu_ret, _ = cv2.threshold(blur_target_chan,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, target_chan_th = cv2.threshold(blur_target_chan,otsu_ret - 30,255,cv2.THRESH_BINARY)
    #target_chan_th = cv2.adaptiveThreshold(target_chan,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 399, 0)
    # kernel = np.ones((3, 3), np.uint8) #must use odd kernel to avoid shifting
    # target_chan_th = cv2.erode(target_chan_th, kernel, iterations=2)
  
    contours,hierarchy = cv2.findContours(target_chan_th, 1, 2)
    vis_target_chan_th = cv2.cvtColor(target_chan_th, cv2.COLOR_GRAY2BGR)
    vis_target_chan_th= cv2.drawContours(vis_target_chan_th, contours, -1, (0,255,0), 1)
    left = float('inf')
    right = float('-inf')
    top = float('inf')
    bottom = float('-inf')
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < min_contour_area:
            continue
        vis_target_chan_th = cv2.rectangle(vis_target_chan_th, (x, y), (x+w, y+h), (0, 0, 255), 1)
        if x + w > right:
            right = x + w
        if x < left: 
            left = x
        if y + h > bottom:
            bottom = y + h
        if y < top: 
            top = y
    
    vis_target_chan_th = cv2.rectangle(vis_target_chan_th, (left, top), (right, bottom), (255, 0, 0), 1)

    cv2.imshow('Identified crops', vis_target_chan_th)
    cv2.waitKey(0)

    #convert downsamplesd crop coordinates to coordinates in original image
    left_crop_og = downsamplesd_to_og_coords(left, downsamples) - padding
    right_crop_og = downsamplesd_to_og_coords(right, downsamples) + padding
    top_crop_og = downsamplesd_to_og_coords(top, downsamples) - padding
    bottom_crop_og = downsamplesd_to_og_coords(bottom, downsamples) + padding

    return [left_crop_og, right_crop_og, top_crop_og, bottom_crop_og]


def generate_heatmap(video_path, low_freq, high_freq, colorspace='ycrcb', target_channel=1, padding=30, downsamples=1):

    vid_name = video_path.split('/')[-1].split('.')[0]

    cap_temp = cv2.VideoCapture(video_path) 
    W, H = cap_temp.get(3), cap_temp.get(4)
    if W*H > 250000: #need to do two rounds of FFT
        #determine number of pyramidal downsampless need to get resolution to acceptable number
        downsamples = 1
        while (W / 2) * (H / 2) >  250000:
            downsamples += 1
        print("Running initial FFT with frames downsamples x{} to identify crop boundaries.".format(downsamples))
        crop_coords = get_crop_boundaries(video_path, colorspace, low_freq, high_freq, downsamples, target_channel, padding)
        
        #now perform fft on crop of original image
        images, fps =  loadVideo(video_path, colorspace=colorspace, crop_coords=crop_coords)
        _ , _ , _  = fftVideo(images, fps, low_freq, high_freq, colorspace=colorspace, output_suffix=vid_name)
    else:
        images, fps =  loadVideo(video_path, colorspace=colorspace)
        _ , _ , _  = fftVideo(images, fps, low_freq, high_freq, colorspace=colorspace, output_suffix=vid_name)
       

generate_heatmap('test_vids/aug23_input_videos/r60_g0_b0_N30_b30_s2.MP4', .45, .55, target_channel=2, padding=30, colorspace='bgr')

