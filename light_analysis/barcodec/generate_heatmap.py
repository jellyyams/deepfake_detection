import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm 
from sklearn.preprocessing import minmax_scale


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
    downsampled_image = image.copy()

    for _ in range(level):
        downsampled_image = pyrDown(image=downsampled_image, kernel=gaussian_kernel)
        image_shape.append(downsampled_image.shape[:2])

    gaussian_pyramid = downsampled_image
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

def heatmap_to_png(heatmap, out_path):
    normed = minmax_scale(heatmap.flatten()).reshape(heatmap.shape) * 255
    cv2.imwrite(out_path, normed)


def fftVideo(video_path, low_freq, high_freq, colorspace = 'ycrcb', sub_path = None, gaussian_pyr=False, level=3, save_heatmap=False, heatmap_out_path = 'heatmap.npy', heatmap_norm = False):
    axis = 0
    images, fps =  loadVideo(video_path, colorspace=colorspace)
    if gaussian_pyr:
        images = getGaussianPyramids(
                                images=images,
                                level=level
                        )
    
    norm_images = images.copy()
    if colorspace == 'gnorm':
        norm_images[:,:,:,0] = norm_images[:,:,:,0] / norm_images[:,:,:,1]
        norm_images[:,:,:,1] = norm_images[:,:,:,1] / norm_images[:,:,:,1]
        norm_images[:,:,:,2] = norm_images[:,:,:,2] / norm_images[:,:,:,1]
    #norm_images = moving_average(norm_images, 0, n=5)
    #norm_images = (norm_images - np.min(norm_images, axis=axis)) / (np.max(norm_images, axis=axis) - np.min(norm_images, axis=axis))
    norm_images = np.where(((np.max(norm_images, axis=axis) - np.min(norm_images, axis=axis)) == 0), 0.5, (norm_images - np.min(norm_images, axis=axis)) / (np.max(norm_images, axis=axis) - np.min(norm_images, axis=axis)))
    norm_images = norm_images - np.mean(norm_images, axis=axis)
   

    ps = np.abs(np.fft.fft(norm_images, axis=axis))**2


    freqs = np.fft.fftfreq(images.shape[0], 1/fps)
    idx = np.argsort(freqs)
    target_freqs = [low_freq, high_freq]
    low_freq_index = (np.abs(freqs - target_freqs[0])).argmin()
    high_freq_index = (np.abs(freqs - target_freqs[1])).argmin()
    
    # plt.plot(images[:, 50, 210, 1])
    # plt.title('Pixel (210, 50) Cr Channel')
    # plt.ylabel('Cr Channel Value')
    # plt.xlabel('Time')
    # plt.show()

    # plt.stem(freqs[idx], ps[idx, 50, 210, 1])
    # plt.xlabel('Frequency')
    # plt.title('PSD: Pixel (210, 50) Cr Channel')
    # plt.show()

    heatmaps = np.sum(ps[low_freq_index:high_freq_index,:,:,:], axis=axis)
    heatmap_chan1 = heatmaps[:,:,0]
    heatmap_chan2 = heatmaps[:,:,1]
    heatmap_chan3 = heatmaps[:,:,2]
    if heatmap_norm:
        heatmap_chan1 = (heatmap_chan1 - np.min(heatmap_chan1))/(np.max(heatmap_chan1) - np.min(heatmap_chan1))
        heatmap_chan2 = (heatmap_chan2 - np.min(heatmap_chan2))/(np.max(heatmap_chan2) - np.min(heatmap_chan2))
        heatmap_chan3 = (heatmap_chan3 - np.min(heatmap_chan3))/(np.max(heatmap_chan3) - np.min(heatmap_chan3))
    
    if sub_path != None:
        sub_heatmap = np.load(sub_path)
        print(heatmap_chan1[100,100])
        print(sub_heatmap[100,100,0])
        heatmap_chan1 = heatmap_chan1 - sub_heatmap[:,:, 0]
        print(heatmap_chan1[100,100])
        heatmap_chan2 = heatmap_chan2 - sub_heatmap[:,:, 1]
        heatmap_chan3 = heatmap_chan3 - sub_heatmap[:,:, 2]
        heatmap_chan1 = np.clip(heatmap_chan1, 0, 1)
        heatmap_chan2 = np.clip(heatmap_chan2, 0, 1)
        heatmap_chan3 = np.clip(heatmap_chan3, 0, 1)
    if save_heatmap:
        out_heatmap = np.dstack((np.dstack((heatmap_chan1, heatmap_chan2)), heatmap_chan3))
        np.save(heatmap_out_path, out_heatmap)

    vid_name = video_path.split('/')[-1].split('.')[0]
    
    ax = sns.heatmap(heatmap_chan2, linewidth=0)
    plt.title('Channel 2')
    plt.show()

    heatmap_to_png(heatmap_chan2, f'heatmap_chan2_{vid_name}_{colorspace}.png')
    
    ax = sns.heatmap(heatmap_chan3, linewidth=0) 
    plt.title('Channel 3')
    plt.show()
    
    heatmap_to_png(heatmap_chan3, f'heatmap_chan3_{vid_name}_{colorspace}.png')
    

    ax = sns.heatmap(heatmap_chan1, linewidth=0)
    plt.title('Channel 1')
    plt.show()

    heatmap_to_png(heatmap_chan1, f'heatmap_chan1_{vid_name}_{colorspace}.png')
    

fftVideo('../test_vids/aug5_cropped_videos/r140_g0_b0_1000mson_1000msoff_rep15_croppedrgba.avi', .45, .55)

