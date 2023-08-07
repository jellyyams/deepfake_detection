import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm 

def loadVideo(video_path, colorspace='bgr'):
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


def fftVideo(video_path, axis=0, gaussian_pyr=False, level=3):
    # on = 500
    # off = 500
    # tp = on
    # T = on + off
    # A = 1
    # n = np.linspace(0, 100)
    # coeffs = ((2*A)/(n*np.pi))*np.sin(n*(np.pi*tp/T))
    # plt.stem(n, coeffs)
    # plt.show()

    # w = n*(1/(T/1000))
    # plt.stem(w, coeffs)
    # plt.show()
   

    images, fps =  loadVideo(video_path, colorspace='ycrcb')
    if gaussian_pyr:
        images = getGaussianPyramids(
                                images=images,
                                level=level
                        )
    # cv2.imshow('test', images[0,:,:,:])
    # cv2.waitKey(0)

    norm_images = images.copy()
    #norm_images = moving_average(norm_images, 0, n=5)
    #norm_images = (norm_images - np.min(norm_images, axis=axis)) / (np.max(norm_images, axis=axis) - np.min(norm_images, axis=axis))
    norm_images = np.where(((np.max(norm_images, axis=axis) - np.min(norm_images, axis=axis)) == 0), 0.5, (norm_images - np.min(norm_images, axis=axis)) / (np.max(norm_images, axis=axis) - np.min(norm_images, axis=axis)))
    norm_images = norm_images - np.mean(norm_images, axis=axis)
    ps = np.abs(np.fft.fft(norm_images, axis=axis))**2
    freqs = np.fft.fftfreq(images.shape[0], 1/fps)
    idx = np.argsort(freqs)
    target_freqs = [.48, .52]
    low_freq_index = (np.abs(freqs - target_freqs[0])).argmin()
    high_freq_index = (np.abs(freqs - target_freqs[1])).argmin()
    
    heatmaps = np.sum(ps[low_freq_index:high_freq_index,:,:,:], axis=axis)
    #heatmaps = heatmaps.astype(int)

    heatmap = heatmaps[:,:,1]
    ax = sns.heatmap(heatmap, linewidth=0)
    plt.title('Channel 2')
    plt.show()
    
    heatmap = heatmaps[:,:,2]
    ax = sns.heatmap(heatmap, linewidth=0) 
    plt.title('Channel 3')
    plt.show()

    heatmap = heatmaps[:,:,0]
    ax = sns.heatmap(heatmap, linewidth=0)
    plt.title('Channel 1')
    plt.show()
    

# a = np.linspace((1, 2, 3, 4, 5, 6, 7, 8, 9, 10),(10, 11, 12, 13, 14, 15, 16, 17, 18, 19),10)
# b = np.linspace((21, 22, 23, 24, 25, 26, 27, 28, 29, 30),(31, 32, 33, 34, 35, 36, 36, 37, 38, 39),10)
# threed = np.dstack((a, b)).T
# print(threed.shape)
# print(threed)
# an = moving_average(threed, 0, n=2)
# print('+++++++++++++++++++')
# print(an)