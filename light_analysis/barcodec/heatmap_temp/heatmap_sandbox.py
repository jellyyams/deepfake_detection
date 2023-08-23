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


colorspace = 'bgr'
gaussian_pyr = False
level = 2

vid_dir = 'aug6_cropped_videos'
video1_path = f'{vid_dir}/cluttered_nothing1_croppedrgba.avi'
video2_path = f'{vid_dir}/r60_g0_b0_1000mson_1000msoff_N30_buff30_rep15_cluttered_croppedrgba.avi'


axis = 0
images1, fps1 =  loadVideo(video1_path, colorspace=colorspace)
images2, fps2 =  loadVideo(video2_path, colorspace=colorspace)
if gaussian_pyr:
    images1 = getGaussianPyramids(
                            images=images1,
                            level=level
                    )
    images2 = getGaussianPyramids(
                            images=images2,
                            level=level
                    )

norm_images1 = images1.copy()
norm_images1 = np.where(((np.max(norm_images1, axis=axis) - np.min(norm_images1, axis=axis)) == 0), 0.5, (norm_images1 - np.min(norm_images1, axis=axis)) / (np.max(norm_images1, axis=axis) - np.min(norm_images1, axis=axis)))
norm_images1 = norm_images1 - np.mean(norm_images1, axis=axis)

norm_images2 = images2.copy()
norm_images2 = np.where(((np.max(norm_images2, axis=axis) - np.min(norm_images2, axis=axis)) == 0), 0.5, (norm_images2 - np.min(norm_images2, axis=axis)) / (np.max(norm_images2, axis=axis) - np.min(norm_images2, axis=axis)))
norm_images2 = norm_images2 - np.mean(norm_images2, axis=axis)

freqs1 = np.fft.fftfreq(images1.shape[0], 1/fps1)
idx1 = np.argsort(freqs1)
ps1 = np.abs(np.fft.fft(norm_images1, axis=axis))**2

freqs2 = np.fft.fftfreq(images2.shape[0], 1/fps2)
idx2 = np.argsort(freqs2)
ps2 = np.abs(np.fft.fft(norm_images2, axis=axis))**2


pdiff = ps2 - ps1

def plot(target_pixel, target_chan):
    fig, axes = plt.subplots(figsize=(12, 4), nrows=2, ncols=3)
    ax = axes[0,0]
    ax.set_title("Video 1 Time")
    ax.plot(norm_images1[:, target_pixel[0], target_pixel[1], target_chan])
    ax = axes[0,1]
    ax.set_title("Video 1 PSD")
    ax.stem(freqs1[idx1], ps1[idx1, target_pixel[0], target_pixel[1], target_chan])
    ax.set_xlim((-1,1))
    ax = axes[1,0]
    ax.set_title("Video 2 Time")
    ax.plot(norm_images2[:, target_pixel[0], target_pixel[1], target_chan])
    ax = axes[1,1]
    ax.set_title("Video 2 PSD")
    ax.stem(freqs2[idx2], ps2[idx2, target_pixel[0], target_pixel[1], target_chan])
    ax.set_xlim((-1,1))
    ax = axes[0, 2]
    ax.set_title('PSD Diff')
    ax.stem(freqs2[idx2], pdiff[idx2, target_pixel[0], target_pixel[1], target_chan])
    ax.set_xlim((-1,1)  )
    ax = axes[1, 2]
    ax.set_title('Clipped PSD Diff')
    ax.stem(freqs2[idx2], np.clip(pdiff[idx2, target_pixel[0], target_pixel[1], target_chan], 0, 100000))
    ax.set_xlim((-1,1))

    plt.show()

target_chan = 2
plot([135, 20], target_chan)
plot([125, 345], target_chan)
plot([125, 50], target_chan)


# freqs = np.fft.fftfreq(images.shape[0], 1/fps)
# idx = np.argsort(freqs)
# target_pixel_freqs = [low_freq, high_freq]
# low_freq_index = (np.abs(freqs - target_pixel_freqs[0])).argmin()
# high_freq_index = (np.abs(freqs - target_pixel_freqs[1])).argmin()


# heatmaps = np.sum(ps[low_freq_index:high_freq_index,:,:,:], axis=axis)
# heatmap_chan1 = heatmaps[:,:,0]
# heatmap_chan2 = heatmaps[:,:,1]
# heatmap_chan3 = heatmaps[:,:,2]
# if heatmap_norm:
#     heatmap_chan1 = (heatmap_chan1 - np.min(heatmap_chan1))/(np.max(heatmap_chan1) - np.min(heatmap_chan1))
#     heatmap_chan2 = (heatmap_chan2 - np.min(heatmap_chan2))/(np.max(heatmap_chan2) - np.min(heatmap_chan2))
#     heatmap_chan3 = (heatmap_chan3 - np.min(heatmap_chan3))/(np.max(heatmap_chan3) - np.min(heatmap_chan3))

# if sub_path != None:
#     sub_heatmap = np.load(sub_path)
#     print(heatmap_chan1[100,100])
#     print(sub_heatmap[100,100,0])
#     heatmap_chan1 = heatmap_chan1 - sub_heatmap[:,:, 0]
#     print(heatmap_chan1[100,100])
#     heatmap_chan2 = heatmap_chan2 - sub_heatmap[:,:, 1]
#     heatmap_chan3 = heatmap_chan3 - sub_heatmap[:,:, 2]
#     heatmap_chan1 = np.clip(heatmap_chan1, 0, 1)
#     heatmap_chan2 = np.clip(heatmap_chan2, 0, 1)
#     heatmap_chan3 = np.clip(heatmap_chan3, 0, 1)
# if save_heatmap:
#     out_heatmap = np.dstack((np.dstack((heatmap_chan1, heatmap_chan2)), heatmap_chan3))
#     np.save(heatmap_out_path, out_heatmap)



# ax = sns.heatmap(heatmap_chan2, linewidth=0)
# plt.title('Channel 2')
# plt.show()


# ax = sns.heatmap(heatmap_chan3, linewidth=0) 
# plt.title('Channel 3')
# plt.show()

# ax = sns.heatmap(heatmap_chan1, linewidth=0)
# plt.title('Channel 1')
# plt.show()

