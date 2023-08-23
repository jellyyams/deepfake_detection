import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm 
import pywt
from modwt import modwt, modwtmra


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


colorspace = 'bgr'
gaussian_pyr = False
level = 2

vid_dir = 'aug6_cropped_videos'
video_path = f'../test_vids/{vid_dir}/r60_g0_b0_1000mson_1000msoff_N30_buff30_rep15_croppedrgba.avi'

axis = 0
images, fps =  loadVideo(video_path, colorspace=colorspace)
# if gaussian_pyr:
#     images = getGaussianPyramids(
#                             images=images,
#                             level=level


norm_images = images.copy()
norm_images = np.where(((np.max(norm_images, axis=axis) - np.min(norm_images, axis=axis)) == 0), 0.5, (norm_images - np.min(norm_images, axis=axis)) / (np.max(norm_images, axis=axis) - np.min(norm_images, axis=axis)))
norm_images = norm_images - np.mean(norm_images, axis=axis)

# freqs1 = np.fft.fftfreq(images.shape[0], 1/fps)
# idx1 = np.argsort(freqs1)
# ps1 = np.abs(np.fft.fft(norm_images, axis=axis))**2




target_pixel = [120, 50]
target_chan = 2
signal = norm_images[:, target_pixel[0], target_pixel[1], target_chan]

# fig, axes = plt.subplots(figsize=(12, 4), nrows=3, ncols=1)
# ax = axes[0]
# ax.plot(signal)
# (cA, cD) = pywt.dwt(signal, 'db1')
# ax = axes[1]
# ax.plot(cA)
# ax = axes[2]
# ax.plot(cD)
# plt.show()

lev = 4
w = modwt(signal, 'haar', lev)
fig, ax = plt.subplots(lev + 2, 1, sharex=True)
ax[0].plot(signal, c='r')
for k in range(1, lev + 1):
    ax[k].plot(w[k], c='b')
plt.show()