import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy
from flash_gt import gen_blink_list,gen_blink_list_from_log,  find_led_blink_start
from datetime import datetime

def temporal_bandpass_filter(data, fps, freq_min=0.833, freq_max=1, axis=0):
   # print "Applying bandpass between " + str(freq_min) + " and " + str(freq_max) + " Hz"
    # perform FFT on each frame
    fft = scipy.fftpack.fft(data, axis=axis)
    # sampling frequencies, where the step d is 1/samplingRate
    frequencies = scipy.fftpack.fftfreq(data.shape[0], d=1.0 / fps)
    # plt.stem(frequencies[:100], fft[:100])
    # plt.show()
    # find the indices of low cut-off frequency
    bound_low = (np.abs(frequencies - freq_min)).argmin()
    # find the indices of high cut-off frequency
    bound_high = (np.abs(frequencies - freq_max)).argmin()
    # band pass filtering
    fft[:bound_low] = 0
    fft[-bound_low:] = 0
    fft[bound_high:-bound_high] = 0
    # perform inverse FFT
    return scipy.fftpack.ifft(fft, axis=0)


"""
https://stackoverflow.com/questions/13728392/moving-average-or-running-mean
"""
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)


input_vid_path = 'cropped_videos/r10_g0_b0_1000mson_1000msoff_rep20_croppedrgba.avi'
input_capture = cv2.VideoCapture(input_vid_path)

"""
r10 start 2023-07-24 16:04:51.19
r13 start '2023-07-24 16:03:41.50'
"""
vid_start_dt = datetime.strptime('2023-07-24 16:04:51.19', '%Y-%m-%d %H:%M:%S.%f')
print(vid_start_dt)
vid_start_millisec = vid_start_dt.timestamp() * 1000
# blink_start_ms = find_led_blink_start(vid_start_millisec, 'input_logs/r50_g0_b0_1000mson_1000msoff_rep20.txt')
# blink_start_dt = datetime.fromtimestamp(start_ts/1000.0))
led_log_path = 'input_logs/r10_g0_b0_1000mson_1000msoff_rep20.txt'
# blinks = gen_blink_list(input_vid_path, vid_start_dt, 1100, 1000, led_log_path=led_log_path)
blinks = gen_blink_list_from_log(vid_start_dt, led_log_path)
print(blinks)
cap = cv2.VideoCapture(input_vid_path)
fps = cap.get(cv2.CAP_PROP_FPS)
# xlim = (0, 40 )
# target_ontime = .2
# target_offtime= .1 #seconds

bgr_chan1 = []
bgr_chan2 = []
bgr_chan3 = []

ycrbr_chan1 = []
ycrbr_chan2 = []
ycrbr_chan3 = []

yuv_chan1 = []
yuv_chan2 = []
yuv_chan3 = []
while input_capture.isOpened():
    ret, frame = input_capture.read()
    if ret:
        bgr_chan1_mean = np.mean(frame[:,:, 0])
        bgr_chan2_mean = np.mean(frame[:,:, 1])
        bgr_chan3_mean = np.mean(frame[:,:, 2])
        bgr_chan1.append(bgr_chan1_mean)
        bgr_chan2.append(bgr_chan2_mean)
        bgr_chan3.append(bgr_chan3_mean)

        ycrcb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
        ycrbr_chan1_mean = np.mean( ycrcb_frame[:,:, 0])
        ycrbr_chan2_mean = np.mean( ycrcb_frame[:,:, 1])
        ycrbr_chan3_mean = np.mean( ycrcb_frame[:,:, 2])
        ycrbr_chan1.append(ycrbr_chan1_mean)
        ycrbr_chan2.append(ycrbr_chan2_mean)
        ycrbr_chan3.append(ycrbr_chan3_mean)

        yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        yuv_chan1_mean = np.mean(yuv_frame[:,:, 0])
        yuv_chan2_mean = np.mean(yuv_frame[:,:, 1])
        yuv_chan3_mean = np.mean(yuv_frame[:,:, 2])
        yuv_chan1.append(yuv_chan1_mean)
        yuv_chan2.append(yuv_chan2_mean)
        yuv_chan3.append(yuv_chan3_mean)

    else:
        break

bgr_chan1 = np.array(bgr_chan1)
bgr_chan2 = np.array(bgr_chan2)
bgr_chan3 = np.array(bgr_chan3)


ycrbr_chan1 = np.array(ycrbr_chan1)
ycrbr_chan2 = np.array(ycrbr_chan2)
ycrbr_chan3 = np.array(ycrbr_chan3)


yuv_chan1 = np.array(yuv_chan1)
yuv_chan2 = np.array(yuv_chan2)
yuv_chan3 = np.array(yuv_chan3)

# bgr_chan12 = bgr_chan1 / bgr_chan2
# bgr_chan13 = bgr_chan1 / bgr_chan3
# bgr_chan23= bgr_chan2 / bgr_chan3

# ycrbr_chan12 = ycrbr_chan1 / ycrbr_chan2
# ycrbr_chan13 = ycrbr_chan1 / ycrbr_chan3
# ycrbr_chan23 = ycrbr_chan2 / ycrbr_chan3

# yuv_chan12 = yuv_chan1 / yuv_chan2
# yuv_chan13 = yuv_chan1 / yuv_chan3
# yuv_chan23 = yuv_chan2 / yuv_chan3


# x = [i/fps for i in range(len(bgr_chan1))]

def make_plot(data_lists, blinks, title=None):
    N = len(data_lists)
    fig, axes = plt.subplots(nrows=N, ncols=1)
    # plt.figure(figsize=(15, 5))
    if title != None:
        fig.suptitle(title)
    for i in range(N):
        data = data_lists[i]
        x = [i/fps for i in range(len(data))]
        axes[i].plot(x, data)
        axes[i].vlines(blinks, data.min(), data.max(), colors='r', linewidth=1, alpha=0.8)
        # axes[i].set_xlim(xlim)
        #x_labels = ['{0:.2f}'.format(i/fps) for i in range(0, len(data), 100)]
        #axes[i].set_xticks(x, x_labels, rotation=60)
    plt.show()

# fig = plt.figure(1, figsize=(15, 5))
# fig.suptitle('YUV Values')
# plt.subplot(311)
# plt.plot(yuv_chan1)
# # plt.xlim(xlim)
# plt.subplot(312)
# plt.plot(yuv_chan2)
# # plt.xlim(xlim)
# plt.subplot(313)
# plt.plot(yuv_chan3)
# # plt.xlim(xlim)
# plt.show()

make_plot([yuv_chan1, yuv_chan2, yuv_chan3], blinks, title='Test')

# fig = plt.figure(1, figsize=(15, 5))
# fig.suptitle('YCrCb Values')
# plt.subplot(311)
# plt.plot(x, ycrbr_chan1)
# plt.xlim(xlim)
# plt.subplot(312)
# plt.plot(x, ycrbr_chan2)
# plt.xlim(xlim)
# plt.subplot(313)
# plt.plot(x, ycrbr_chan3)
# plt.xlim(xlim)
# plt.show()

# fig = plt.figure(1, figsize=(15, 5))
# fig.suptitle('BGR Values')
# plt.subplot(311)
# plt.plot(x, bgr_chan1)
# plt.xlim(xlim)
# plt.subplot(312)
# plt.plot(x, bgr_chan2)
# plt.xlim(xlim)
# plt.subplot(313)
# plt.plot(x, bgr_chan3)
# plt.xlim(xlim)
# plt.show()


# bgr_chan1_zeromean = bgr_chan1 - np.mean(bgr_chan1)
# bgr_chan2_zeromean = bgr_chan2 - np.mean(bgr_chan2)
# bgr_chan3_zeromean = bgr_chan3 - np.mean(bgr_chan3)
# filtered_bgr_chan1 = temporal_bandpass_filter(bgr_chan1_zeromean, fps, freq_min=4, freq_max=6)
# filtered_bgr_chan2 = temporal_bandpass_filter(bgr_chan2_zeromean, fps, freq_min=4, freq_max=6)
# filtered_bgr_chan3 = temporal_bandpass_filter(bgr_chan3_zeromean, fps, freq_min=4, freq_max=6)

# fig = plt.figure(1, figsize=(15, 5))
# fig.suptitle('R:G vs. Filtered R:Filtered G')
# plt.subplot(611)
# plt.plot(x, bgr_chan2)
# plt.xlim(xlim)
# plt.subplot(612)
# plt.plot(x, filtered_bgr_chan2)
# plt.xlim(xlim)
# plt.subplot(613)
# plt.plot(x, bgr_chan3)
# plt.xlim(xlim)
# plt.subplot(614)
# plt.plot(x, filtered_bgr_chan3)
# plt.xlim(xlim)
# plt.subplot(615)
# plt.plot(x, bgr_chan3/bgr_chan2)
# plt.xlim(xlim)
# plt.subplot(616)
# plt.plot(x, filtered_bgr_chan3/filtered_bgr_chan2)
# plt.xlim(xlim)
# plt.show()

# meow = bgr_chan3_zeromean + filtered_bgr_chan3*4
# fig = plt.figure(1, figsize=(15, 5))
# fig.suptitle('hi')
# plt.subplot(611)
# plt.plot(x, bgr_chan3_zeromean)
# plt.subplot(612)
# plt.plot(x, filtered_bgr_chan3)
# meow_minmax = (meow - meow.min()) / (meow.max() - meow.min())
# bgr_chan2_zeromean_minmax = (bgr_chan2_zeromean - bgr_chan2_zeromean.min()) / (bgr_chan2_zeromean.max() - bgr_chan2_zeromean.min())
# plt.subplot(613)
# plt.plot(x, meow_minmax)
# plt.subplot(614)
# plt.plot(x, bgr_chan2_zeromean_minmax)
# plt.subplot(615)
# plt.title('og')
# plt.plot(x, bgr_chan3/ bgr_chan2)
# plt.subplot(616)
# plt.title('amp?')
# plt.plot(x, meow_minmax/bgr_chan2_zeromean_minmax)
# plt.ylim(0, 10)
# plt.show()


# N = int(fps * target_offtime)
# bgr_chan1_runavg = running_mean(bgr_chan1, N)
# bgr_chan2_runavg = running_mean(bgr_chan2, N)
# bgr_chan3_runavg = running_mean(bgr_chan3, N)
# ycrbr_chan2_runavg = running_mean(ycrbr_chan2, N)
# fig = plt.figure(1, figsize=(15, 5))
# fig.suptitle('hi2')
# x_runavg = [i/fps for i in range(len(bgr_chan1_runavg))]
# plt.subplot(611)
# plt.plot(x, bgr_chan3/bgr_chan2, c='black', label='R:G')
# plt.xlim(xlim)
# plt.legend()
# plt.subplot(612)
# plt.plot(x_runavg, bgr_chan3_runavg/bgr_chan2_runavg, c='orange', label='R:G Avg')
# plt.xlim(xlim)
# plt.legend()
# plt.subplot(613)
# plt.plot(x, bgr_chan3/bgr_chan1, c='black', label='R:B')
# plt.xlim(xlim)
# plt.legend()
# plt.subplot(614)
# plt.plot(x_runavg, bgr_chan3_runavg/bgr_chan1_runavg, c='purple', label='R:B Avg')
# plt.xlim(xlim)
# plt.legend()
# plt.subplot(615)
# plt.plot(x, ycrbr_chan2, c='black', label='Cr')
# plt.xlim(xlim)
# plt.legend()
# plt.subplot(616)
# plt.plot(x_runavg, ycrbr_chan2_runavg, c='red', label='Cr Avg')
# plt.xlim(xlim)
# plt.legend()
# plt.show()
 
# W = int(fps * (target_offtime + target_offtime)) + 1
# print(W)
# bgr_chan32 = bgr_chan3 / bgr_chan2
# b = bgr_chan32.cumsum()
# b[W:] = b[W:] - b[:-W]
# fig = plt.figure(1, figsize=(15, 5))
# fig.suptitle('Test overlapping window')
# plt.subplot(211)
# plt.plot(x, bgr_chan32)
# plt.xlim(xlim)
# plt.subplot(212)
# plt.xlim(xlim)
# xlim_min_i = x.index(xlim[0])
# xlim_max_i = x.index(xlim[1])
# print(xlim_min_i)
# print(xlim_max_i)
# print(len(b))
# plt.ylim(b[xlim_min_i:xlim_max_i].min(), b[xlim_min_i:xlim_max_i].max())
# plt.plot(x, b)
# plt.show()


# ycrbr_chan1_zeromean = ycrbr_chan1 - np.mean(ycrbr_chan1)
# ycrbr_chan2_zeromean = ycrbr_chan2 - np.mean(ycrbr_chan2)
#ycrbr_chan3_zeromean = ycrbr_chan3 - np.mean(ycrbr_chan3)

# filtered_ycrbr_chan1 = temporal_bandpass_filter(ycrbr_chan1_zeromean, fps, freq_min = 1, freq_max=1.4)
# filtered_ycrbr_chan2 = temporal_bandpass_filter(ycrbr_chan2_zeromean, fps, freq_min = 1, freq_max=1.4)
# filtered_ycrbr_chan3 = temporal_bandpass_filter(ycrbr_chan3_zeromean, fps, freq_min = 1, freq_max=1.4)

# fig = plt.figure(1, figsize=(15, 5))
# plt.subplot(311)
# plt.plot(x, ycrbr_chan3_zeromean)
# plt.subplot(312)
# plt.plot(x, filtered_ycrbr_chan3)
# plt.subplot(313)
# plt.plot(x, ycrbr_chan3_zeromean - filtered_ycrbr_chan3)
# plt.show()


# test_chan2 = ycrbr_chan2_zeromean - filtered_ycrbr_chan2
# test_chan3 = ycrbr_chan3_zeromean - filtered_ycrbr_chan3
# test_chan2_minmax = (test_chan2 - test_chan2.min()) / (test_chan2.max() - test_chan2.min())
# test_chan3_minmax = (test_chan3 - test_chan3.min()) / (test_chan3.max() - test_chan3.min())
# fig = plt.figure(1, figsize=(15, 5))
# fig.suptitle('Test')
# plt.subplot(311)
# plt.plot(x, test_chan2_minmax)
# plt.subplot(312)
# plt.plot(x, test_chan3_minmax)
# plt.subplot(313)
# plt.plot(x, test_chan2_minmax/test_chan3_minmax)
# plt.ylim(0, 5)
# plt.show()





# test2_chan2 = ycrbr_chan2_zeromean + filtered2_ycrbr_chan2*2
# test2_chan3 = ycrbr_chan3_zeromean 
# test2_chan2_minmax = (test2_chan2 - test2_chan2.min()) / (test2_chan2.max() - test2_chan2.min())
# test2_chan3_minmax = (test2_chan3 - test2_chan3.min()) / (test2_chan3.max() - test2_chan3.min())
# fig = plt.figure(1, figsize=(15, 5))
# fig.suptitle('Test 2')
# plt.subplot(411)
# plt.plot(x, ycrbr_chan2_zeromean)
# plt.subplot(412)
# plt.plot(x, test2_chan2_minmax)
# plt.subplot(413)
# plt.plot(x, test2_chan3_minmax)
# plt.subplot(414)
# plt.plot(x, test2_chan2_minmax/test2_chan3_minmax)
# plt.show()



# N = 5
# ycrbr_chan1_zeromean_runavg = running_mean(ycrbr_chan1_zeromean, N)
# ycrbr_chan2_zeromean_runavg = running_mean(ycrbr_chan2_zeromean, N)
# ycrbr_chan3_zeromean_runavg = running_mean(ycrbr_chan3_zeromean, N)
# x_runavg = [i/fps for i in range(len(ycrbr_chan1_zeromean_runavg))]

# fig = plt.figure(1, figsize=(15, 5))
# fig.suptitle('YCrBr Zero-Mean Values')
# plt.subplot(411)
# plt.plot(x, ycrbr_chan1_zeromean)
# plt.subplot(412)
# plt.plot(x, ycrbr_chan2_zeromean)
# plt.subplot(413)
# plt.plot(x, ycrbr_chan3_zeromean)
# plt.subplot(414)
# plt.title('Cr:Br')
# plt.plot(x, ycrbr_chan2_zeromean/ycrbr_chan3_zeromean)
# plt.ylim(-10, 10)
# plt.show() 

# fig = plt.figure(1, figsize=(15, 5))
# fig.suptitle('YCrBr Zero-Mean Values - Running Avergae')
# plt.subplot(411)
# plt.plot(x_runavg, ycrbr_chan1_zeromean_runavg)
# plt.subplot(412)
# plt.plot(x_runavg, ycrbr_chan2_zeromean_runavg)
# plt.subplot(413)
# plt.plot(x_runavg, ycrbr_chan3_zeromean_runavg)
# plt.subplot(414)
# plt.plot(x_runavg, ycrbr_chan2_zeromean_runavg/ycrbr_chan3_zeromean_runavg)

# plt.show() 




# fig = plt.figure(1, figsize=(15, 5))
# plt.subplot(311)
# plt.plot(x, ycrbr_chan2)
# plt.subplot(312)
# plt.plot(x, ycrbr_chan3)
# plt.subplot(313)
# plt.plot(x, ycrbr_chan2/ycrbr_chan3)
# plt.show()