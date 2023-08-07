import cv2
import numpy as np
import matplotlib.pyplot as plt

input_path = 'mac_test3.png'
video = False
target_frame_num = 31
img  = None
if video:
    cap = cv2.VideoCapture(input_path)
    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_num += 1
            if frame_num == target_frame_num:
                img = frame
                break
        else:
            break
else:
    img = cv2.imread(input_path)

img_b = img[:,:,0]
img_g = img[:,:,1]
img_r = img[:,:,2]

# Compute the discrete Fourier Transform of the image
fourier = cv2.dft(np.float32(img_r), flags=cv2.DFT_COMPLEX_OUTPUT)
 
# Shift the zero-frequency component to the center of the spectrum
fourier_shift = np.fft.fftshift(fourier)
 
#calculate the magnitude of the Fourier Transform
magnitude = 20*np.log(cv2.magnitude(fourier_shift[:,:,0],fourier_shift[:,:,1]))
 
# Scale the magnitude for display
magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
 
# Display the magnitude of the Fourier Transform
cv2.imshow('Fourier Transform', magnitude)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('test_r.png', img_r)

img_b_mean = img_b - np.mean(img_b)
img_g_mean = img_g - np.mean(img_g)
img_r_mean = img_r - np.mean(img_r)

img_r_minb = img_r - img_g
cv2.imwrite('test_min.png', img_r_minb)


# for i in [1000, 1500, 2000]:
#     row = img_r[i,:]
#     plt.plot(row - row.mean())

# plt.show()

# ps = np.abs(np.fft.fft(row - np.mean(row)))**2

# time_step = 1 / 3000
# freqs = np.fft.fftfreq(row.size, time_step)
# idx = np.argsort(freqs)

# plt.plot(freqs[idx], ps[idx])
# plt.show()