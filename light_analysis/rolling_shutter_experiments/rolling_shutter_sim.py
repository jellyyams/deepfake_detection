
import numpy as np
from PIL import Image
import sys

N = 2160
row_vals = [0 for i in range(N)]
mod_freq = 500 #Hz
Tr = 10/(10**6) #readout time - seconds
TeF = 1/72 #exposure time for line - seconds
gap = 1/(10**6) #seconds
#Te = TeF - gap - (N*Tr) #exposure time for line - seconds
Te = TeF #approximation 
if Te < 0:
    print("Invalid line exposure")
    sys.exit(0)
cap_start = 0.01 #start offset relative to light modulation start - seconds
dt = 0.00001 #integration timestep - seconds
max_charge = 2000
im_shape = (N, N*2)

def rect_mod_func(t, freq, phase=0):
    """return 0 if light off at time t, 1 if light on at time t, assuming rectangular modulation function with 
    frequency = freq, phase = 0. Phase = 0 corresponds to pulse train starting off
    """
    pulse_duration  = 1/(freq*2)
    period = 1/freq
    # print(t)
    # print(t % period)
    # print(pulse_duration)
    if t % period > pulse_duration:
        #print('1')
        if phase == 0:
            return 1 
        else:
            return 0
    else:
        #print('0')
        if phase == 0:
            return 0
        else:
            return 1


for i in range(N):
    curr_row_charge = 0
    row_start_t = cap_start + (i * Tr)
    row_end_t = row_start_t + Te
    print(row_start_t, row_end_t)
    for j in np.arange(row_start_t, row_end_t, dt): #"integrate"
        if rect_mod_func(j, mod_freq) == 1:
            curr_row_charge += 1
    row_vals[i] = curr_row_charge
    
print(row_vals)
row_vals = np.array(row_vals)
norm_row_vals = row_vals / np.max(row_vals)
print(norm_row_vals)

im_row_vals = (row_vals / max_charge) * 255
#im_row_vals = norm_row_vals * 255
im_row_vals = im_row_vals.astype('uint8')
im_row_vals = np.clip(im_row_vals, 0, 255)
out_im = np.zeros(im_shape).astype('uint8')
for i in range(N):
    out_im[i, :] = im_row_vals[i]
img = Image.fromarray(out_im, 'L')
img.show()