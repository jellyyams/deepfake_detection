from ctypes import c_double, c_int, c_char_p, CDLL
import sys
import numpy as np
from PIL import Image as im
import cv2
import argparse

W = 640
H = 360

def generate_gradient(output_dir_path, channel):
    #not tested
    for i in range(0, 255):
        img = np.zeros((H, W, 3))
        i_str = "{:0>{}}".format(i, 3)
        if channel == 'r':
            img[:,:, 0] = 0
            img[:,:, 1] = 0
            img[:,:, 2] = i
            labeled_img = cv2.putText(img, '0,0,{}'.format(i), (6, 14), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
            output_path = '{}/r{}-g0-b0.bmp'.format(output_dir_path, i_str)
        elif channel == 'b':
            img[:,:, 0] = i
            img[:,:, 1] = 0
            img[:,:, 2] = 0
            labeled_img = cv2.putText(img, '{},0,0'.format(i), (6, 14), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
            output_path = '{}/r0-g0-b{}.bmp'.format(output_dir_path, i_str)
        else:
            img[:,:, 0] = 0
            img[:,:, 1] = i
            img[:,:, 2] = 0
            labeled_img = cv2.putText(img, '0,{},0'.format(i), (6, 14), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
            output_path = '{}/r0-g{}-b0.bmp'.format(output_dir_path, i_str)

        labeled_img = cv2.flip(labeled_img, 0)
        cv2.imwrite(output_path, labeled_img)

def generate_solid_color(output_path, r, g, b):
    img = np.zeros((H, W, 3))
    img[:,:, 0] = b
    img[:,:, 1] = g
    img[:,:, 2] = r

    cv2.imwrite(output_path, img)



parser = argparse.ArgumentParser()
parser.add_argument("red", type=int, help = "red channel value")
parser.add_argument("green", type=int,  help = "green channel value")
parser.add_argument("blue", type=int,  help = "blue channel value")
parser.add_argument("-o", "--on", type=int, action='store', dest='on', help = "On time (ms)")
parser.add_argument("-f", "--off", type=int, action='store', dest='off', help = "Off time (ms)")
#parser.add_argument("-l", "--logpath", type=str, action='store', dest='log_path', help = "path to log file")
parser.add_argument("-rep", "--repeat", type=int, action='store', dest='rep', help = "number of times to repeat image flash")
args = parser.parse_args()

lib_path = './mod_disp'

basic_function_lib = CDLL(lib_path)

python_c_mod = basic_function_lib.python_main_handler
python_c_mod.restype = None

print("Creating bmp")

generate_solid_color('temp.bmp', args.red, args.green, args.blue)
img_path = "temp.bmp"
on_us = args.on * 1000 #us
off_us = args.off * 1000 #us
r = 30 

log_path = 'r{}_g{}_b{}_{}mson_{}msoff_rep{}.txt'.format(args.red, args.green, args.blue, args.on, args.off, args.rep)

print("Running display")

img_path = img_path.encode('utf-8')
log_path = log_path.encode('utf-8')
python_c_mod.argtypes = [c_char_p, c_char_p, c_int, c_int, c_int]
python_c_mod(img_path, log_path, on_us, off_us, args.rep)

