from ctypes import c_double, c_int, c_char_p, CDLL
import sys
import numpy as np
from PIL import Image as im
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("on_img_path", type=str, help = "on image path")
parser.add_argument("off_img_path", type=str,  help = "off img path")
parser.add_argument("-o", "--on", type=int, action='store', dest='on', help = "On time (ms)")
parser.add_argument("-f", "--off", type=int, action='store', dest='off', help = "Off time (ms)")
#parser.add_argument("-l", "--logpath", type=str, action='store', dest='log_path', help = "path to log file")
parser.add_argument("-rep", "--repeat", type=int, action='store', dest='rep', help = "number of times to repeat image flash")
args = parser.parse_args()

lib_path = './code_test'

basic_function_lib = CDLL(lib_path)

python_c_mod = basic_function_lib.python_main_handler
python_c_mod.restype = None

on_us = args.on * 1000 #us
off_us = args.off * 1000 #us
r = 30 

log_path = '{}_{}mson_{}msoff_rep{}.txt'.format(args.on_img_path.split('on_frame_')[-1].split('.bmp')[0], args.on, args.off, args.rep)

print("Running display")

on_img_path = args.on_img_path.encode('utf-8')
off_img_path = args.off_img_path.encode('utf-8')
log_path = log_path.encode('utf-8')
python_c_mod.argtypes = [c_char_p, c_char_p, c_char_p, c_int, c_int, c_int]
python_c_mod(on_img_path, off_img_path, log_path, on_us, off_us, args.rep)
