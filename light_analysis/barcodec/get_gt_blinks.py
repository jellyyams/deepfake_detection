import cv2
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

def find_led_blink_start(vid_start_millisec, led_log_path):
    """
    Given a video start timestamp vid_start_dt and light flash log
    , find the timestamp of the first light flash in the video
    """
    led_log = open(led_log_path, 'r')
    for line in led_log:
        if 'On:' in line:
            log_timestamp = line.split('On: ')[-1].replace('\n', '')
            led_blink_dt = datetime.strptime(log_timestamp,
                                             '%Y:%m:%d %H:%M:%S.%f')
            led_blink_dt = led_blink_dt - timedelta(hours=4)
            led_blink_millisec = led_blink_dt.timestamp() * 1000
            if led_blink_millisec >= vid_start_millisec:
                print('First blink in video: ', led_blink_dt)
                return led_blink_millisec

def gen_blink_list_from_log(vid_start_dt, led_log_path):
    vid_start_millisec = vid_start_dt.timestamp() * 1000
    led_log = open(led_log_path, 'r')
    out = []
    for line in led_log:
        if 'On:' in line:
            log_timestamp = line.split('On: ')[-1].replace('\n', '')
            led_blink_dt = datetime.strptime(log_timestamp,
                                             '%Y:%m:%d %H:%M:%S.%f')
            led_blink_dt = led_blink_dt - timedelta(hours=4)
            led_blink_millisec = led_blink_dt.timestamp() * 1000
            if led_blink_millisec >= vid_start_millisec:
                out.append((led_blink_millisec - vid_start_millisec)/1000)
    return out

def gen_blink_list(video_path, vid_start_dt, LED_ontime, LED_offtime, blink_start_dt = None, led_log_path=None):
    vid_start_millisec = vid_start_dt.timestamp() * 1000
    if blink_start_dt == None:
        blink_start_millisec =  find_led_blink_start(vid_start_millisec, led_log_path)
    else:
        blink_start_millisec = blink_start_dt.timestamp() * 1000
    
    cap = cv2.VideoCapture(video_path)
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    input_cap_fps = cap.get(cv2.CAP_PROP_FPS)
    len_sec = N // input_cap_fps
    offset = (blink_start_millisec - vid_start_millisec)/1000
    out = []
    blink_until = offset + LED_ontime/1000
    curr_blink_start = offset
    next_blink_start = offset + LED_ontime/1000 + LED_offtime/1000
    inc = True
    for i in np.arange(offset, len_sec, .001):
        if i > blink_until and inc:
            inc = False
            curr_blink_start = next_blink_start
            blink_until = curr_blink_start + LED_ontime/1000
            next_blink_start += LED_ontime/1000 + LED_offtime/1000
        if i > curr_blink_start and i < blink_until:
            inc = True
            out.append(i)
    return out
