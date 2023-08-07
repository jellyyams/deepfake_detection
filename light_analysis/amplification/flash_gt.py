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
# def get_blink_list(video_path, vid_start_dt, led_log_path, LED_ontime, LED_offtime, blink_start_dt = None, epsilon=10, out_form = 'framestamps'):
#     """
#     Given a video of N frames, video start timestamp, and LED blink log, and known LED on/off times (ms),
#     return either:
#     1) a Boolean list of length N, where an entry 1 corresponds to the LED being 
#     on at that frame, 0 off at that frame IF out_form = 'booleanlist'
#     2) a list of frame numbers during which the LED is on IF out_form = 'framestamps'
#     """
#     vid_start_millisec = vid_start_dt.timestamp() * 1000
#     if blink_start_dt == None:
#         blink_start_millisec =  find_led_blink_start(vid_start_millisec, led_log_path)
#     else:
#         blink_start_millisec = blink_start_dt.timestamp() * 1000
#     cap = cv2.VideoCapture(video_path)
#     N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     input_cap_fps = cap.get(cv2.CAP_PROP_FPS)
#     print('FPS is: ', input_cap_fps)
#     frame_duration = (1 / input_cap_fps) * 1000#ms

#     if out_form == 'framestamps':
#         out = []
#     elif out_form == 'booleanlist':
#         out = np.zeros(N)
#     else:
#         print("Invalid out form")
#         return

#     curr_blink_start_ts = blink_start_millisec
#     blink_until = blink_start_millisec + LED_ontime
#     next_blink_start_ts = blink_start_millisec + LED_ontime + LED_offtime
#     frame_num = 0
#     inc = True
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if ret:
#             frame_ts = vid_start_millisec + (frame_num * frame_duration)#ms
            
#             if frame_ts > blink_until and inc:
#                 inc = False
#                 curr_blink_start_ts = next_blink_start_ts
#                 blink_until = curr_blink_start_ts + LED_ontime
#                 next_blink_start_ts += LED_ontime + LED_offtime
                
#             if frame_ts > curr_blink_start_ts and frame_ts < blink_until:
#                     inc = True
#                     if out_form == 'booleanlist':
#                         out[frame_num] = 1
#                     else:
#                         out.append(frame_num)
            
#             frame_num += 1
#         else:
#             break
    
#     return out

# def get_blink_list(video_path, vid_start_dt, led_log_path, LED_ontime, LED_offtime, blink_start_dt = None, epsilon=10, out_form = 'framestamps'):
#     """
#     Given a video of N frames, video start timestamp, and LED blink log, and known LED on/off times (ms),
#     return either:
#     1) a Boolean list of length N, where an entry 1 corresponds to the LED being 
#     on at that frame, 0 off at that frame IF out_form = 'booleanlist'
#     2) a list of frame numbers during which the LED is on IF out_form = 'framestamps'
#     """
#     vid_start_millisec = vid_start_dt.timestamp() * 1000
#     if blink_start_dt == None:
#         blink_start_millisec =  find_led_blink_start(vid_start_millisec, led_log_path)
#     else:
#         blink_start_millisec = blink_start_dt.timestamp() * 1000
#     cap = cv2.VideoCapture(video_path)
#     N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     input_cap_fps = cap.get(cv2.CAP_PROP_FPS)
#     print('FPS is: ', input_cap_fps)
#     frame_duration = (1 / input_cap_fps) * 1000#ms

#     if out_form == 'framestamps':
#         out = []
#     elif out_form == 'booleanlist':
#         out = np.zeros(N)
#     else:
#         print("Invalid out form")
#         return

#     first_blink_detected = False
#     next_blink_start_ts = 0
#     frame_num = 0
#     blink_until = 0
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if ret:
#             frame_ts = vid_start_millisec + (frame_num * frame_duration)#ms
            
            
#             print(frame_ts)
#             if first_blink_detected:
#                 if frame_ts > next_blink_start_ts and frame_ts < blink_until:
#                     print('yo')
#                     if out_form == 'booleanlist':
#                         out[frame_num] = 1
#                     else:
#                         out.append(frame_num)
#                 elif abs(frame_ts - next_blink_start_ts) < epsilon:
#                     print('howdy')
#                     next_blink_start_ts = frame_ts + LED_ontime + LED_offtime
#                     blink_until = frame_ts + LED_ontime
#                     print('hi', blink_until)
#                     if out_form == 'booleanlist':
#                         out[frame_num] = 1
#                     else:
#                         out.append(frame_num)
                
#             else:
#                 if abs(frame_ts - blink_start_millisec) < epsilon:
#                     first_blink_detected = True
#                     next_blink_start_ts = frame_ts + LED_ontime + LED_offtime
#                     blink_until = frame_ts + LED_ontime
#                     print('hi', frame_ts, 'hi', blink_until)
#                     if out_form == 'booleanlist':
#                         out[frame_num] = 1
#                     else:
#                         out.append(frame_num)
#             frame_num += 1
#         else:
#             break
    
#     return out

# video_path = '/Users/hadleigh/led_tests_jul4/set2/videos/100mson_500msoff_d1.MP4'
# vid_start_dt = datetime.strptime('04.07.2023 10:39:09.96', '%d.%m.%Y %H:%M:%S.%f')
# print(vid_start_dt)
# blink_start_dt = datetime.strptime('04.07.2023 10:39:10.90', '%d.%m.%Y %H:%M:%S.%f')
# led_log_path = '/Users/hadleigh/led_tests_jul4/set2/gt_blinks/gobetwino_100mson_500msoff.txt'
# test = get_blink_list(video_path, vid_start_dt, led_log_path, 100, 500, blink_start_dt=blink_start_dt, epsilon=20)
# x = [i/30 for i in range(test.shape[0])]
# plt.plot(x, test)
# plt.show()
# blink_box_size = 50


# vid_start_dt = datetime.strptime('28.06.2023 17:20:47,11',
#                            '%d.%m.%Y %H:%M:%S,%f')
# vid_start_millisec = vid_start_dt.timestamp() * 1000
# blink_start_millisec =  find_led_blink_start(vid_start_millisec, '/Users/hadleigh/gobetwinotest.txt')
# blink_on = 2000 #ms
# blink_off = 2000

# input_video_path = '/Users/hadleigh/test1_jun28.mp4'
# cap = cv2.VideoCapture(input_video_path)
# input_cap_fps = int(cap.get(cv2.CAP_PROP_FPS))
# W , H = int(cap.get(3)), int(cap.get(4)) 
# out_vid = cv2.VideoWriter('test_blink.mp4', cv2.VideoWriter_fourcc(*'mp4v'), input_cap_fps, (W, H))
        
# frame_duration = (1 / input_cap_fps) * 1000#ms

# epsilon = 10
# first_blink_detected = False
# next_blink_start_ts = 0
# frame_num = 0
# blink_until = 0
# while cap.isOpened():
#     ret, frame = cap.read()
#     if ret:
#         frame_ts = vid_start_millisec + (frame_num * frame_duration)#ms
#         frame_num += 1
#         if first_blink_detected:
#             if abs(frame_ts - next_blink_start_ts) < epsilon:
#                 next_blink_start_ts = frame_ts + blink_on + blink_off
#                 blink_until = frame_ts + blink_on
#                 #draw the blink on the frame
#                 frame = cv2.rectangle(frame, (0, H - blink_box_size), (blink_box_size, H), (0, 0, 255), -1)
#                 out_vid.write(frame)
#                 print('write 1: ', datetime.fromtimestamp(frame_ts/1000.0))
#             elif frame_ts < blink_until:
#                 frame = cv2.rectangle(frame, (0, H - blink_box_size), (blink_box_size, H), (0, 0, 255), -1)
#                 out_vid.write(frame)
#                 print('write 2: ', datetime.fromtimestamp(frame_ts/1000.0))
#             else:
#                 print('no ', datetime.fromtimestamp(frame_ts/1000.0))
#                 out_vid.write(frame)
#         else:
#             if abs(frame_ts - blink_start_millisec) < epsilon:
#                 first_blink_detected = True
#                 next_blink_start_ts = frame_ts + blink_on + blink_off
#                 blink_until = frame_ts + blink_on
#                 #draw the blink on the frame
#                 frame = cv2.rectangle(frame, (0, H - blink_box_size), (blink_box_size, H), (0, 0, 255), -1)
#                 out_vid.write(frame)
#                 print('first write: ', datetime.fromtimestamp(frame_ts/1000.0))
#                 print('now blink until ', datetime.fromtimestamp(blink_until/1000.0))
#             else:
#                 out_vid.write(frame)
#     else:
#         break

# out_vid.release()
# cap.release()
