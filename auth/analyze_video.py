import cv2

input_video_path = 'auth_test_output/hadleigh1_pattern_4Hz_alpha0.5_region50.mp4'
try:
    input_capture = cv2.VideoCapture(input_video_path)
except:
    raise ValueError('Input video path %s not valid' % input_video_path)

if input_capture is not None:    
    W , H = int(input_capture.get(3)), int(input_capture.get(4)) #input video dimensions
else:
    raise ValueError("Invalid input video")

input_cap_fps = int(input_capture.get(cv2.CAP_PROP_FPS))

