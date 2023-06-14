from mp import MPFeatureExtractor
import sys
sys.path.append('/home/hadleigh/deepfake_detection/common')
from stack_vids import stack_vids

initial_detect = True
draw_all_landmarks = False


input_1 = '../../../Desktop/Deepfake_Detection/Test_Videos/Kelly_Front/kelly_front_s4_v1.mp4'

app = MPFeatureExtractor(input_1, 'mp_output', draw_all_landmarks=draw_all_landmarks, initial_detect=initial_detect)
p1_out_name = app.get_output_video_path()
app.run_extraction()
