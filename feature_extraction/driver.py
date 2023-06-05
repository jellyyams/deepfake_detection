from mp import MPFeatureExtractor
#from fan import FANApp
import sys
sys.path.append('/home/hadleigh/deepfake_detection/common')
from stack_vids import stack_vids


#['1_4', '1_4_take2', '5_8', '9_12', '13_16', '17_20']
# for trial in ['1_4']:

initial_detect = True

trial_name = '5_8'
input_p1 = '../../inputs/hadleigh1.mp4'
input_p2 = '../../inputs_may18/1_4_phone.mp4'

app = MPFeatureExtractor(input_p1, draw_all_landmarks=True, initial_detect=initial_detect)
p1_out_name = app.get_output_video_path()
app.run_extraction()
# app = MPFeatureExtractor(input_p2, draw_all_landmarks=True, initial_detect=initial_detect)
# app.run_extraction()
# p2_out_name = app.get_output_vid_name()

# # p1_out_name = 'initdet_mp_5_8_comp.mp4'
# # p2_out_name = 'initdet_mp_5_8_phone.mp4'

# stacked_out_name = 'mp_{}.mp4'.format(trial_name)
# if initial_detect:
#     stacked_out_name = 'stacked_output/stacked_initdet_{}'.format(stacked_out_name)
# else:
#     stacked_out_name = 'stacked_output/stacked__{}'.format(stacked_out_name)

# stack_vids('mp_output/{}'.format(p1_out_name),'mp_output/{}'.format(p2_out_name), stacked_out_name, vertical=True, out_fps = 30)


