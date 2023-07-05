from mp_auth import MPAuthApp
import os
import cv2
from reconstruct_vid import reconstruct_vid
import sys
sys.path.append('/home/hadleigh/df_pipeline/common')
from stack_vids import stack_vids
import re
import glob
from vis_channels import channel_visualizer
from vis_channels_pixelwise import channel_visualizer as pixelwise_channel_visualizer

target_landmarks = [50, 280, 109, 338]
track_channels = False
colorspace = 'bgr'
initial_detect=True
pattern_rel_width = 20
pattern_color = (0, 255, 0)
pattern_out_width = 10

#evm parameters
level = 2
alpha = 60
low = 0.05
high = 5

for input_video_path  in ['../../led_tests_jul4/set2/videos/100mson_500msoff_d1.MP4']: #['../../test1_jun28.MP4', '../../test1_nothing_jun28.MP4', '../../test2_jun28.MP4']:
    dummy_input_capture = cv2.VideoCapture(input_video_path) #just to get fps to know for output vids
    input_cap_fps = int(dummy_input_capture.get(cv2.CAP_PROP_FPS))

    #run auth to extract patterns from led vids
    try:
        auth_app = MPAuthApp(input_video_path, initial_detect=initial_detect, target_landmarks=target_landmarks, pattern_rel_width=pattern_rel_width, pattern_out_width=pattern_out_width, track_channels=track_channels, colorspace=colorspace)
    except:
        sys.exit()

    auth_app.run()
    auth_output_dir_path = auth_app.get_auth_output_dir()

    # auth_output_dir_path = f"auth_test_output/{input_video_path.split('/')[-1].split('.MP4')[0]}_target6"
    # print(auth_output_dir_path)
   
    for target_lm in target_landmarks:
        target_region_vid_path = f'{auth_output_dir_path}/target_region{target_lm}.avi'
        print('Channel vis on {}'.format(target_region_vid_path))
        chan_vis = channel_visualizer(target_region_vid_path, auth_output_dir_path, colorspace = colorspace) 
        chan_vis.run()


#create appropriate output dir in evm output
# auth_output_dir_name = auth_output_dir_path.split('/')[-1]
# evm_output_dir_path = 'evm_output/{}/'.format(auth_output_dir_name)
# print(evm_output_dir_path)
# if os.path.exists(evm_output_dir_path):
#     inp = input(f'The directory {evm_output_dir_path} exists. Enter y to overwrite it, or n to terminate video authentication. : ')
#     if inp != 'y':
#         print(f'Terminating processing. The directory  {evm_output_dir_path}  will not be overwritten.')
#         sys.exit()
# os.makedirs(evm_output_dir_path, exist_ok=True)
# #run evm on extracted patterns
# for t in target_landmarks:
#     evm_input_video_path = f'{auth_output_dir_path}/target_region{t}.avi'
#     print(f'EVM on {input_video_path}')
#     command = f'python Eulerian-Video-Magnification/src/evm.py --video_path {evm_input_video_path}  --level {level} --alpha {alpha} --lambda_cutoff 0 --low_omega {low} --high_omega {high} --saving_path {evm_output_dir_path}target_region{t}.avi'
#     os.system(command)

#Run channel visualization on EVM'd videos
# evm_video_paths = glob.glob(evm_output_dir_path + '*')
# for target_lm in target_landmarks:
#     target_region_vid_path = list(filter(lambda x: f'target_region{target_lm}' in x, evm_video_paths))[0]
#     print('Channel vis on {}'.format(target_region_vid_path))
#     chan_vis = channel_visualizer(target_region_vid_path, evm_output_dir_path) 
#     chan_vis.run()

# #reconstruct to visualize
#reconstruct_vid(input_video_path, evm_output_dir_path, auth_output_dir_path)

