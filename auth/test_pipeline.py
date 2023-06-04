from mp_pattern import MPPatternApp
from mp_auth import MPAuthApp
import os
import cv2
import matlab.engine
from reconstruct_vid import reconstruct_vid
import sys
sys.path.append('/home/hadleigh/df_pipeline/common')
from stack_vids import stack_vids
import re
import glob
from vis_channels import channel_visualizer

target_landmarks = [50, 280, 109, 338]
alpha = 0.97
track_channels = True
initial_detect=True
pattern_rel_width = 20
pattern_color = (0, 255, 0)
pattern_out_width = 20

input_video_path = '../../inputs_may18/9_12_comp.mp4'
dummy_input_capture = cv2.VideoCapture(input_video_path)
input_cap_fps = int(dummy_input_capture.get(cv2.CAP_PROP_FPS))

#add imperceptible pattern at target landmarks
pattern_app = MPPatternApp(input_video_path, alpha=alpha, initial_detect=initial_detect, target_landmarks=target_landmarks, pattern_rel_width=pattern_rel_width, pattern_color=pattern_color)
pattern_app.run()
pattern_output_vid_path = pattern_app.get_output_vid_path()
ref_pattern_output_vid_path = re.sub(r'(alpha\d{1}(.?[0123456789]{1,})?)', 'ref', pattern_output_vid_path)
print('regex test ', ref_pattern_output_vid_path)

#run auth to extract patterns
auth_app =  MPAuthApp(pattern_output_vid_path, initial_detect=initial_detect, target_landmarks=target_landmarks, pattern_rel_width=pattern_rel_width, pattern_out_width=pattern_out_width, track_channels=track_channels)
auth_app.run()
auth_output_dir_path = auth_app.get_auth_output_dir()

#run evm on extracted patterns
eng = matlab.engine.start_matlab()
eng.addpath('/home/hadleigh/vidmag/')
eng.setPath_hadleigh(nargout=0)

#create appropriate output dir in evm output
auth_output_dir_name = auth_output_dir_path.split('/')[-1]
evm_output_dir = 'evm_output/{}/'.format(auth_output_dir_name)
print(evm_output_dir)
os.makedirs(evm_output_dir, exist_ok=False)

for target_lm in target_landmarks:
    input_video = 'target_region{}.mp4'.format(target_lm)
    print('EVM on {}'.format(input_video))
    print(auth_output_dir_path)
    #resultsDir, dataDir, inFileName, alpha, level, fl, fh, samplingRate, chromAttenuation
    eng.hadleigh_vidmag(evm_output_dir, auth_output_dir_path  + '/', input_video, 60.0, 3.0, 3.9, 4.1, float(input_cap_fps), 1.0, nargout=0)


#Run channel visualization on EVM'd videos
evm_video_paths = glob.glob(evm_output_dir + '*')
for target_lm in target_landmarks:
    target_region_vid_path = list(filter(lambda x: f'target_region{target_lm}' in x, evm_video_paths))[0]
    print('Channel vis on {}'.format(target_region_vid_path))
    chan_vis = channel_visualizer(target_region_vid_path, evm_output_dir) 
    chan_vis.run()

# evm_output_dir = 'evm_output/9_12_comp_pattern_b0_g255_r0_4Hz_alpha0.97_sz20/'
# auth_output_dir_path = 'auth_test_output/9_12_comp_pattern_b0_g255_r0_4Hz_alpha0.97_sz20'
# pattern_output_vid_path = 'pattern_output/9_12_comp_pattern_b0_g255_r0_4Hz_alpha0.97_sz20.mp4'
# ref_pattern_output_vid_path =  'pattern_output/9_12_comp_pattern_b0_g255_r0_4Hz_ref_sz20.mp4'

#reconstruct to visualize
reconstruct_vid(input_video_path, evm_output_dir, auth_output_dir_path)

#input and evm output side by side
print(ref_pattern_output_vid_path)
print(pattern_output_vid_path)
stack_vids(pattern_output_vid_path, ref_pattern_output_vid_path, evm_output_dir + 'stacked.mp4',  vid3_path = evm_output_dir + 'reconstructed.mp4', out_fps=input_cap_fps)
