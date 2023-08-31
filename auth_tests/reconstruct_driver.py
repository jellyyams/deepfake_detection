from reconstruct_vid import reconstruct_vid
from vis_channels import channel_visualizer
import sys
sys.path.append('/home/hadleigh/deepfake_detection/common')
from stack_vids import stack_vids
import glob

target_landmarks = [50, 280, 109, 338]
evm_output_dir = 'evm_output/test1_jun28/'
# reconstruct_vid('../../test1_jun28.MP4', evm_output_dir, 'auth_test_output/test1_jun28')

#Run channel visualization on EVM'd videos
evm_video_paths = glob.glob(evm_output_dir + '*')
for target_lm in target_landmarks:
    target_region_vid_path = list(filter(lambda x: f'target_region{target_lm}' in x, evm_video_paths))[0]
    print('Channel vis on {}'.format(target_region_vid_path))
    chan_vis = channel_visualizer(target_region_vid_path, evm_output_dir) 
    chan_vis.run()


stack_vids('evm_output/test1_jun28/reconstructed.mp4', 'evm_output/test1_jun28/mp_target_region109_channel_vis.avi', 'stacked_channel_vis.mp4', vid3_path=None, out_fps=29, vertical=False)