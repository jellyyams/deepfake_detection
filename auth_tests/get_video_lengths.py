import cv2

cap = cv2.VideoCapture("../../test1_jun28.MP4")
og_vid_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
og_vid_fps = int(cap.get(cv2.CAP_PROP_FPS))

cap = cv2.VideoCapture("auth_test_output/test1_jun28/target_region50.avi")
evm_input_vid_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
evm_input_vid_fps = int(cap.get(cv2.CAP_PROP_FPS))

cap = cv2.VideoCapture("evm_output/test1_jun28/target_region50.avi")
evm_output_vid_length =  int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
evm_output_vid_fps = int(cap.get(cv2.CAP_PROP_FPS))

cap = cv2.VideoCapture("evm_output/test1_jun28/mp_target_region50_channel_vis.avi")
channel_vis_vid_length =  int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
channel_vis_vid_fps = int(cap.get(cv2.CAP_PROP_FPS))

cap = cv2.VideoCapture("evm_output/test1_jun28/reconstructed.mp4")
reconstructed_output_vid_length =  int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
reconstructed_output_vid_fps = int(cap.get(cv2.CAP_PROP_FPS))


print('Original video length: ', og_vid_length, og_vid_fps)
print('EVM input video length: ', evm_input_vid_length, evm_input_vid_fps)
print('EVM output video length: ', evm_output_vid_length, evm_output_vid_fps)
print('Channel vis video length: ', channel_vis_vid_length, channel_vis_vid_fps)
print('Reconstructed output video length: ', reconstructed_output_vid_length, reconstructed_output_vid_fps)