# from mp import MPFeatureExtractor
import sys
sys.path.append('/home/hadleigh/deepfake_detection/common')
from stack_vids import stack_vids
from mesh_data import MeshData
from mp_extractor import MPFeatureExtractor

# initial_detect = True
# draw_all_landmarks = False


# input_1 = '../../../Desktop/Deepfake_Detection/Test_Videos/hadleigh_Front/kelly_front_s4_v1.mp4'

# app = MPFeatureExtractor(
#     input_1, 
#     draw_all_landmarks=True, 
#     target_landmarks=[0, 17])
    
# app.run_extraction()

md = MeshData()

# nums = [270, 317, 375, 409, 312, 375, 311, 270, 321, 13, 321, 310, 375, 405, 402, 311, 318, 80, 181, 146, 191, 312, 321, 181, 312, 311, 321, 80, 91, 13, 405, 312, 402, 81, 181, 81, 91, 312, 405, 84, 312, 311, 402, 82, 181, 13, 181, 317, 312, 317, 81, 178, 13, 14, 82, 87, 40, 88, 146, 185, 80, 146, 40, 91, 39, 88, 91, 191, 91, 185, 39, 91, 95, 185, 181, 311]

# for n in nums:
#     for k, v in md.landmarks.items():
#         if n in v:
#             print(str(n) + " is in " + str(k))
#             break

trial_name = "pairs_hadleigh_s31_approach2"

stacked_out_name = 'mp_{}.mp4'.format(trial_name)

stacked_out_name = 'stacked_output/stacked__{}'.format(stacked_out_name)

p1_out_name = 'hadleigh_low_s31_v2.mp4'
p2_out_name = 'hadleigh_left_threequarter_s31_v2.mp4'
p3_out_name = 'hadleigh_right_threequarter_s31_v2.mp4'

stack_vids('mp_output_videos/{}'.format(p1_out_name),'mp_output_videos/{}'.format(p2_out_name), stacked_out_name, vid3_path = 'mp_output_videos/{}'.format(p3_out_name), vertical=True, out_fps = 30)





# allnum = set(list(range(0, 478)))

# md_set = set(md.upper_left_landmarks + md.upper_right_landmarks + md.lower_right_landmarks + md.lower_left_landmarks) 
# missing = allnum - md_set


# print(missing)
# print(len(allnum) - len(md_set))
# print(len(missing))
# print(len(allnum-md_set))
# print(len(md_set))
# print(len(list(md_set)))
# print(len(md.upper_right_landmarks) + len(md.upper_left_landmarks) + len(md.lower_right_landmarks) + len(md.lower_left_landmarks)) 

# all = md.upper_left_landmarks + md.upper_right_landmarks + md.lower_left_landmarks + md.lower_right_landmarks
# all.sort()
# print(len(all))
# print(len(list(md_set)))

# l = []
# print("duplicate numbers")
# for n in all:

#     if n not in l:
#         l.append(n)
#     else:
#         print(n)

# print("missing numbers")
# for n in allnum: 
#     if n not in all:
#         print(n)