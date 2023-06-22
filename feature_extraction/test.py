# from mp import MPFeatureExtractor
import sys
sys.path.append('/home/hadleigh/deepfake_detection/common')
from stack_vids import stack_vids
from mesh_data import MeshData
from mp_extractor import MPFeatureExtractor

# initial_detect = True
# draw_all_landmarks = False


input_1 = '../../../Desktop/Deepfake_Detection/Test_Videos/Kelly_Front/kelly_front_s4_v1.mp4'

app = MPFeatureExtractor(
    input_1, 
    draw_all_landmarks=True, 
    target_landmarks=[0, 17])
    
app.run_extraction()

# md = MeshData()

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