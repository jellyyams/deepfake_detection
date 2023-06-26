from itertools import combinations
import os
import csv
import pandas as pd
import numpy as np
import statistics
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import math
import scipy.stats as stats
from mesh_data import MeshData
from mp_extractor import MPFeatureExtractor
from correlation_analyzer import CorrAnalyzer 


def get_landmarks(keywords):
    md = MeshData()
    res = []
    for k, v in md.landmarks.items():
        if any(sub in k for sub in keywords): 
            res += v
    
    return res
     

def run_extractions(video_paths, draw_all_landmarks, anchor_landmark, target_landmarks, generate_video, norm_approach, analysis_types, root_video_path, initial_detect, landmark_pairs): 
    for vid_path in video_paths: 
        input_path = root_video_path + vid_path + ".mp4"
        app = MPFeatureExtractor(
            input_path, 
            draw_all_landmarks=draw_all_landmarks, 
            initial_detect=initial_detect, 
            anchor_landmark = anchor_landmark,
            target_landmarks = target_landmarks,
            landmark_pairs = landmark_pairs,
            generate_video=generate_video, 
            norm_approach=norm_approach, 
            analysis_types=analysis_types)
        landmark_coords, landmark_data, landmark_groups = app.run_extraction()
        pathsplit = vid_path.split('/')
        write_data_into_csv(landmark_coords, landmark_data, landmark_groups, pathsplit[2])


def write_data_into_csv(landmark_coords, landmark_data, landmark_groups, filename):
    directory = "./landmark_analysis_data/" + filename + "/"

    if not os.path.exists(directory):
        os.makedirs(directory) 
    
    with open(directory + "coords" + ".csv", 'w') as f:
        writer = csv.DictWriter(f, landmark_coords.keys())
        writer.writeheader()
        writer.writerow(landmark_coords)

    with open(directory + "data" + ".csv", 'w') as f:
        writer = csv.DictWriter(f, landmark_data.keys())
        writer.writeheader()
        writer.writerow(landmark_data)
    
    with open(directory + "groups" + ".csv", 'w') as f:
        writer = csv.DictWriter(f, landmark_groups.keys())
        writer.writeheader()
        writer.writerow(landmark_groups)


def read_csv_into_dict(file_paths, fnames):
    data_dict = {}

    for f_path in file_paths:
        for fname in fnames: 

            full_path = "./landmark_analysis_data/" + f_path + fname + ".csv"
            df = pd.read_csv(full_path, header=None)
            df = df.rename({0:"Landmark", 1:"Distances"}, axis="index")
            df = df.T

            #Convert from string to list
            for index, row in df.iterrows():
                l = row["Distances"].replace("]", "").replace("[", "").split(",")
                l = [float(i) for i in l]
                df.at[index, "Distances"] = l
            
            data_dict[f_path + fname] = df
    
    return data_dict


def find_landmark_pairs(similar, different, analysis):
    top_pairs = {}
    for k, v in similar.items():

        data = read_csv_into_dict(v, analysis)
        corr_analyzer = CorrAnalyzer(v, analysis, False, data, "correlation_reports/" +k, most_similar=True, top_num = 100)
        top_pairs[k] = corr_analyzer.compare_data()

    for k, v in different.items():

        data = read_csv_into_dict(v, analysis)
        corr_analyzer = CorrAnalyzer(v, analysis, False, data, "correlation_reports/"+k, most_similar=False, top_num = 100)
        top_pairs[k] = corr_analyzer.compare_data()
    
    find_best_performing_l(top_pairs)
    


def find_best_performing_l(pairs, top_num=50):
    total_count = {}
    data = {}
    for k , v in pairs.items(): 
        for p in v: 
            if p in total_count:
                total_count[p] += 1
            else: 
                total_count[p] = 1
            
            if p in data:
                data[p].append(k)
            else:
                data[p] = [k]
    
    best_performing = sorted(total_count.items(), key=lambda x:x[1], reverse=True)

def write_final_report(best_performing, data):
    with open("report.txt", 'w') as f:
        for k, v in best_performing.items():
            f.write("\n" + k + " appeared " + v + " times in: ")
            for f in data[v]: 
                f.write(f + ", ")


def main():

    #data extraction 

    initial_detect = True
    draw_all_landmarks = True
    generate_video = False
    analysis_types = ["landmark_pairs", "landmark_to_anchor"] #vs "landmark_pairs" vs "landmark_to_anchor" vs "landmark_displacement_sig"
    anchor_landmark = 4
    # target_landmarks = [0, 287, 52, 17, 244, 464, 159, 145, 386, 374]
    key_regions = ["Outer", "Inner", "Corner", "0", "Eyebrow"]
    target_landmarks = get_landmarks(key_regions)
    anchor_pairs = []
    norm_approach = "first_upper_lower_bbox" #vs "first_quarters_bbox" vs "face_bbox" vs "none" vs "upper_lower_bbox" vs "first_face_bbox" vs "first_upper_lower_bbox"
    root_video_path = "../../../Desktop/Deepfake_Detection/Test_Videos"
    input_paths = [
        "/Krithika_Low/krithika_low_s3_v1",
        "/Krithika_Threequarter/krithika_threequarter_s3_v1", 
        "/Krithika_Front/krithika_front_s3_v1", 
        "/Kelly_Low/kelly_low_s3_v1",
        "/Kelly_Threequarter/kelly_threequarter_s3_v1", 
        "/Kelly_Front/kelly_front_s3_v1", 
        "/Krithika_Front/krithika_front_s29_v1", 
        "/Krithika_Front/krithika_front_s30_v1", 
        "/Krithika_Front/krithika_front_s27_v1", 
        "/Krithika_Front/krithika_front_s28_v1", 

        ]

    filenames = [a.split("/")[-1] for a in input_paths]
    filenames_every_pair = list(combinations(filenames, 2))
    landmark_pairs = list(combinations(target_landmarks, 2))

    #run_extractions(input_paths, draw_all_landmarks, anchor_landmark, target_landmarks, generate_video, norm_approach, analysis_types, root_video_path, initial_detect, landmark_pairs)
    

    #data analysis 
    paths_to_analyze = ["krithika_front_s2_v1/", "krithika_low_s2_v1/", "krithika_threequarter_s2_v1/"]
    look_for_sim = {
        "kelly_angles1" : ["kelly_front_s2_v1/", "kelly_low_s2_v1/", "kelly_threequarter_s2_v1/"], 
        "kelly_angles2" : ["kelly_front_s1_v1/", "kelly_low_s1_v1/", "kelly_threequarter_s1_v1/"], 
        "krithika_angles1" : ["krithika_front_s2_v1/", "krithika_low_s2_v1/", "krithika_threequarter_s2_v1/"], 
        "krithika_angles2" : ["krithika_front_s3_v1/", "krithika_low_s3_v1/", "krithika_threequarter_s3_v1/"], 
    }
    look_for_diff = {
        "identities1" : ["kelly_front_s2_v1", "krithika_front_s2_v1"], 
        "identities2" : ["kelly_low_s2_v1", "krithika_low_s2_v1"],
        "identities3" : ["kelly_threequarter_s2_v1", "krithika_threequarter_s2_v1"], 
        "identities4" : ["kelly_front_s3_v1", "krithika_front_s3_v1"],  
        "utterances1" : ["kelly_front_s28_v1", "kelly_front_s27_v1"], 
        "utterances2" : ["kelly_front_s29_v1", "kelly_front_s30_v1"], 
        "utterances3" : ["krithika_front_s28_v1", "krithika_front_s27_v1"], 
        "utterances4" : ["krithika_front_s29_v1", "krithika_front_s30_v1"], 

    }

    find_landmark_pairs(look_for_sim, look_for_diff, "groups")

    
    


if __name__ == "__main__": 
    main()