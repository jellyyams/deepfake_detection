from mp_extractor import MPFeatureExtractor
from stack_vids import stack_vids
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


def get_landmarks(keywords):
    md = MeshData()
    res = []
    for k, v in md.landmarks.items():
        if any(sub in k for sub in keywords): 
            res += v
    
    return res
     

def run_extractions(video_paths, draw_all_landmarks, anchor_landmark, target_landmarks, generate_video, norm_approach, analysis_type, root_video_path, initial_detect): 
    for vid_path in video_paths: 
        input_path = root_video_path + vid_path + ".mp4"
        app = MPFeatureExtractor(
            input_path, 
            draw_all_landmarks=draw_all_landmarks, 
            initial_detect=initial_detect, 
            anchor_landmark = anchor_landmark,
            target_landmarks = target_landmarks,
            generate_video=generate_video, 
            norm_approach=norm_approach, 
            analysis_type=analysis_type)
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

def read_csv_into_dict(file_paths, data_dict, fname):
    for f_path in file_paths:
        full_path = "./correlation_reports/" + f_path + fname + ".csv"
        df = pd.read_csv(full_path, header=None)
        df = df.rename({0:"Landmark", 1:"Distances"}, axis="index")
        df = df.T

        #Convert from string to list
        for index, row in df.iterrows():
            l = row["Distances"].replace("]", "").replace("[", "").split(",")
            l = [float(i) for i in l]
            df.at[index, "Distances"] = l
        
        data_dict[f_path] = df

def write_analysis_report(output_dir, file_path1, file_path2, r_and_p, r_window_size):
    with open(output_dir + "report.txt", "w") as f:
        f.write("Comparing " + file_path1 + " with " + file_path2)
        f.write("\nNormalized by " + normalize_by.replace("_"," "))
        f.write("\nWindow size: " + str(r_window_size))
        f.write("\nAnalysis type: " + analysis_type)
        f.write("\n")
        s = 0
        rvalues = []
        for key, value in r_and_p.items():
            s += float(value[0])
            rvalues.append(float(value[0]))
            f.write("\nLandmark " + key + " r value: " + value[0] + ", p value: " + value[1])

        f.write ("\n\n Median Pearson R value: " + str(round(statistics.median(rvalues), 4)))
        f.write ("\n Average Pearson R value: " + str(round(s/len(r_and_p), 4)))

def plot_landmarks_corr(file_path1, file_path2, data_dict):
    df_1 = data_dict[file_path1]
    df_2 = data_dict[file_path2]

    pearson_r_and_p = {}

     # Set window size to compute moving window synchrony
    r_window_size = 25

    output_directory = "analysis_results/" + file_path1 + "_compare_" + file_path2 + "_normby" + normalize_by + "_windowsize" + str(r_window_size) + "_analysistype" + analysis_type + "/"

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    
    for index, row in df_1.iterrows():
        if len(row["Distances"]) > len(df_2.at[index, "Distances"]):
            row["Distances"] = row["Distances"][:len(df_2.at[index, "Distances"])]
        else:
            df_2.at[index, "Distances"] = df_2.at[index, "Distances"][:len(row["Distances"])]
        

        df = pd.DataFrame(data = {file_path1 : row["Distances"], file_path2: df_2.at[index, "Distances"]})
   
        r_and_p = stats.pearsonr(row["Distances"], df_2.at[index, "Distances"])

        r_and_p_str = str(r_and_p)
        print(r_and_p_str)
        r_and_p_l = r_and_p_str.replace("PearsonRResult(statistic=","").replace(" pvalue=","").replace(")","").split(",")

        pearson_r_and_p[row["Landmark"]] = r_and_p_l
   
        # Interpolate missing data.
        df_interpolated = df.interpolate()
        # Compute rolling window synchrony
        rolling_r = df_interpolated[file_path1].rolling(window=r_window_size, center=True).corr(df_interpolated[file_path2])
        f, ax =plt.subplots(2,1,figsize=(14,6),sharex=True)

        ax[0].set(xlabel='Frame',ylabel='Distance from anchor point')
      
        df_interpolated.plot(ax=ax[0])
        rolling_r.plot(ax=ax[1])
        ax[1].set(xlabel='Frame',ylabel='Pearson r')
        
        plt.suptitle("Distance data and rolling window correlation \n" + 
            "landmark number: " + 
            row["Landmark"] + 
            ", rolling window: " + 
            str(r_window_size) + 
            " frames, Pearson value: " + 
            str(round(float(r_and_p_l[0]), 5)))

        plt.savefig(output_directory + file_path1 + "_compare_" + file_path2 + "_landmark_" + row["Landmark"])
    
    write_analysis_report(output_directory, file_path1, file_path2, pearson_r_and_p, r_window_size)

def main():

    initial_detect = True
    draw_all_landmarks = True
    generate_video = False
    analysis_type = "landmark_pairs" #vs "landmark_pairs" vs "landmark_to_anchor" vs "landmark_displacement_sig"
    anchor_landmark = 4
    # target_landmarks = [0, 287, 52, 17, 244, 464, 159, 145, 386, 374]
    key_regions = ["Outer", "Inner", "Corner", "0", "Eyebrow"]
    target_landmarks = get_landmarks(key_regions)
    anchor_pairs = []
    norm_approach = "first_upper_lower_bbox" #vs "first_quarters_bbox" vs "face_bbox" vs "none" vs "upper_lower_bbox" vs "first_face_bbox" vs "first_upper_lower_bbox"
    root_video_path = "../../../Desktop/Deepfake_Detection/Test_Videos"
    input_paths = ["/Kelly_Front/kelly_front_s1_v1", "/Kelly_Low/kelly_low_s1_v1", "/Kelly_Threequarter/kelly_threequarter_s1_v1"]
    data_dict = {}

    filenames = [a.split("/")[-1] for a in input_paths]

    filenames_every_pair = list(combinations(filenames, 2))

    landmark_pairs = list(combinations(target_landmarks, 2))
    print(len(target_landmarks))
    print(len(landmark_pairs))
    

    run_extractions(input_paths, draw_all_landmarks, anchor_landmark, target_landmarks, generate_video, norm_approach, analysis_type, root_video_path, initial_detect)
    # read_csv_into_dict(pair_to_analyze, data_dict)
    # plot_landmarks_corr(pair_to_analyze[0], pair_to_analyze[1], data_dict)

if __name__ == "__main__": 
    main()