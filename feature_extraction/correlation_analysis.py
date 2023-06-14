from mp import MPFeatureExtractor
from stack_vids import stack_vids
import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import math
import scipy.stats as stats

initial_detect = True
draw_all_landmarks = False
generate_video = True
anchor_landmark = 4
target_landmarks = [0, 287, 52, 17, 244, 464, 159, 145, 386, 374]
normalize_by = "first_bbox" #vs "face_bbox" vs "none" vs "region_bbox" vs "first_bbox"
analysis_type = "landmarks_to_anchor"
root_video_path = "../../../Desktop/Deepfake_Detection/Test_Videos"
input_vids = ["/Kelly_Front/kelly_front_s1_v1", "/Kelly_Low/kelly_low_s1_v1", "/Kelly_Profile/kelly_profile_s2_v1", "/Kelly_Threequarter/kelly_threequarter_s1_v1"]
data_dict = {}


def run_extractions(video_paths): 
    for vid_path in video_paths: 
        input_path = root_video_path + vid_path + ".mp4"
        app = MPFeatureExtractor(
            input_path, 
            'mp_output', 
            draw_all_landmarks=draw_all_landmarks, 
            initial_detect=initial_detect, 
            anchor_landmark = anchor_landmark,
            target_landmarks = target_landmarks,
            generate_video=generate_video, 
            norm_approach=normalize_by)
        extraction_data = app.run_extraction()
        pathsplit = vid_path.split('/')
        with open("./extracted_data/" + pathsplit[2] + ".csv", 'w') as f:
            writer = csv.DictWriter(f, extraction_data.keys())
            writer.writeheader()
            writer.writerow(extraction_data)

def read_csv_into_dict(file_paths):
    for f_path in file_paths:
        full_path = "./extracted_data/" + f_path + ".csv"
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
        for key, value in r_and_p.items():
            s += float(value[0])
            f.write("\nLandmark " + key + " r value: " + value[0] + ", p value: " + value[1])
        f.write ("\n\n Average Pearson R value: " + str(round(s/len(r_and_p), 4)))


def plot_landmarks_corr(file_path1, file_path2):

    df_1 = data_dict[file_path1]
    df_2 = data_dict[file_path2]
    pearson_r_and_p = {}

     # Set window size to compute moving window synchrony
    r_window_size = 25

    output_directory = "analysis_results/" + file_path1 + "_compare_" + file_path2 + "_normby" + normalize_by + "_windowsize" + str(r_window_size) + "_analysistype" + analysis_type + "/"

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    
    for index, row in df_1.iterrows():
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
        df.rolling(window=r_window_size,center=True).median().plot(ax=ax[0])
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


pair_to_analyze = ["kelly_front_s1_v1", "kelly_low_s1_v1"]

# run_extractions(input_vids)
read_csv_into_dict(pair_to_analyze)
plot_landmarks_corr(pair_to_analyze[0], pair_to_analyze[1])

