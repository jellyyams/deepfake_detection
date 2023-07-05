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

def generate_landmark_pairs(landmarks):
    c = list(combinations(landmarks, 2))
    sorted_list = []
    for pair in c: 
        if pair[0] > pair[1]:
            sorted_list.append((pair[1], pair[0]))
        else:
            sorted_list.append(pair)
    
    return sorted_list
    
def min_max_norm(distances): 
    smallest = min(distances)
    largest = max(distances)
    d_scaled = [(x - smallest)/(largest - smallest) for x in distances]

    return d_scaled


def run_extractions(video_paths, draw_all_landmarks, anchor_landmark, target_landmarks, generate_video, norm_approach, analysis_types, root_video_path, initial_detect, landmark_pairs, target_pairs): 

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
            analysis_types=analysis_types, 
            target_pairs= target_pairs)
        landmark_coords, landmark_data, landmark_groups = app.run_extraction()
        pathsplit = vid_path.split('/')
        write_data_into_csv(landmark_coords, "coords", pathsplit[2])
        write_data_into_csv(landmark_data, "data", pathsplit[2])
        write_data_into_csv(landmark_groups, "groups", pathsplit[2])


def write_data_into_csv(data, name_suff, filename):
    directory = "./landmark_analysis_data/" + str(filename) + "/"

    if not os.path.exists(directory):
        os.makedirs(directory) 
    
    with open(directory + name_suff + ".csv", 'w') as f:
        writer = csv.DictWriter(f, data.keys())
        writer.writeheader()
        writer.writerow(data)


def read_csv_into_dict(file_paths, fnames):
    data_dict = {}

    for f_path in file_paths:
        for fname in fnames: 

            full_path = "./landmark_analysis_data/" + f_path + fname + ".csv"
            df = pd.read_csv(full_path, header=None)
            df = df.rename({0:"Landmark_key", 1:"Distances"}, axis="index")
            df = df.T
           
            distances = []
            normalized = []

            #Convert from string to list and add normalized column
            for index, row in df.iterrows():
                l = row["Distances"].replace("]", "").replace("[", "").split(",")
                l = [float(i) for i in l]
                distances.append(l)
                normalized.append(min_max_norm(l))
         
            df["Distances"] = distances
            df["Normalized"] = normalized
            
            data_dict[f_path] = df
    
    return data_dict


def find_sim_landmark_pairs(similar, analysis, top_cutoff, top_num):
    top_sim_pairs = {}
    for k, v in similar.items():

        data = read_csv_into_dict(v, analysis)
        corr_analyzer = CorrAnalyzer(v, analysis, False, data, "correlation_reports/", k, most_similar=True, cutoff = top_cutoff)
        newpairs = corr_analyzer.compare_data()
        add_to_final_dict(top_sim_pairs, newpairs, k)
        
    write_data_into_csv(top_sim_pairs, "similar_pairs", top_cutoff)
    best_performing_sim, total_count, total_score = find_best_performing(top_sim_pairs, True)
    write_report(best_performing_sim, top_sim_pairs, total_count, "sim_", top_num)
    best_sim_pairs = [x for x, y in best_performing_sim]
    return best_sim_pairs[:top_num], top_sim_pairs, best_performing_sim, total_count, total_score

def find_diff_landmark_pairs(different, analysis, bottom_cutoff, top_num):
    top_diff_pairs = {}
    for k, v in different.items():

        data = read_csv_into_dict(v, analysis)
        corr_analyzer = CorrAnalyzer(v, analysis, False, data, "correlation_reports/", k, most_similar=False, cutoff=bottom_cutoff)
        newpairs = corr_analyzer.compare_data()
        add_to_final_dict(top_diff_pairs, newpairs, k)

    write_data_into_csv(top_diff_pairs, "different_pairs", bottom_cutoff)
    best_performing_diff, total_count, total_score = find_best_performing(top_diff_pairs, False)
    write_report(best_performing_diff, top_diff_pairs, total_count, "diff_", top_num)
    best_diff_pairs = [x for x, y in best_performing_diff]
    
    return best_diff_pairs[:top_num], top_diff_pairs, best_performing_diff, total_count, total_score
    
def find_top_diff_from_sim(best_sim_list, top_diff_pairs, top_num):
    overall = {k: v for k, v in top_diff_pairs.items() if k in best_sim_list}
    best_performing_diff, total_count, total_score = find_best_performing(overall, False)
    write_report(best_performing_diff, overall, total_count, "sim_then_diff_", top_num)

def find_intersection(s_list, d_list, sorted_sim_avg, sorted_diff_avg, total_sim, total_diff):
    s_set = set(s_list)
    d_set = set(d_list)
    intersect = d_set.intersection(s_set)
    with open("intersect_report.txt", "w") as f:
        for s in intersect: 
            s_avg = sorted_sim_avg[s] / total_sim[s]
            d_avg = sorted_diff_avg[s] / total_diff[s]
            f.write(s + " similarity avg pearson: " + str(s_avg) + " difference avg pearson: " + str(d_avg) + "\n")

def add_to_final_dict(d, newpairs, testcase):
 
    for k, v in newpairs.items():
        if k in d: 
            d[k][testcase] = v
        else: 
            d[k] = {testcase : v}


def find_best_performing(sim_pairs, reverse):
    #only count as best performing if the landmark pair shows up in both similar and different tests
    total_count = {}
    total_score = {}
    data = {}
    for pair , v in sim_pairs.items(): 
        for testcase, avg in v.items(): 
            if pair in total_count:
                total_count[pair] += 1
                total_score[pair] += abs(avg)

            else: 
                total_count[pair] = 1
                total_score[pair] = avg
    
    best_performing = sorted(total_score.items(), key=lambda x:x[1], reverse=reverse)
    return best_performing, total_count, total_score 

def write_report(best_performing, data, total_count, pref, top_num):

    with open(pref + "report.txt", 'w') as f:
        for i in range(top_num): 
            pair_and_score = best_performing[i]
            count = total_count[str(pair_and_score[0])]
      
            f.write("\n" + str(pair_and_score[0]) + " appeared " + str(count) + " times with an average score of " + str(float(pair_and_score[1])/count) + " in: ")
            for testcase, pearsonr in data[pair_and_score[0]].items(): 
                f.write(str(testcase) + "(r value: " + str(pearsonr) + ")")

def plot_pairs(pairs, vid_names, data):
    for pair in pairs: 
        plt.clf()
        f, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        ax[0].set(xlabel="Frames", ylabel="Distances between landmarks", ylim=(0, 0.5))
        ax[1].set(xlabel="Frames", ylabel="Min max normalized distances", ylim=(0,1))
        for vid in vid_names: 
            df = data[vid]
            index = list(np.where(df["Landmark_key"] == str(pair)))[0][0]

            ax[0].plot(df.loc[index, "Distances"], label=vid)
            ax[1].plot(df.loc[index, "Normalized"], label=vid)
        
        f.tight_layout(pad=3.0)
        plt.legend(bbox_to_anchor=(1.0, 2),loc="upper left")
        plt.suptitle("Raw and normalized distance data for pair: " + str(pair))
        plt.savefig("plots/" + str(pair), bbox_inches='tight')

def make_plots(pairs, vid_names):
    data = read_csv_into_dict(vid_names, ["groups"])
    plot_pairs(pairs, vid_names, data)


def main():

    #data extraction 

    initial_detect = True
    draw_all_landmarks = True
    generate_video = True
    analysis_types = ["landmark_pairs", "landmark_to_anchor"] #vs "landmark_pairs" vs "landmark_to_anchor" vs "landmark_displacement_sig"
    anchor_landmark = 57
    # target_landmarks = [57, 0, 267, 269, 270, 409, 306, 292, 37, 308, 291]
    key_regions = ["Outer", "Inner", "Corner", "0", "Eyebrow"]
    target_landmarks = get_landmarks(key_regions)
    target_pairs = [
        (13, 321), 
        (375, 409), 
        (311, 375), 
        (312, 321), 
        (312, 375), 
        (270, 321), 
        (311, 321), 
        (310, 405), 
        (13, 405), 
        (270, 375), 
        (81, 181), 
        (311, 314)
    ]
    norm_approach = "first_upper_lower_bbox" #vs "first_quarters_bbox" vs "face_bbox" vs "none" vs "upper_lower_bbox" vs "first_face_bbox" vs "first_upper_lower_bbox"
    root_video_path = "../../../Desktop/Deepfake_Detection/Test_Videos"
    input_paths = [
        "/Hadleigh_Low/hadleigh_low_s31_v2",
        "/Hadleigh_Right_Threequarter/hadleigh_right_threequarter_s31_v2", 
        "/Hadleigh_Left_Threequarter/hadleigh_left_threequarter_s31_v2",
        "/Hadleigh_Front/hadleigh_front_s31_v2", 
        # "/Hadleigh_Low/hadleigh_low_s32_v2",
        # "/Hadleigh_Right_Threequarter/hadleigh_right_threequarter_s32_v2", 
        # "/Hadleigh_Left_Threequarter/hadleigh_left_threequarter_s32_v2",
        # "/Hadleigh_Front/hadleigh_front_s32_v2", 
        # "/Hadleigh_Low/hadleigh_low_s27_v2",
        # "/Hadleigh_Right_Threequarter/hadleigh_right_threequarter_s27_v2", 
        # "/Hadleigh_Left_Threequarter/hadleigh_left_threequarter_s27_v2",
        # "/Hadleigh_Front/hadleigh_front_s27_v2", 
        # "/Hadleigh_Low/hadleigh_low_s28_v2",
        # "/Hadleigh_Right_Threequarter/hadleigh_right_threequarter_s28_v2", 
        # "/Hadleigh_Left_Threequarter/hadleigh_left_threequarter_s28_v2",
        # "/Hadleigh_Front/hadleigh_front_s28_v2", 
        "/Kelly_Low/kelly_low_s31_v2",
        "/Kelly_Right_Threequarter/kelly_right_threequarter_s31_v2", 
        "/Kelly_Left_Threequarter/kelly_left_threequarter_s31_v2",
        "/Kelly_Front/kelly_front_s31_v2", 
        # "/Kelly_Low/kelly_low_s32_v2",
        # "/Kelly_Right_Threequarter/kelly_right_threequarter_s32_v2", 
        # "/Kelly_Left_Threequarter/kelly_left_threequarter_s32_v2",
        # "/Kelly_Front/kelly_front_s32_v2", 
        # "/Kelly_Low/kelly_low_s27_v2",
        # "/Kelly_Right_Threequarter/kelly_right_threequarter_s27_v2", 
        # "/Kelly_Left_Threequarter/kelly_left_threequarter_s27_v2",
        # "/Kelly_Front/kelly_front_s27_v2", 
        # "/Kelly_Low/kelly_low_s28_v2",
        # "/Kelly_Right_Threequarter/kelly_right_threequarter_s28_v2", 
        # "/Kelly_Left_Threequarter/kelly_left_threequarter_s28_v2",
        # "/Kelly_Front/kelly_front_s28_v2", 
        ]

    filenames = [a.split("/")[-1] for a in input_paths]
    filenames_every_pair = list(combinations(filenames, 2))
    landmark_pairs = generate_landmark_pairs(target_landmarks)
    print(len(target_landmarks))
    print(len(landmark_pairs))

    run_extractions(
        input_paths, 
        draw_all_landmarks, 
        anchor_landmark, 
        target_landmarks, 
        generate_video, 
        norm_approach, 
        analysis_types, 
        root_video_path, 
        initial_detect, 
        landmark_pairs, 
        target_pairs)

    

    #data analysis 
    look_for_sim = {
        "kelly_angles1" : ["kelly_front_s1_v2/", "kelly_low_s1_v2/", "kelly_right_threequarter_s1_v2/", "kelly_left_threequarter_s1_v2/"], 
        "kelly_angles2" : ["kelly_front_s2_v2/", "kelly_low_s2_v2/", "kelly_right_threequarter_s2_v2/", "kelly_left_threequarter_s2_v2/"], 
        "kelly_angles3" : ["kelly_front_s3_v2/", "kelly_low_s3_v2/", "kelly_right_threequarter_s3_v2/", "kelly_left_threequarter_s3_v2/"], 
        "kelly_angles4" : ["kelly_front_s4_v2/", "kelly_low_s4_v2/", "kelly_right_threequarter_s4_v2/", "kelly_left_threequarter_s4_v2/"], 
        "hadleigh_angles1" : ["hadleigh_front_s1_v2/", "hadleigh_low_s1_v2/", "hadleigh_right_threequarter_s1_v2/", "hadleigh_left_threequarter_s1_v2/"], 
        "hadleigh_angles2" : ["hadleigh_front_s2_v2/", "hadleigh_low_s2_v2/", "hadleigh_right_threequarter_s2_v2/", "hadleigh_left_threequarter_s2_v2/"], 
        "hadleigh_angles3" : ["hadleigh_front_s3_v2/", "hadleigh_low_s3_v2/", "hadleigh_right_threequarter_s3_v2/", "hadleigh_left_threequarter_s3_v2/"], 
        "hadleigh_angles4" : ["hadleigh_front_s4_v2/", "hadleigh_low_s4_v2/", "hadleigh_right_threequarter_s4_v2/", "hadleigh_left_threequarter_s4_v2/"], 
    }
    look_for_diff = {
        "identities1" : ["kelly_front_s1_v2/", "hadleigh_front_s1_v2/"], 
        "identities2" : ["kelly_front_s2_v2/", "hadleigh_front_s2_v2/"], 
        "identities3" : ["kelly_front_s3_v2/", "hadleigh_front_s3_v2/"], 
        "identities4" : ["kelly_front_s4_v2/", "hadleigh_front_s4_v2/"], 
        "utterances1" : ["kelly_front_s28_v2/", "kelly_front_s27_v2/"], 
        "utterances2" : ["hadleigh_front_s28_v2/", "hadleigh_front_s27_v2/"], 
        "utterances3" : ["kelly_front_s31_v2/", "kelly_front_s32_v2/"]
    }

    top_num = 150

    # best_sim_list, best_sim_pairs, sorted_sim_avg, sim_total_count, sim_total_score = find_sim_landmark_pairs(look_for_sim, ["groups"], top_cutoff=-1, top_num=top_num)
    # best_diff_list, best_diff_pairs, sorted_diff_avg, diff_total_count, diff_total_score = find_diff_landmark_pairs(look_for_diff, ["groups"], bottom_cutoff=1, top_num=2000)
    # find_top_diff_from_sim(best_sim_list, best_diff_pairs, top_num)
    # find_intersection(best_sim_list, best_diff_list, sim_total_score, diff_total_score, sim_total_count, diff_total_count)

    vid_names = [
        "kelly_front_s31_v2/", 
        "kelly_low_s31_v2/",
        "kelly_left_threequarter_s31_v2/", 
        "kelly_right_threequarter_s31_v2/",
        "hadleigh_right_threequarter_s31_v2/" 
    ]

    make_plots(target_pairs, vid_names)

    
    


if __name__ == "__main__": 
    main()