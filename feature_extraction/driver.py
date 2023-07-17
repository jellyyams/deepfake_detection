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
from statistics import mean
from mesh_data import MeshData
from mp_extractor import MPFeatureExtractor
from correlation_analyzer import CorrAnalyzer
from config import ExtractionSettings, TestCases, PlottingSettings, ComparisonSettings, OutputDir, CompareResults, ProcessingSettings
from signal_processing import SignalProcessesor


class Driver:
    def __init__(self, extraction_settings, plotting_settings, test_cases, comp_settings, output_dir, processing_settings):
        self.read_csv_data = {} 
        self.signalP = SignalProcessesor()
        self.window = processing_settings["window"]
        self.signalptype = processing_settings["type"]
        self.should_avg_pairs = processing_settings["averagepairs?"]
        self.postavg_process = processing_settings["postavg_process"]
        self.pairs_for_avg = processing_settings["pairs_for_avg"]
        self.target_pairs = plotting_settings["pairs_for_plotting"]
        self.videos_for_plotting = plotting_settings["videos_for_plotting"]
        self.sim_test_cases = [test_cases["angle_test_cases"]]
        self.diff_test_cases = [test_cases["identity_test_cases"], test_cases["utterance_test_cases"]] 
        self.top_cutoff = comp_settings["top_cutoff"]
        self.top_num = comp_settings["top_num"]
        self.bottom_cutoff = comp_settings["bottom_cutoff"]
        self.bottom_num = comp_settings["bottom_num"]
        self.should_filter_pairs = comp_settings["filterpairs?"]
        self.filter_pairs = self.get_filter_pairs()

        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.initial_detect = extraction_settings["initial_detect"]
        self.draw_all_landmarks = extraction_settings["draw_all_landmarks"]
        self.generate_video = extraction_settings["generate_video"]
        self.analysis_types = extraction_settings["analysis_types"] #vs "landmark_pairs" vs "landmark_to_anchor" vs "landmark_displacement_sig"
        self.anchor_landmark = extraction_settings["anchor_landmark"]
        self.target_landmarks = self.get_landmarks(extraction_settings["key_regions"])
        self.landmark_pairs = self.generate_landmark_pairs(self.target_landmarks)
        self.video_paths = extraction_settings["video_paths"]
        self.norm_approach = extraction_settings["norm_approach"]
        self.root_video_path = extraction_settings["root_video_path"]
            
        print(len(self.target_landmarks))
        print(len(self.landmark_pairs))

    def get_filter_pairs(self): 
        res = self.target_pairs.copy()
        for p in self.target_pairs: 
            res.append(self.pairs_for_avg[p])
        return res

    def get_landmarks(self, keywords):
        md = MeshData()
        res = []
        for k, v in md.landmarks.items():
            if any(sub in k for sub in keywords):
                res += v
        
        return res

    def generate_landmark_pairs(self, landmarks):
        c = list(combinations(landmarks, 2))
        sorted_list = []
        for pair in c:
            if pair[0] > pair[1]:
                sorted_list.append((pair[1], pair[0]))
            else:
                sorted_list.append(pair)
        
        return sorted_list


    def run_extractions(self):
        for vid_path in self.video_paths:
            input_path = self.root_video_path + vid_path + ".mp4"
            app = MPFeatureExtractor(
                input_path,
                draw_all_landmarks= self.draw_all_landmarks,
                initial_detect = self.initial_detect,
                anchor_landmark = self.anchor_landmark,
                target_landmarks = self.target_landmarks,
                landmark_pairs = self.landmark_pairs,
                generate_video = self.generate_video,
                norm_approach = self.norm_approach,
                analysis_types = self.analysis_types,
                target_pairs = self.target_pairs, 
                output_directory = self.output_dir + "videos/")
            landmark_coords, landmark_data, landmark_groups = app.run_extraction()
            pathsplit = vid_path.split('/')
            self.write_data_into_csv(landmark_coords, "coords", pathsplit[2])
            self.write_data_into_csv(landmark_data, "data", pathsplit[2])
            self.write_data_into_csv(landmark_groups, "groups", pathsplit[2])

    def write_data_into_csv(self, data, name_suff, filename):
        directory = "./landmark_analysis_data/" + str(filename) + "/"

        if not os.path.exists(directory):
            os.makedirs(directory)
        
        with open(directory + name_suff + ".csv", 'w') as f:
            writer = csv.DictWriter(f, data.keys())
            writer.writeheader()
            writer.writerow(data)


    def read_csv_into_dict(self, file_paths, fnames):
        data = {}
        
        for f_path in file_paths:
            for fname in fnames:

                full_path = "./landmark_analysis_data/" + f_path + fname + ".csv"
                df = pd.read_csv(full_path, header=None)
                df = df.rename({0:"Landmark_key", 1:"Distances"}, axis="index")
                df = df.T
            
                distances = []
                processed = []
                normalized = []

                for index, row in df.iterrows():
                    if not self.should_filter_pairs or (self.should_filter_pairs and any(str(pair) in row["Landmark_key"] for pair in self.filter_pairs)): 
                        l = row["Distances"].replace("]", "").replace("[", "").split(",")
                        l = [float(i) for i in l]

                        distances.append(l)
                        normed, p = self.process_signal(l, self.signalptype)
                        processed.append(p)
                        normalized.append(normed)
                    else: 
                        df.drop(index, inplace=True)
            
                df["Distances"] = distances
                df["Normalized"] = normalized
                df["Processed"] = processed

                self.read_csv_data[f_path] = df
                data[f_path] = df 
        
        
        return data
    
    def process_signal(self, raw, processtype): 
        normed = self.signalP.min_max_norm(raw)
        if processtype == "moving_average": 
            processed = self.signalP.simple_moving_avg(normed, self.window)
            return normed, processed
        elif processtype == "normalize": 
            return normed, normed
        elif processtype == "moving_average_nonorm": 
            processed = self.signalP.simple_moving_avg(raw, self.window)
            return normed, processed
        else:
            return normed, raw

    def avg_data(self, data):
        if self.should_avg_pairs: 
            res = {}
            for fname, pd_df in data.items(): 
                avg_data = {"Landmark_key" : [], "Distances" : [], "Normalized" : [], "Processed" : []}
            
                for index, row in pd_df.iterrows(): 
                    if any(row["Landmark_key"] == str(pair) for pair in self.target_pairs):
                        tup = eval(row["Landmark_key"].replace("(", "").replace(")", ""))
                        pair2 = self.pairs_for_avg[tup]
                        row2index = pd_df.index[pd_df["Landmark_key"] == str(pair2)]
                        row2data = pd_df["Processed"].loc[row2index].values[0]
             
                        averaged = [(g + h) / 2 for g, h in zip(row["Processed"], row2data)]
                        
                        normed, p = self.process_signal(averaged, self.postavg_process)
                        avg_data["Landmark_key"].append(str(tup))
                        avg_data["Distances"].append(row["Distances"])
                        avg_data["Normalized"].append(row["Normalized"])
                        avg_data["Processed"].append(p)
                      
                avg_df = pd.DataFrame(avg_data)
                res[fname] = avg_df
                      
            return res

        return data
                   

    def run_test_cases(self, testcases, cutoff, topnum, reverse, reportname):
        top_pairs = {}
        for k, v in testcases.items():
            data = self.read_csv_into_dict(v, ["groups"])
            d = self.avg_data(data)
            print(len(d))
            corr_analyzer = CorrAnalyzer(v, ["groups"], False, d, "correlation_reports/", k, most_similar=reverse, cutoff = cutoff)
            newpairs = corr_analyzer.compare_data()
            self.add_to_final_dict(top_pairs, newpairs, k)


        self.write_data_into_csv(top_pairs, reportname, cutoff)
        best_performing, total_count, total_score = self.find_best_performing(top_pairs, reverse)
        self.write_report(best_performing, top_pairs, total_count, reportname, topnum)
        best_pairs = [x for x, y in best_performing]
        
        return best_pairs[:topnum], top_pairs, total_count, total_score
    
    def run_corr_analyzer(self):
        best_sim_list, best_sim_pairs, sim_total_count, sim_total_score = self.run_test_cases(self.sim_test_cases, self.top_cutoff, self.top_num, True, "sim_")
        best_diff_list, best_diff_pairs, diff_total_count, diff_total_score = self.run_test_cases(self.diff_test_cases, self.bottom_cutoff, self.bottom_num, False, "diff_")
        self.find_top_diff_from_sim(best_sim_list, best_diff_pairs)
        self.find_intersection(best_sim_list, best_diff_list, sim_total_score, diff_total_score, sim_total_count, diff_total_count)

        
    def find_top_diff_from_sim(self, best_sim_list, top_diff_pairs):
        overall = {k: v for k, v in top_diff_pairs.items() if k in best_sim_list}
        best_performing_diff, total_count, total_score = self.find_best_performing(overall, False)
        self.write_report(best_performing_diff, overall, total_count, "sim_then_diff_", self.top_num)


    def find_intersection(self, s_list, d_list, sorted_sim_avg, sorted_diff_avg, total_sim, total_diff):
        s_set = set(s_list)
        d_set = set(d_list)
        intersect = d_set.intersection(s_set)
        with open(self.output_dir + "intersect_report.txt", "w") as f:
            for s in intersect:
                s_avg = sorted_sim_avg[s] / total_sim[s]
                d_avg = sorted_diff_avg[s] / total_diff[s]
                f.write(s + " similarity avg pearson: " + str(s_avg) + " difference avg pearson: " + str(d_avg) + "\n")


    def add_to_final_dict(self, d, newpairs, testcase):

        for k, v in newpairs.items():
            if k in d:
                d[k][testcase] = v
            else:
                d[k] = {testcase : v}

    def find_best_performing(self, sim_pairs, reverse):
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


    def write_report(self, best_performing, data, total_count, pref, top_num):
        if top_num > len(best_performing):
            r = len(best_performing) - 1
        else: 
            r = top_num

        with open(self.output_dir + pref + "report.txt", 'w') as f:
            for i in range(r):
                pair_and_score = best_performing[i]
                count = total_count[str(pair_and_score[0])]
        
                f.write("\n" + str(pair_and_score[0]) + " appeared " + str(count) + " times with an average score of " + str(float(pair_and_score[1])/count) + " in: ")
                for testcase, pearsonr in data[pair_and_score[0]].items():
                    f.write(str(testcase) + "(r value: " + str(pearsonr) + ")")


    def plot_pairs(self, data):
        for pair in self.target_pairs:
            plt.clf()
            f, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            ax[0].set(xlabel="Frames", ylabel="Normalized Distances", ylim=(0,1))
            ax[1].set(xlabel="Frames", ylabel="Processed Distances", ylim=(0,1))
            for vid in self.videos_for_plotting:
                df = data[vid]
                index = int(np.where(df["Landmark_key"] == str(pair))[0][0])
              

                # row2index = df.index[df["Landmark_key"] == str(pair)]
                # row2data = df_copy["Processed"].loc[row2index].values[0]

                ax[0].plot(df["Normalized"].iloc[index], label=vid)
                ax[1].plot(df["Processed"].iloc[index], label=vid)
                n = vid.replace("/","_")
                # self.signalP.simple_moving_avg(df.loc[index,"Normalized"], 2, n)
            
            f.tight_layout(pad=3.0)
            plt.legend(bbox_to_anchor=(1.0, 2),loc="upper left")
            plt.suptitle("Raw and processed distance data for pair: " + str(pair))

            if not os.path.exists(self.output_dir + "plots/"):
                os.makedirs(self.output_dir + "plots/")

            plt.savefig(self.output_dir + "plots/" + str(pair), bbox_inches='tight')


    def make_plots(self):
        data = self.read_csv_into_dict(self.videos_for_plotting, ["groups"])
        d = self.avg_data(data)
        self.plot_pairs(d)
    
    def analyze_results(self, tocomparedir, tocomparefiles, titles):
        data = {}
  
        for f in tocomparefiles:
            plt.clf()
            data[f] = {
                "index" : [],
                "directory" : [], 
                "avgs" : []
            }
            
            index = 0
            for directory in tocomparedir: 
                index += 1
                fi = open("./correlation_reports/" + directory + f, 'r') 
                lines = fi.readlines()
                dirname = directory.split("/")[-2] 
                data[f]["directory"].append(dirname)
                data[f]["index"].append(index)
                avgs = []
                for line in lines: 
                    if any(str(pair) in line for pair in self.target_pairs):
                        words = line.split(" ")
                        num1 = int(words[0].replace("(","").replace(",",""))
                        num2 = int(words[1].replace(")",""))
                        t = (num1, num2)
                        avgs.append(float(words[10]))
                data[f]["avgs"].append(avgs)

            plt.boxplot(data[f]["avgs"])
            plt.xticks(data[f]["index"], data[f]["directory"], rotation=80)
            plt.suptitle(titles[f])
            fname = f.replace(".txt", "")
            plt.savefig(fname, bbox_inches='tight')
            


def main():

    driver = Driver(ExtractionSettings, PlottingSettings, TestCases, ComparisonSettings, OutputDir, ProcessingSettings)

    driver.run_extractions()

    driver.run_corr_analyzer()
    driver.make_plots()
    driver.analyze_results(CompareResults["directories"], CompareResults["files"], CompareResults["titles"])


if __name__ == "__main__":
    main()

