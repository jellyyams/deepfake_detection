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
from config import ExtractionSettings, TestCases, PlottingSettings, ComparisonSettings, OutputDir, CompareResults, SignalProcessingSettings, CoordProcessingSettings, Run
from signal_processing import SignalProcessesor


class Driver:
    def __init__(self, extraction_settings, plotting_settings, test_cases, comp_settings, output_dir, coordprocessing_settings, signalprocessing_settings):
        self.signalP = SignalProcessesor(signalprocessing_settings["moving_avg_window"])
        self.should_avg_pairs = signalprocessing_settings["averagepairs?"]
        self.preavg_process = signalprocessing_settings["preavg_process"]
        self.postavg_process = signalprocessing_settings["postavg_process"]
        self.pairs_for_avg = coordprocessing_settings["pairs_for_avg"]
        self.target_pairs = coordprocessing_settings["target_pairs"]
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
        self.output_data = extraction_settings["output_data"] 
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
        if self.should_avg_pairs:
            for p in self.target_pairs: 
                for x in self.pairs_for_avg[p]:
                    res.append(x)
            
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
                output_data = self.output_data,
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
                    if not self.should_filter_pairs or (any(str(pair) in row["Landmark_key"] for pair in self.filter_pairs)): 
                        l = row["Distances"].replace("]", "").replace("[", "").split(",")
                        l = [float(i) for i in l]

                        distances.append(l)
                        normed, p = self.signalP.process_signal(l, self.preavg_process)
                        processed.append(p)
                        normalized.append(normed)
                        
                    else: 
                        df.drop(index, inplace=True)
            
                df["Distances"] = distances
                df["Normalized"] = normalized
                df["Processed"] = processed

                data[f_path] = df
        
        return data
        
    def retrieve_data(self, files, datatype):
       
        data = self.read_csv_into_dict(files, datatype)
        if self.should_avg_pairs:
            return self.signalP.avg_across_signals(data, self.postavg_process, self.target_pairs, self.pairs_for_avg)
        else:
            return data
        

    def run_test_cases(self, testcases, cutoff, topnum, reverse):
        top_pairs = {}
        for k, v in testcases["data"].items():
            d = self.retrieve_data(v, ["groups"])
            corr_analyzer = CorrAnalyzer(v, ["groups"], False, d, "correlation_reports/", k, most_similar=reverse, cutoff = cutoff)
            newpairs = corr_analyzer.compare_data()
            self.add_to_final_dict(top_pairs, newpairs, k)


        self.write_data_into_csv(top_pairs, testcases["name"], cutoff)
        best_performing, total_count, total_score = self.find_best_performing(top_pairs, reverse)
        self.write_report(best_performing, top_pairs, total_count, testcases["name"], topnum)
        best_pairs = [x for x, y in best_performing]
        
        return best_pairs[:topnum], top_pairs, total_count, total_score
    
    def run_corr_analyzer(self):
        for sim in self.sim_test_cases: 
            best_sim_list, best_sim_pairs, sim_total_count, sim_total_score = self.run_test_cases(sim, self.top_cutoff, self.top_num, reverse=True)
        
        for diff in self.diff_test_cases: 
            best_diff_list, best_diff_pairs, diff_total_count, diff_total_score = self.run_test_cases(diff, self.bottom_cutoff, self.bottom_num, reverse=False)
        # self.find_top_diff_from_sim(best_sim_list, best_diff_pairs)
        # self.find_intersection(best_sim_list, best_diff_list, sim_total_score, diff_total_score, sim_total_count, diff_total_count)

        
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
            r = len(best_performing) 
        else: 
            r = top_num

        with open(self.output_dir + pref + "report.txt", 'w') as f:
            for i in range(r):
                pair_and_score = best_performing[i]
                count = total_count[str(pair_and_score[0])]
        
                f.write("\n" + str(pair_and_score[0]) + " appeared " + str(count) + " times with an average score of " + str(float(pair_and_score[1])/count) + " in: ")
                for testcase, pearsonr in data[pair_and_score[0]].items():
                    f.write(str(testcase) + "(r value: " + str(pearsonr) + ")")


    def plot_pairs_across_vids(self, data, labels):
        for pair in self.target_pairs:
            plt.clf()
            f, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            ax[0].set(xlabel="Frames", ylabel="Min-max normed distances", ylim=(0,1))
            ax[1].set(xlabel="Frames", ylabel="Processed distances")
            for vid in self.videos_for_plotting:
                df = data[vid]
                index = int(np.where(df["Landmark_key"] == str(pair))[0][0])

                ax[0].plot(df["Normalized"].iloc[index], label=labels[vid])
                ax[1].plot(df["Processed"].iloc[index], label=labels[vid])
                n = vid.replace("/","_")
                # self.signalP.simple_moving_avg(df.loc[index,"Normalized"], 2, n)
            
            f.tight_layout(pad=3.0)
            plt.legend(bbox_to_anchor=(1.0, 2),loc="upper left")
            plt.suptitle("Raw vs. Processed Distance Data For Pair " + str(pair))

            if not os.path.exists(self.output_dir + "plots/"):
                os.makedirs(self.output_dir + "plots/")

            plt.savefig(self.output_dir + "plots/" + str(pair), bbox_inches='tight')
    
    def plot_single_vid(self, data, vid): 
        plt.clf()
        for pair in self.target_pairs:

            df = data[vid]
            index = int(np.where(df["Landmark_key"] == str(pair))[0][0])

            plt.plot(df["Processed"].iloc[index], label=pair)
          
        plt.xlabel("Video frames")
        plt.ylabel("Smoothed Distance (meters)")
        plt.legend(bbox_to_anchor=(1.08, 1), loc="upper left", title="landmark pairs")
        plt.suptitle("Moving Average Smoothed Distance Signals For Sample Video")
        plt.savefig("plots/test2", bbox_inches='tight')



    def make_plots(self, vid, labels):
        d = self.retrieve_data(self.videos_for_plotting, ["groups"])
        self.plot_pairs_across_vids(d, labels)
        self.plot_single_vid(d, vid)
    
    def analyze_results(self, tocomparedir, tocomparefiles, titles, xlabels):
        filedata = {}
  
        for f in tocomparefiles:
            
            filedata[f] = {
                "index" : [],
                "xlabels" : [], 
                "avgs" : [], 
                "pairs" : {}
            }
            
            index = 0
            for directory in tocomparedir: 
                index += 1
                fi = open("./correlation_reports/" + directory + f, 'r') 
                lines = fi.readlines()
                dirname = directory.split("/")[-2] 
                filedata[f]["xlabels"].append(xlabels[dirname])
                filedata[f]["index"].append(index)
                avgs = []
                filedata[f]["pairs"][dirname] = {}
                for line in lines: 
                    if any(str(pair) in line for pair in self.target_pairs):
                        words = line.split(" ")
                        num1 = int(words[0].replace("(","").replace(",",""))
                        num2 = int(words[1].replace(")",""))
                        t = (num1, num2)
                        avg = float(words[10])
                        filedata[f]["pairs"][dirname][t] = avg
                        avgs.append(avg)

                filedata[f]["avgs"].append(avgs)
            
            self.box_and_whisker(filedata, f, titles)
        
        return filedata
                    
    def box_and_whisker(self, data, f, titles): 
        plt.clf()
        plt.boxplot(data[f]["avgs"])
        plt.xticks(data[f]["index"], data[f]["xlabels"], rotation=80)
        plt.suptitle(titles[f])
        fname = f.replace(".txt", "")
        plt.savefig(fname, bbox_inches='tight')  
              
    
    def scatter_plot(self, tocomparefiles, dirs, titles, markers, data):
        plt.clf()
        pairdata = {dirs[0] : {}, dirs[1] : {}}
        tempdata = {dirs[0] : [], dirs[1] : []}
    
        for i, d in enumerate(dirs):
            simavg = {}
            diffavg = {}
            for f in tocomparefiles: 
                for tup, avg in data[f]["pairs"][d].items():
                    if "angle" in f: 
                        simavg[tup] = avg
                    elif "utterance" in f:
                        diffavg[tup] = avg
                    # elif tup in diffavg:
                    #     diffavg[tup] = float((diffavg[tup] + avg) / 2)
                    # else: 
                    #     diffavg[tup] = avg
            
            for k, v in simavg.items():
                diff = diffavg[k]
                pairdata[d][k] = (v, diff)
                tempdata[d].append((v, diff))
                plt.plot(v, diff, marker=markers[d], label=k)

       
        plt.xlabel("Pearson Correlation for Similarity Test Cases")

        plt.ylabel("Pearson Correlation for Differences Test Cases")
        # handles, labels = plt.gca().get_legend_handles_labels()
        # by_label = dict(zip(labels, handles))
        plt.legend(bbox_to_anchor=(1.0, 2),loc="upper left")
        plt.savefig("scatterplottest2", bbox_inches='tight')  
            

def main():

    driver = Driver(ExtractionSettings, PlottingSettings, TestCases, ComparisonSettings, OutputDir, CoordProcessingSettings, SignalProcessingSettings)
    
    if Run["video extractions"]: driver.run_extractions()
    if Run["test cases"]: driver.run_corr_analyzer()
    if Run["pair plotting"]: driver.make_plots(PlottingSettings["singlevid"], PlottingSettings["labels"])
    if Run["correlation comparison"]: 
        data = driver.analyze_results(CompareResults["directories"], CompareResults["files"], CompareResults["titles"], CompareResults["xlabels"])
        driver.scatter_plot(CompareResults["files"], CompareResults["scatter_dirs"], CompareResults["titles"], CompareResults["scatter_markers"], data)



if __name__ == "__main__":
    main()

