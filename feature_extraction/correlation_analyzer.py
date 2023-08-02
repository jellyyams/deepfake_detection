import pandas as pd 
from IPython.display import display
import matplotlib.pylab as plt
from itertools import combinations
import os
import scipy.stats as stats


class CorrAnalyzer: 
    def __init__(self, files, analysis, makeplots, data, output_dir_root, file_extension, most_similar, cutoff, processed=True,  r_window_size=25): 
        self.most_similar = most_similar
        self.file_extension = file_extension
        self.cutoff = cutoff
        self.files = files
        self.data = data
        self.analysis = analysis
        self.output_dir_root = output_dir_root
        self.init_output_dir()
        self.makeplots = makeplots 
        self.results = {}
        self.results_avg = {}
        self.results_avg_list = []
        self.r_window_size = r_window_size
        self.bounded_results = {}
        self.best_landmarks = []
        self.processed = processed

        self.filepairs = list(combinations(self.files, 2))

    def process_signal(self, raw, processtype): 
        normed = self.signalP.min_max_norm(raw)
        if processtype == "moving_average": 
            processed = self.signalP.simple_moving_avg(raw, self.window)
            return normed, processed
        elif processtype == "normalize": 
            return normed, normed
        else:
            return normed, raw

    def init_output_dir(self):

        if not os.path.exists(self.output_dir_root):
            os.makedirs(self.output_dir_root)


    def find_single_corr(self, row1, row2, file_path1, file_path2):

        if len(row1["Distances"]) > len(row2["Distances"]):
            cropped_dist1 = row1["Distances"][:len(row2["Distances"])]
            cropped_processed1 = row1["Processed"][:len(row2["Processed"])]
            cropped_dist2 = row2["Distances"]
            cropped_processed2 = row2["Processed"]
        else:
            cropped_dist2 = row2["Distances"][:len(row1["Distances"])]
            cropped_processed2 = row2["Processed"][:len(row1["Processed"])]
            cropped_dist1 = row1["Distances"]
            cropped_processed1 = row1["Processed"]

        if self.processed: 
            row1_data = cropped_processed1
            row2_data = cropped_processed2
        else: 
            row1_data = cropped_dist1
            row2_data = cropped_dist2

        

        df = pd.DataFrame(data = {file_path1 : row1_data, file_path2: row2_data})

        r_and_p = str(stats.pearsonr(row1_data, row2_data))

        r_and_p_l = r_and_p.replace("PearsonRResult(statistic=","").replace(" pvalue=","").replace(")","").split(",")

        res_key = file_path1 + "_vs_" + file_path2
        res_key = res_key.replace("/","")
        key = row1["Landmark_key"]


        if key in self.results:
            self.results[key][res_key] = round(float(r_and_p_l[0]), 5)
        else:
            self.results[key] = {res_key : round(float(r_and_p_l[0]), 5)}


    def find_corr_between_filepairs(self, file_path1, file_path2):
        df_1 = self.data[file_path1]
        df_2 = self.data[file_path2]    
        
        for index, row in df_1.iterrows():
            self.find_single_corr(row, df_2.loc[index], file_path1, file_path2)
        
        # write_analysis_report(output_directory, file_path1, file_path2, pearson_r_and_p, r_window_size)

    
    def compare_data(self): 

        for filepair in self.filepairs: 
            self.find_corr_between_filepairs(filepair[0], filepair[1])
        
        self.get_avg()
        self.write_analysis_report()
        print("done")
        
        return self.bounded_results

    def get_avg(self):
        for k, v in self.results.items():
            s = sum(v.values())
            l = len(v)
            self.results_avg[k] = round(s/l, 5)


    def write_analysis_report(self):
        #create final dataframe that can be displayed 
        #rows: each landmark pair 
        #columns: each comparison pair and final avg 

        self.results_avg_list = sorted(self.results_avg.items(), key=lambda x:x[1], reverse=self.most_similar)
        with open(self.output_dir_root + self.file_extension + "_report.txt", 'w') as f:
            f.write("Comparing the following files: \n")
            for fil in self.files:
                f.write(fil)
                f.write(",  ")
            
            f.write("\n\nCorrelation cuttoff is " + str(self.cutoff)) 
            f.write("\nlandmark(s),  ")
            for k, v in self.results[self.results_avg_list[0][0]].items():
                f.write(k + ",  ")
            f.write("average")

            for i in range(len(self.results_avg_list)):

                avg = self.results_avg_list[i]
                if avg[1] > self.cutoff and not self.most_similar: 
                    break
                elif avg[1] < self.cutoff and self.most_similar: 
                    break 

                f.write("\n" + str(avg[0]) + ",     ")

                for k, v in self.results[avg[0]].items():
                    f.write(str(v) + ",       ")

                f.write(str(avg[1]))
                

                self.bounded_results[avg[0]] = avg[1]

                






