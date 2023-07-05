import pandas as pd 
from IPython.display import display
import matplotlib.pylab as plt
from itertools import combinations
import os
import scipy.stats as stats
# from importlib.metadata import version
# import dataframe_image as dfi  
# from sklearn import preprocessing


class CorrAnalyzer: 
    def __init__(self, files, analysis, makeplots, data, output_dir_root, file_extension, most_similar, cutoff, min_max_scale=True,  r_window_size=25): 
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
        self.min_max_scale = min_max_scale

        self.pairs = list(combinations(self.files, 2))

    def init_output_dir(self):

        if not os.path.exists(self.output_dir_root):
            os.makedirs(self.output_dir_root)


    def find_single_corr(self, row1, row2, file_path1, file_path2):

        if len(row1["Distances"]) > len(row2["Distances"]):
            cropped_dist1 = row1["Distances"][:len(row2["Distances"])]
            cropped_norm1 = row1["Normalized"][:len(row2["Normalized"])]
            cropped_dist2 = row2["Distances"]
            cropped_norm2 = row2["Normalized"]
        else:
            cropped_dist2 = row2["Distances"][:len(row1["Distances"])]
            cropped_norm2 = row2["Normalized"][:len(row1["Normalized"])]
            cropped_dist1 = row1["Distances"]
            cropped_norm1 = row1["Normalized"]

        if self.min_max_scale: 
            row1_data = cropped_norm1
            row2_data = cropped_norm2
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


        if self.makeplots:

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
                row1["Landmark_key"] + 
                ", rolling window: " + 
                str(r_window_size) + 
                " frames, Pearson value: " + 
                str(round(float(r_and_p_l[0]), 5)))

            plt.savefig(output_directory + "_landmark_" + row1["Landmark_key"])


    def find_corr_between_pairs(self, file_path1, file_path2):
        df_1 = self.data[file_path1]
        df_2 = self.data[file_path2]    
        
        for index, row in df_1.iterrows():
            self.find_single_corr(row, df_2.iloc[index], file_path1, file_path2)
        
        # write_analysis_report(output_directory, file_path1, file_path2, pearson_r_and_p, r_window_size)

    
    def compare_data(self): 

        for pair in self.pairs: 
            self.find_corr_between_pairs(pair[0], pair[1])
        
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

                


        # df = pd.DataFrame(data=final_res)



    # def write_analysis_report1(self, output_dir, file_path1, file_path2, r_and_p, r_window_size):
    #     with open(output_dir + "report.txt", "w") as f:
    #         f.write("Comparing " + file_path1 + " with " + file_path2)
    #         f.write("\nNormalized by " + normalize_by.replace("_"," "))
    #         f.write("\nWindow size: " + str(r_window_size))
    #         f.write("\nAnalysis type: " + analysis_type)
    #         f.write("\n")
    #         s = 0
    #         rvalues = []
    #         for key, value in r_and_p.items():
    #             s += float(value[0])
    #             rvalues.append(float(value[0]))
    #             f.write("\nLandmark " + key + " r value: " + value[0] + ", p value: " + value[1])

    #         f.write ("\n\n Median Pearson R value: " + str(round(statistics.median(rvalues), 4)))
    #         f.write ("\n Average Pearson R value: " + str(round(s/len(r_and_p), 4)))


  
            



