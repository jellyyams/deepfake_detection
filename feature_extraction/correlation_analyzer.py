import pandas as pd 
from IPython.display import display
import matplotlib.pylab as plt
from itertools import combinations
import os
import scipy.stats as stats
# from importlib.metadata import version
# import dataframe_image as dfi  


class CorrAnalyzer: 
    def __init__(self, files, analysis, makeplots, data, output_dir_root, most_similar, top_num, r_window_size=25): 
        self.most_similar = most_similar
        self.top_num = top_num
        self.files = files
        self.data = data
        self.analysis = analysis
        self.output_dir_root = output_dir_root
        self.init_output_dir()
        self.makeplots = makeplots 
        self.results = {}
        self.results_avg = {}
        self.r_window_size = r_window_size
        self.final_results = {}

        self.full_paths = self.set_full_paths()
        self.pairs = list(combinations(self.full_paths, 2))

    def init_output_dir(self):

        if not os.path.exists(self.output_dir_root):
            os.makedirs(self.output_dir_root)
    

    def set_full_paths(self):
        full_paths = []
        for path in self.files: 
            for a in self.analysis: 
                full_paths.append(path+a)
        
        return full_paths

    def sort_landmark(self, pair):
        pair_list = pair.replace("(","").replace(" ", "").replace(")", "").split(",")
        smaller = min(int(pair_list[0]), int(pair_list[1]))
        larger = max(int(pair_list[0]), int(pair_list[1]))

        return (smaller, larger)


    def find_single_corr(self, row1, row2, file_path1, file_path2):
        if len(row1["Distances"]) > len(row2["Distances"]):
            row1["Distances"] = row1["Distances"][:len(row2["Distances"])]
        else:
            row2["Distances"] = row2["Distances"][:len(row1["Distances"])]
        

        df = pd.DataFrame(data = {file_path1 : row1["Distances"], file_path2: row2["Distances"]})

        r_and_p = str(stats.pearsonr(row1["Distances"], row2["Distances"]))

        r_and_p_l = r_and_p.replace("PearsonRResult(statistic=","").replace(" pvalue=","").replace(")","").split(",")

        res_key = file_path1 + "_vs_" + file_path2
        res_key = res_key.replace("/","")
        key = self.sort_landmark(row1["Landmark"])


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
                row1["Landmark"] + 
                ", rolling window: " + 
                str(r_window_size) + 
                " frames, Pearson value: " + 
                str(round(float(r_and_p_l[0]), 5)))

            plt.savefig(output_directory + "_landmark_" + row1["Landmark"])


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
        self.write_analysis_report(self.top_num)
        print("done")
        
        return self.results

    def get_avg(self):
        for k, v in self.results.items():
            s = sum(v.values())
            l = len(v)
            self.results_avg[k] = round(s/l, 5)


    def write_analysis_report(self, top):
        #create final dataframe that can be displayed 
        #rows: each landmark pair 
        #columns: each comparison pair and final avg 

        self.results_avg = sorted(self.results_avg.items(), key=lambda x:x[1], reverse=self.most_similar)
        with open(self.output_dir_root + "report.txt", 'w') as f:
            f.write("Comparing the following files: \n")
            for fil in self.full_paths:
                f.write(fil)
                f.write(",  ")
            
            f.write("\n\nTop " + str(top) + " correlations: ") 
            f.write("\nlandmark(s),  ")
            for k, v in self.results[self.results_avg[0][0]].items():
                f.write(k + ",  ")
            f.write("average")

            for i in range(top):

                avg = self.results_avg[i]
                f.write("\n" + str(avg[0]) + ",  ")

                if "landmark(s)" in self.final_results:
                    self.final_results["landmark(s)"].append(avg[0])
                else:
                    self.final_results["landmark(s)"] = [avg[0]]

                for k, v in self.results[avg[0]].items():
                    f.write(str(v) + ",  ")
                    if k in self.final_results:
                        self.final_results[k].append(v)
                    else:
                        self.final_results[k] = [v]

                f.write(str(avg[1]) + ",  ")
                
                if "average" in self.final_results:
                    self.final_results["average"].append(avg[1])
                else:
                    self.final_results["average"] = [avg[1]]

                


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


  
            



