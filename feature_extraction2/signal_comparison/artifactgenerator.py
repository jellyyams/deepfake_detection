import os

class ArtifactGenerator(): 
    def __init__(self, output_dir): 
        self.output_dir = output_dir
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def write_testc_report(self, results, averaged_results, tc_name, looking_for_sim, cutoff, tc_files, dirname):
        #create final dataframe that can be displayed 
        #rows: each landmark pair 
        #columns: each comparison pair and final avg 

        bounded_res = {}
        results_avg_list = sorted(averaged_results.items(), key=lambda x:x[1], reverse=looking_for_sim)
        path = self.output_dir + dirname + "/"
        if not os.path.exists(path):
            os.makedirs(path)

        with open(path + tc_name + "_report.txt", 'w') as f:
            f.write("Comparing the following files: \n")
            for fil in tc_files:
                f.write(fil)
                f.write(",  ")
            
            f.write("\n\nCorrelation cuttoff is " + str(cutoff) + "\nlandmark(s),  ") 
            for k, v in results[results_avg_list[0][0]].items():
                f.write(k + ",  ")
            f.write("average")

            for i in range(len(results_avg_list)):

                avg = results_avg_list[i]
                if avg[1] > cutoff and not looking_for_sim: 
                    break
                elif avg[1] < cutoff and looking_for_sim: 
                    break 

                f.write("\n" + str(avg[0]) + ",     ")

                for k, v in results[avg[0]].items():
                    f.write(str(v) + ",       ")

                f.write(str(avg[1]))
                bounded_res[avg[0]] = avg[1]
            
        return bounded_res

    def write_bestperforming_report(self, best_performing, data, total_count, num_cutoff, dirname, fprefix=""):
        path = self.output_dir + dirname + "/"
        if not os.path.exists(path):
            os.makedirs(path)
 
        r = min(len(best_performing), num_cutoff)

        with open(path + fprefix + "report.txt", 'w') as f:
            for i in range(r):
                pair_and_score = best_performing[i]
                count = total_count[str(pair_and_score[0])]
        
                f.write("\n" + str(pair_and_score[0]) + " appeared " + str(count) + " times with an average score of " + str(float(pair_and_score[1])/count) + " in: ")
                for testcase, pearsonr in data[pair_and_score[0]].items():
                    f.write(str(testcase) + "(r value: " + str(pearsonr) + ")")
    
    def write_intersect_report(self, intersection, scount, dcount, sscore, dscore): 
        with open(self.output_dir + "intersect_report.txt", "w") as f:
            for s in intersection:
                s_avg = sscore[s] / scount[s]
                d_avg = dscore[s] / dcount[s]
                f.write(s + " similarity avg pearson: " + str(s_avg) + " difference avg pearson: " + str(d_avg) + "\n")
    
    def plot_before_after(self, data, labels):
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
   
    
  