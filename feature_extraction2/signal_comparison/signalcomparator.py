import pandas as pd 
from IPython.display import display
import matplotlib.pylab as plt
from itertools import combinations
import os
import scipy.stats as stats
from signal_comparison.artifactgenerator import ArtifactGenerator


class SignalComparator: 
    def __init__(self, odata, pdata, ndata, output_dir, test_sets, top_cutoff, bottom_cutoff, top_num, bottom_num, make_plots, run_test_sets, use_processed, lkeys_to_plot, vids_to_plot, plot_labels): 
        self.odata = odata
        self.ndata = ndata
        self.pdata = pdata
    
        if use_processed: 
            self.data = pdata
        else:
            self.data = odata
            
        self.artifactgen = ArtifactGenerator(output_dir)
        self.test_sets = test_sets
        self.top_cutoff = top_cutoff 
        self.bottom_cutoff = bottom_cutoff 
        self.top_num = top_num
        self.bottom_num = bottom_num
        self.make_plots = make_plots
        self.tc_lkey_dict = {}
        self.diffsim_dict = {}
        self.run_test_sets = run_test_sets
        self.lkeys_to_plot = lkeys_to_plot
        self.vids_to_plot = vids_to_plot
        self.plot_labels = plot_labels
        
    
    def run(self): 
        if self.run_test_sets: 
            self.run_testsets()
            self.find_best_lkeys()
        if self.make_plots: 
            self.artifactgen.plot_comparison(self.odata, self.pdata, self.ndata, self.lkeys_to_plot, self.vids_to_plot, self.plot_labels)
        

    def find_best_lkeys(self): 
        self.find_top_diff_from_sim()
        self.find_intersection()

    def find_top_diff_from_sim(self):
        top_sim = self.diffsim_dict["sim"]["tc data"]
        top_diff = self.diffsim_dict["diff"]["tc data"]
        overall = {k : v for k, v in top_diff.items() if k in top_sim}
        best_performing_diff, total_count, total_score = self.find_best_performing(overall, False)
        self.artifactgen.write_bestperforming_report(best_performing_diff, overall, total_count, self.top_num, "", "sim_then_diff")

    def find_intersection(self):
        s_best_pairs = [x for x, y in self.diffsim_dict["sim"]["best performing"]][:self.top_num]
        d_best_pairs = [x for x, y in self.diffsim_dict["diff"]["best performing"]][:self.bottom_num]
        s_best_pairs_set = set(s_best_pairs)
        d_best_pairs_set = set(d_best_pairs)
        intersection = s_best_pairs_set.intersection(d_best_pairs_set)
        s_total_count = self.diffsim_dict["sim"]["total count"]
        d_total_count = self.diffsim_dict["diff"]["total count"]
        s_total_score = self.diffsim_dict["sim"]["total score"]
        d_total_score = self.diffsim_dict["diff"]["total score"]
    
        self.artifactgen.write_intersect_report(intersection, s_total_count, d_total_count, s_total_score, d_total_score)
  
    def run_testsets(self): 
        for ts_name, testset in self.test_sets.items(): 
            self.tc_lkey_dict = {}
            if "angle" in ts_name: #if testing across different angles, we're looking for high similarity 
                finding_sim = True
                r_cutoff = self.top_cutoff
                num_cutoff = self.top_num
                diffsim_key = "sim"
                
            else: #otherwise we're looking for low correlation 
                finding_sim = False
                r_cutoff = self.bottom_cutoff
                num_cutoff = self.bottom_num
                diffsim_key = "diff"

            for testcase_name, testcase_files in testset.items(): 
                print("completed testcase: " + testcase_name)
                
                testcase_res = self.run_testcases(testcase_files, testcase_name, True, r_cutoff, ts_name)
                self.add_to_lkey_dict(testcase_res, testcase_name)
        
            best_performing, total_count, total_score = self.find_best_performing(self.tc_lkey_dict, finding_sim)
            self.artifactgen.write_bestperforming_report(best_performing, self.tc_lkey_dict, total_count, num_cutoff, ts_name)
            self.update_diffsim_dict(best_performing, total_count, total_score, self.tc_lkey_dict, diffsim_key)
            print("completed testcases for testset: " + ts_name)

    def update_diffsim_dict(self, best_performing, total_count, total_score, tc_data, key):
        if key not in self.diffsim_dict:
            self.diffsim_dict[key] = {
                "best performing" : best_performing,
                "total count" : total_count,
                "total score" : total_score,
                "tc data" : tc_data
            }
        else:
            incomingbp_dict = {tup[0] : tup[1] for tup in best_performing}
            ogbp_dict = {tup[0] : tup[1] for tup in self.diffsim_dict[key]["best performing"]}

            bp_lkey_list, merged_bestperforming = self.merge_best_performing(incomingbp_dict, ogbp_dict)
            merged_total_count = {}
            merged_total_score = {}
            
            for lkey in bp_lkey_list: 
                merged_total_count[lkey] = self.diffsim_dict[key]["total count"][lkey] + total_count[lkey]
                merged_total_score[lkey] = self.diffsim_dict[key]["total score"][lkey] + total_score[lkey]
                self.diffsim_dict[key]["tc data"][lkey].update(tc_data[lkey])
            
            self.diffsim_dict[key]["best performing"] = merged_bestperforming
            self.diffsim_dict[key]["total count"] = merged_total_count
            self.diffsim_dict[key]["total score"] = merged_total_score
    
    def merge_best_performing(self, og_bp, incoming_bp): 
        merged = []
        lkey_list = []
        for k, v in og_bp.items(): 
            if k in incoming_bp: 
                avg = (v + incoming_bp[k]) / 2
                merged.append((k, avg))
                lkey_list.append(k)

        return lkey_list, merged
            
    
    def find_best_performing(self, lkey_data, reverse):

        total_count = {}
        total_score = {}
        data = {}
        for lkey, v in lkey_data.items():
            for testcase, avg in v.items():
                if lkey in total_count:
                    total_count[lkey] += 1
                    total_score[lkey] += abs(avg)

                else:
                    total_count[lkey] = 1
                    total_score[lkey] = avg
        
        best_performing = sorted(total_score.items(), key=lambda x:x[1], reverse=reverse)
        return best_performing, total_count, total_score
      
    def run_testcases(self, tc_files, tc_name, looking_for_sim, cutoff, ts_name):
        filepairs = list(combinations(tc_files, 2))
        res_dict = {}
        for fpair in filepairs: 
            self.find_corr_between_filepair(fpair[0], fpair[1], res_dict)
        
        averaged_corr = self.get_avg(res_dict)
        bounded_res = self.artifactgen.write_testc_report(
            res_dict, 
            averaged_corr, 
            tc_name, 
            looking_for_sim, 
            cutoff, 
            tc_files, 
            ts_name)
        return bounded_res
    
    def get_avg(self, corr_dict):
        averaged_corr = {}
        for lkey, r_corr in corr_dict.items():
            s = sum(r_corr.values())
            l = len(r_corr)
            averaged_corr[lkey] = round(s/l, 5)
        
        return averaged_corr

    def find_corr_between_filepair(self, file_path1, file_path2, corr_dict):
        df_1 = self.data[file_path1]
        df_2 = self.data[file_path2]   

        for index, row in df_1.iterrows():
            rkey, corr = self.find_single_corr(row, df_2.loc[index], file_path1, file_path2)
            lkey = row["Landmark_key"]
            if lkey in corr_dict:
                corr_dict[lkey][rkey] = corr
            else:
                corr_dict[lkey] = {rkey : corr}
        
    def find_single_corr(self, row1, row2, file_path1, file_path2):

        if not isinstance(row1["Data"], str):
            #for some reason some rows are lists and some are str. This converts everything to a string for consistency 
            row1["Data"] = str(row1["Data"])
            row2["Data"] = str(row2["Data"])

        row1_l = row1["Data"].replace("]", "").replace("[", "").split(",")
        row1_l = [float(i) for i in row1_l]
        row1["Data"] = row1_l
        row2_l = row2["Data"].replace("]", "").replace("[", "").split(",")
        row2_l = [float(i) for i in row2_l]
        row2["Data"] = row2_l

        if len(row1["Data"]) > len(row2["Data"]):
            cropped1 = row1["Data"][:len(row2["Data"])]
            cropped2 = row2["Data"]
        else:
            cropped2 = row2["Data"][:len(row1["Data"])]
            cropped1 = row1["Data"]

        df = pd.DataFrame(data = {file_path1 : cropped1, file_path2: cropped2})

        r_and_p = str(stats.pearsonr(cropped1, cropped2))

        r_and_p_l = r_and_p.replace("PearsonRResult(statistic=","").replace(" pvalue=","").replace(")","").split(",")
        rounded_randp = round(float(r_and_p_l[0]), 5)

        res_key = file_path1 + "_vs_" + file_path2
        res_key = res_key.replace("/","")

        return res_key, rounded_randp

    
    def add_to_lkey_dict(self, bounded_res, tc_name):

        for k, v in bounded_res.items():
            if k in self.tc_lkey_dict:
                self.tc_lkey_dict[k][tc_name] = v
            else:
                self.tc_lkey_dict[k] = {tc_name : v}



                






