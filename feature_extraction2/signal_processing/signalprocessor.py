import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal 
import os
import math

class SignalProcessor:
    def __init__(self, make_plots_for, should_filter, pipeline, movingavg_window, butter_settings, output_dir, pairs_to_avg):
        self.make_plots_for = make_plots_for
        self.pipeline = pipeline
        self.should_filter = should_filter
        self.movingavg_window = movingavg_window
        self.butter_settings = butter_settings
        
        self.pairs_to_avg = pairs_to_avg
        self.filter_lkeys = self.gen_filter_lkeys()
        self.output_dir = output_dir
        self.plot_tracker = {}
        self.last_step = 0

        for lkey in make_plots_for: 
            self.plot_tracker[lkey] = {}
    
    def gen_filter_lkeys(self): 
        res = []
        
        for k, v in self.pairs_to_avg.items(): 
            res.append(k)
            res + v
            
        return list(set(res))

    
    def run(self, dframe, dirname, fname):
        self.last_step = 0
        plotdir = self.output_dir + dirname + "plots/"
        if "avg_across_signals" in self.pipeline: 
            avg_index = self.pipeline.index("avg_across_signals")
            preavg_pipeline = self.pipeline[:avg_index]
            postavg_pipeline = self.pipeline[avg_index+1:]

            pdata, ndata = self.filter_and_preavg_process(preavg_pipeline, dframe)
            self.average_across_signals()

            #average 
            
            print(preavg_pipeline)
            print(postavg_pipeline)
        else:
            pdata, ndata = self.filter_and_preavg_process(self.pipeline, dframe)
        self.make_plots(plotdir, fname)
        
    
    def make_plots(self, plotdir, fname): 
        if not os.path.exists(plotdir):
            os.makedirs(plotdir)
   
        for lkey in self.make_plots_for: 
            self.plot_tracker[str(lkey)][0]["after"] = self.plot_tracker[str(lkey)][self.last_step]["after"]
            self.plot_res(self.plot_tracker[str(lkey)], str(lkey) + fname, plotdir)

        
    def filter_and_preavg_process(self, pipeline, dframe): 
        pdata = {}
        ndata = {}
        for index, row in dframe.iterrows():
            if not self.should_filter or (any(str(lkey) in row["Landmark_key"] for lkey in self.lkeys)): 
                l = row["Raw_data"].replace("]", "").replace("[", "").split(",")
                l = [float(i) for i in l]
                pair = row["Landmark_key"]
                self.plot_tracker[pair] = {
                    0 : {
                        "type" : "Entire pipeline", 
                        "before" : l, 
                        "after" : []
                    }
                }
                n, p, i = self.process_signal(l, pipeline, pair)
     
                pdata[pair] = p
                ndata[pair] = n 
        self.last_step = self.last_step + i + 1
        return pdata, ndata
         
                
    def process_signal(self, raw, pipeline, pair): 
        normed_raw = self.min_max_norm(raw)
        curr_signal = raw
      
        for i, ptype in enumerate(pipeline):
            self.plot_tracker[pair][self.last_step + i + 1] = {"type" : ptype, "before":curr_signal}
            if ptype == "moving_average": 
                curr_signal = self.simple_moving_avg(curr_signal)
            elif ptype == "normalize": 
                curr_signal = self.min_max_norm(curr_signal)
            elif ptype == "butterworth":
                curr_signal = self.butterworth(curr_signal)
            else:
                #implement throw error
                print("please revisit configuration file and enter a valid signal processing type")
            self.plot_tracker[pair][self.last_step + i + 1]["after"] = curr_signal
       
        
        return normed_raw, curr_signal, i
        
    
    def butterworth(self, original):
      
        w = self.butter_settings["fc"] / (self.butter_settings["fs"]  / 2)
        b, a = signal.butter(self.butter_settings["cutoff"], w, self.butter_settings["type"])
        output = signal.filtfilt(b, a, original)

        return output
    
    def simple_moving_avg(self, original):
        i = 0
        res = []

        while i < len(original) - self.movingavg_window + 1:
            window_avg = round(np.sum(original[i:i+self.movingavg_window]) / self.movingavg_window, 6)
            res.append(window_avg)
            i += 1

        # for i in range(self.movingavg_window - 1):
        #     res.append(window_avg)

        return res 

    def min_max_norm(self, original): 
        smallest = min(original)
        largest = max(original)
        d_scaled = [(x - smallest)/(largest - smallest) for x in original]

        return d_scaled


    def avg_one_pair(self, pairs_toavg, row1_data, pd_df):
        averaged = row1_data
        for toavg in pairs_toavg: 
            nextrow_index = pd_df.index[pd_df["Landmark_key"] == str(toavg)]
            nextrow_data = pd_df["Processed"].loc[nextrow_index].values[0]
             
            averaged = [(g + h) / 2 for g, h in zip(averaged, nextrow_data)]

        return averaged


    def avg_across_signals(self, data, postavg_process, target_pairs, pairs_for_avg):
        res = {}
        for fname, pd_df in data.items(): 
            avg_data = {"Landmark_key" : [], "Distances" : [], "Normalized" : [], "Processed" : []}
        
            for index, row in pd_df.iterrows(): 
                if any(row["Landmark_key"] == str(pair) for pair in target_pairs):
                    
                    tup = eval(row["Landmark_key"].replace("(", "").replace(")", ""))
                    toavg_pairs = pairs_for_avg[tup]
                    averaged = self.avg_one_pair(toavg_pairs, row["Processed"], pd_df)
                
                    normed, p = self.process_signal(averaged, postavg_process)
    
                    avg_data["Landmark_key"].append(str(tup))
                    avg_data["Distances"].append(row["Distances"])
                    avg_data["Normalized"].append(row["Normalized"])
                    avg_data["Processed"].append(p)
                    
            avg_df = pd.DataFrame(avg_data)
            res[fname] = avg_df
                    
        return res


    def plot_res(self, data, title, plotdir):

        plt.clf()
        fig, axs = plt.subplots(len(data))
        for step, v in data.items(): 
            axs[step].plot(v["before"], label="before")
            axs[step].plot(v["after"], label="after")
            axs[step].legend()
            axs[step].set_title(v["type"])

        plt.subplots_adjust(hspace = 1)
        plt.suptitle(title)
        plt.savefig(plotdir + str(title))







