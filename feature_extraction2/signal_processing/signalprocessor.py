import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal 
import os
import math

class SignalProcessor:
    def __init__(self, make_plots_bool, should_filter, pipeline, movingavg_window, butter_settings, output_dir, lkeys_to_avg, lkeys_to_plot):
        self.make_plots_bool = make_plots_bool
        self.lkeys_to_plot = lkeys_to_plot
        self.pipeline = pipeline
        self.should_filter = should_filter
        self.movingavg_window = movingavg_window
        self.butter_settings = butter_settings
        
        self.lkeys_to_avg = lkeys_to_avg
        self.filter_lkeys = self.gen_filter_lkeys()
        self.output_dir = output_dir
        self.plot_tracker = {}
        self.last_step = 0

        for lkey in self.lkeys_to_plot: 
            self.plot_tracker[lkey] = {}
    
    def gen_filter_lkeys(self): 
        res = []
        
        for k, v in self.lkeys_to_avg.items(): 
            res.append(k)
            res = res + v
            
        return list(set(res))

    
    def run(self, dframe, dirname, fname):
        self.last_step = 0
        plotdir = self.output_dir + dirname + "plots/"
        if "avg_across_signals" in self.pipeline: 
            avg_index = self.pipeline.index("avg_across_signals")
            preavg_pipeline = self.pipeline[:avg_index]
            postavg_pipeline = self.pipeline[avg_index+1:]
            

            pre, ndata = self.filter_and_preavg(preavg_pipeline, dframe)
            avg = self.avg_across_signals(pre)
            pdata, n = self.postavg(postavg_pipeline, avg)

        else:
            pdata, ndata = self.filter_and_preavg(self.pipeline, dframe)
        
        self.make_plots(plotdir, fname)
        return pdata, ndata

    
    def make_plots(self, plotdir, fname): 
        if not os.path.exists(plotdir):
            os.makedirs(plotdir)
   
        if self.make_plots_bool: 
            for lkey in self.lkeys_to_plot: 
                self.plot_tracker[str(lkey)][0]["after"] = self.plot_tracker[str(lkey)][self.last_step]["after"]
                self.plot_res(self.plot_tracker[str(lkey)], str(lkey) + fname, plotdir)

    def postavg(self, pipeline, data):
        pdata = {}
        ndata = {}
        for lkey, signal in data.items():
            n, p, i = self.process_signal(signal, pipeline, lkey)
            pdata[lkey] = p
            ndata[lkey] = n
        if len(pipeline) > 0: 
            self.last_step = self.last_step + i + 1
        return pdata, ndata
    
    def process_row(self, row, pipeline):
        l = row["Data"].replace("]", "").replace("[", "").split(",")
        l = [float(i) for i in l]
        lkey = row["Landmark_key"]
        self.plot_tracker[lkey] = {
            0 : {
                "type" : "Entire pipeline", 
                "before" : l, 
                "after" : []
            }
        }
        p, n, i = self.process_signal(l, pipeline, lkey)
        return p, n, i, lkey

    def filter_and_preavg(self, pipeline, dframe): 
        pdata = {}
        ndata = {}
        i = 0
        for index, row in dframe.iterrows():
            if self.should_filter:
                if (any(str(lkey) == row["Landmark_key"] for lkey in self.filter_lkeys)): 
                    p, n, i, lk = self.process_row(row, pipeline)
                    pdata[lk] = p
                    ndata[lk] = n 

                    
            else: 
                p, n, i, lk = self.process_row(row, pipeline)
                pdata[lk] = p
                ndata[lk] = n 

        if len(pipeline) > 0: 
            self.last_step = self.last_step + i + 1
       
        return pdata, ndata
         
                
    def process_signal(self, raw, pipeline, pair): 
        normed_raw = self.min_max_norm(raw)
        curr_signal = raw
        
        i = 0
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
       
        return curr_signal, normed_raw, i
        
    
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


    def avg_one_pair(self, lkeys_toavg, data, lkey):
        averaged = data[str(lkey)]
        for toavg in lkeys_toavg: 
            nextrow_data = data[str(toavg)]
            averaged = [(g + h) / 2 for g, h in zip(averaged, nextrow_data)]

        return averaged


    def avg_across_signals(self, original):
        res = {}
        self.last_step += 1
        for lkey, lkeys in self.lkeys_to_avg.items():
            res[str(lkey)] = self.avg_one_pair(lkeys, original, lkey)
            self.plot_tracker[str(lkey)][self.last_step] = {
                "type" : "avg_across_signals", 
                "before" : original[str(lkey)], 
                "after" : res[str(lkey)]
            }

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







