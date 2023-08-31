import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal 
import os
import math

class SignalProcessor:
    '''
    A class that processes raw video extracted data and outputs processed signals

    Attributes
    ----------
    make_plots_bool: bool
        boolean for whether to create plots or not
    lkeys_to_plot: list
        landmark keys that will be plotted of make_plots_bool is True
    pipeline: list
        list of strings that indicate the sequence of processing steps for each signal 
    should_filter: boolean
        boolean for whether to filter out lkeys for faster run time
    movingavg_window: int
        the size of window when running moving average smoothing 
    butter_settings: dict
        a dict of settings to be used when running butterworth filtering
    lkeys_to_avg: dict
        dict of landmark keys (as keys) and their corresponding lists of lkeys to average their signal with
        for instance if lkeys_to_avg looked like this
            {
                (336, 384) : [(336, 385), (296, 384)], 
                (296, 385) : [(296, 386), (334, 386)],
            }
        and "avg_across_signals" was in self.pipeline, then the signal corresponding to (336, 384) would be averaged 
        with (336, 385) and (296, 384). The same would be true for (296, 385) etc. 
    filter_lkeys: list
        list of lkeys that will be kept/processed. all other lkeys will be included in future stages of pipeline
    output_dir: string
        string for output directory where all output plots will be written to 
    plot_tracker: dict
        dict for keeping track of data that will be used when generating plots. This dict stores the state of each signal at every step of the processing pipeline
    last_step: int
        tracks the most recent last step of the pipeline. The first string in self.pipeline is step 1


        
    '''
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
        '''
        collects all the landmark keys that will NOT be filtered out
        '''
        res = []
        
        for k, v in self.lkeys_to_avg.items(): #if the lkey is used in averaging across signals
            res.append(k)
            res = res + v
        
        for lkey in self.lkeys_to_plot: #if the lkey is used in plotting
            res.append(lkey)
            
        return list(set(res))

    
    def run(self, dframe, dirname, fname):
        '''
        the main function that runs this class. It accomplishes two main steps: signal processing and plotting
        '''
        self.last_step = 0
        plotdir = self.output_dir + dirname + "plots/"
        #Step one: signal processing 
        if "avg_across_signals" in self.pipeline: 
            #if you're averaging across signals then the pipeline is broken up into pre averaging and post averaging
            avg_index = self.pipeline.index("avg_across_signals")
            preavg_pipeline = self.pipeline[:avg_index]
            postavg_pipeline = self.pipeline[avg_index+1:]
            

            pre, ndata = self.filter_and_preavg(preavg_pipeline, dframe) #pre averaging
            avg = self.avg_across_signals(pre) #average
            pdata, n = self.postavg(postavg_pipeline, avg) #post averaging

        else: #if you're not averaging across signals, then each lkey can be processed individually from start to end
            pdata, ndata = self.filter_and_preavg(self.pipeline, dframe)
        
        #Step two: making plots. This must be run sequentially after step 1
        if self.make_plots_bool: 
            self.make_plots(plotdir, fname)
        
        return pdata, ndata #returns the processed data and the normed data 

    
    def make_plots(self, plotdir, fname): 
        '''
        makes line plots for each lkey in lkeys_to_plot
        '''
        if not os.path.exists(plotdir):
            os.makedirs(plotdir)
  
        for lkey in self.lkeys_to_plot: 
            self.plot_tracker[str(lkey)][0]["after"] = self.plot_tracker[str(lkey)][self.last_step]["after"]
            self.plot_res(self.plot_tracker[str(lkey)], str(lkey) + fname, plotdir)

    def postavg(self, pipeline, data):
        '''
        Runs the post averaging pipeline
        '''
        pdata = {}
        ndata = {}
        for lkey, signal in data.items():
            n, p, i = self.process_signal(signal, pipeline, lkey)
            pdata[lkey] = p
            ndata[lkey] = n

        if len(pipeline) > 0: #updates to the most recent last step 
            self.last_step = self.last_step + i + 1
        return pdata, ndata
    
    def process_row(self, row, pipeline):
        '''
        processes a single row from the dataframe that was originally passed in from reading the csv files
        '''

        l = row["Data"].replace("]", "").replace("[", "").split(",") #converts the string of a list into a list
        l = [float(i) for i in l] #converts each list item from a string to a float 
        lkey = row["Landmark_key"]
        self.plot_tracker[lkey] = {
            0 : { #key 0 is used to store the before and after of the entire pipeline
                "type" : "Entire pipeline", 
                "before" : l, #original unprocessed data 
                "after" : []
            }
        }
        p, n, i = self.process_signal(l, pipeline, lkey)
        return p, n, i, lkey

    def filter_and_preavg(self, pipeline, dframe): 
        '''
        Always the first step run, regardless of whether averaging across signals is in pipeline
        Filters out lkeys if filtering is turned on and runs the pre averaging pipeline
        '''
        pdata = {}
        ndata = {}
        i = 0
        for index, row in dframe.iterrows():
            #if not filtering then include row in final data 
            #if filtering, only include row if lkey is in self.filter_lkeys 
            if not self.should_filter or (any(str(lkey) == row["Landmark_key"] for lkey in self.filter_lkeys)):
                p, n, i, lk = self.process_row(row, pipeline)
                pdata[lk] = p
                ndata[lk] = n 
          
        if len(pipeline) > 0: #as long as pipeline was run, update last_step to most recent step
            self.last_step = self.last_step + i + 1
       
        return pdata, ndata
         
                
    def process_signal(self, raw, pipeline, lkey): 
        '''
        runs a signal through pipeline 

        parameters:
        ----------
        raw: list of floats, the original signal
        pipeline: list of strings, the sequence of processing steps that the raw signal should run through 
        lkey: The landmark key that identifies the signal being processed

        '''
        normed_raw = self.min_max_norm(raw)
        curr_signal = raw
        
        i = 0
        for i, ptype in enumerate(pipeline):
            self.plot_tracker[lkey][self.last_step + i + 1] = {"type" : ptype, "before":curr_signal}
            if ptype == "moving_average": 
                curr_signal = self.simple_moving_avg(curr_signal)
            elif ptype == "normalize": 
                curr_signal = self.min_max_norm(curr_signal)
            elif ptype == "butterworth":
                curr_signal = self.butterworth(curr_signal)
            else:
                #implement throw error
                print("please revisit configuration file and enter a valid signal processing type")
            self.plot_tracker[lkey][self.last_step + i + 1]["after"] = curr_signal
       
        return curr_signal, normed_raw, i
        
    
    def butterworth(self, original):
        '''
        Butterworth frequency filtering
        '''
      
        w = self.butter_settings["fc"] / (self.butter_settings["fs"]  / 2)
        b, a = signal.butter(self.butter_settings["cutoff"], w, self.butter_settings["type"])
        output = signal.filtfilt(b, a, original)

        return output
    
    def simple_moving_avg(self, original):
        '''
        moving average smotthing
        '''
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
        '''
        min max normalization. Note that this can't be included in the final signal processsing pipeline because 
        it relies on identifying the min and max of the entire signal (which can't happen in real time)
        '''
        smallest = min(original)
        largest = max(original)
        d_scaled = [(x - smallest)/(largest - smallest) for x in original]

        return d_scaled


    def avg_signal(self, lkeys_toavg, data, lkey):
        '''
        Averaging all the signals in lkeys_toavg
        '''
        averaged = data[str(lkey)]
        for toavg in lkeys_toavg: 
            nextrow_data = data[str(toavg)]
            averaged = [(g + h) / 2 for g, h in zip(averaged, nextrow_data)]

        return averaged


    def avg_across_signals(self, original):
        '''
        Averages across signals
        '''
        res = {}
        self.last_step += 1
        for lkey, lkeys in self.lkeys_to_avg.items():
            res[str(lkey)] = self.avg_signal(lkeys, original, lkey)
            self.plot_tracker[str(lkey)][self.last_step] = {
                "type" : "avg_across_signals", 
                "before" : original[str(lkey)], 
                "after" : res[str(lkey)]
            }

        return res


    def plot_res(self, data, title, plotdir):
        '''
        Plots the signal at each stage of processing
        '''

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







