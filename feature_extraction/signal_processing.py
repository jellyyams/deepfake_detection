import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal 

class SignalProcessesor:
    def __init__(self, window):
        self.window = window
    
    def butterworth(self, raw):
        fs = 30
        fc = 4
        w = fc / (fs / 2)
        b, a = signal.butter(5, w, 'low')
        output = signal.filtfilt(b, a, raw)
        # self.plot_before_after(raw, output, "butterworth")
        return output

    
    def process_signal(self, raw, processtype): 
        normed = self.min_max_norm(raw)
        if processtype == "moving_average": 
            processed = self.simple_moving_avg(raw, self.window)
            nprocessed = self.min_max_norm(processed)
            return normed, nprocessed
        elif processtype == "normalize": 
            return normed, normed
        elif processtype == "butterworth":
            return normed, self.butterworth(raw)
        else:
            return normed, raw
        
    
    def simple_moving_avg(self, signal, window_size, title=""):
        i = 0
        res = []

        while i < len(signal) - window_size + 1:
            window_avg = round(np.sum(signal[i:i+window_size]) / window_size, 6)
            res.append(window_avg)
            i += 1

        # self.plot_before_after(signal, res, title + "_Simple_Moving_Average")

        return res 

    def min_max_norm(self, distances): 
        smallest = min(distances)
        largest = max(distances)
        d_scaled = [(x - smallest)/(largest - smallest) for x in distances]

        return d_scaled


    def average_one_pair(self, pairs_toavg, row1_data, pd_df):
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
                    averaged = self.average_one_pair(toavg_pairs, row["Processed"], pd_df)
                

                    normed, p = self.process_signal(averaged, postavg_process)
    
                    avg_data["Landmark_key"].append(str(tup))
                    avg_data["Distances"].append(row["Distances"])
                    avg_data["Normalized"].append(row["Normalized"])
                    avg_data["Processed"].append(p)
                    
            avg_df = pd.DataFrame(avg_data)
            res[fname] = avg_df
                    
        return res


    
    def plot_before_after(self, before, after, title):
        plt.clf()
        plt.plot(before, label="before")
        plt.plot(after, label="after")
        plt.legend()
        plt.suptitle(title)
        plt.savefig("plots/" + title)


    def plot_test(self, data1, data2, averaged, title):
        plt.clf()
        plt.plot(data1, label="data1")
        plt.plot(data2, label="data2")
        plt.plot(averaged, label="averaged")
        print(len(data1), len(data2), len(averaged))
        plt.legend()
        plt.suptitle(title)
        plt.savefig("plots/" + title)






