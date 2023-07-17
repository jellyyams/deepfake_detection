import numpy as np
import matplotlib.pyplot as plt

class SignalProcessesor:
    def __init__(self):
        pass
    
    def simple_moving_avg(self, signal, window_size, title=""):
        i = 0
        res = []

        while i < len(signal) - window_size + 1:
            window_avg = round(np.sum(signal[i:i+window_size]) / window_size, 2)
            res.append(window_avg)
            i += 1

        # self.plot_before_after(signal, res, title + "_Simple_Moving_Average")

        
        return res 

    def min_max_norm(self, distances): 
        smallest = min(distances)
        largest = max(distances)
        d_scaled = [(x - smallest)/(largest - smallest) for x in distances]

        return d_scaled
    
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






