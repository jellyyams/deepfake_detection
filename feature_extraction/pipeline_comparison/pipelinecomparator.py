import pandas as pd 
import matplotlib.pylab as plt
import os
import scipy.stats as stats
from signal_comparison.artifactgenerator import ArtifactGenerator


class PipelineComparator: 
    '''
    A class to generate plots that compare results across different pipelines/runs

    Attributes: 
    -----------
    baw_run_dirs: list
        list of directories that contain each run information, to be used in box and whisker plots
    fpaths: list
        list of file paths to be used in box and whisker plots 
    res: dict
        dict keeping track of results from pipeline comparison
    input_root: string
        name of root directory we're reading data from
    output_dir: string
        name of root directory we're outputing data to 
    baw_xlabels: dict
        dict of box and whisker x axis labels 
    baw_titles: dict
        dict of box and whisker plot titles
    lkeys_to_plot: list
        list of lkeys that will be included in box and whisker plots
    
    '''
    def __init__(self, baw_run_dirs, fpaths, input_root, output_dir, baw_xlabels, baw_titles, lkeys_to_plot, scatter_dirs, scatter_markers): 
        self.baw_run_dirs = baw_run_dirs
        self.fpaths = fpaths
        self.res = {}
        self.input_root = input_root
        self.output_dir = output_dir
        self.baw_xlabels = baw_xlabels
        self.baw_titles = baw_titles
        self.lkeys_to_plot = lkeys_to_plot
        self.scatter_dirs = scatter_dirs
        self.scatter_markers = scatter_markers

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def run(self):
  
        for fpath in self.fpaths:
            
            self.res[fpath] = {
                "baw index" : [],
                "baw xlabels" : [], 
                "avgs" : [], 
                "lkeys" : {}
            }
            
            for index, directory in enumerate(self.baw_run_dirs): 
                self.extract_data(directory, index, fpath)
            
            self.box_and_whisker(self.res, fpath)
            #self.scatter_plot()
    
    def extract_data(self, directory, index, fpath): 
        fi = open(self.input_root + directory + fpath, 'r') 
        lines = fi.readlines()
        dirname = directory.replace("/","") 
        self.res[fpath]["baw xlabels"].append(self.baw_xlabels[dirname])
        self.res[fpath]["baw index"].append(index + 1)
        avgs = []
        self.res[fpath]["lkeys"][dirname] = {}
        for line in lines: 
            if any(str(lkey) in line for lkey in self.lkeys_to_plot):
                words = line.split(" ")
                num1 = int(words[0].replace("(","").replace(",",""))
                num2 = int(words[1].replace(")",""))
                t = (num1, num2)
                avg = float(words[10])
                self.res[fpath]["lkeys"][dirname][t] = avg
                avgs.append(avg)
        
        self.res[fpath]["avgs"].append(avgs)

    
    def box_and_whisker(self, data, f): 
        plt.clf()
        plt.boxplot(data[f]["avgs"])
        plt.xticks(data[f]["baw index"], data[f]["baw xlabels"], rotation=80)
        plt.suptitle(self.baw_titles[f])
        fname = f.replace(".txt", "").split("/")[-2]
        plt.savefig(self.output_dir + fname, bbox_inches='tight')  
              
    
    def scatter_plot(self):
        '''
        todo. Not completed testing/implementing yet 
        '''
        plt.clf()
        pairdata = {self.scatter_dirs[0] : {}, self.scatter_dirs[1] : {}}
        tempdata = {self.scatter_dirs[0] : [], self.scatter_dirs[1] : []}
    
        for i, d in enumerate(self.scatter_dirs):
            d = d.replace("/","")
            simavg = {}
            diffavg = {}
            for f in self.fpaths: 
                for tup, avg in self.res[f]["lkeys"][d].items():
                    if "angle" in f: 
                        simavg[tup] = avg
                    elif "utterance" in f:
                        diffavg[tup] = avg
                    # elif tup in diffavg:
                    #     diffavg[tup] = float((diffavg[tup] + avg) / 2)
                    # else: 
                    #     diffavg[tup] = avg
            
            for k, v in simavg.items():
                diff = diffavg[k]
                pairdata[d][k] = (v, diff)
                tempdata[d].append((v, diff))
                plt.plot(v, diff, marker=self.scatter_markers[d], label=k)

       
        plt.xlabel("Pearson Correlation for Similarity Test Cases")
        plt.ylabel("Pearson Correlation for Differences Test Cases")
   
        plt.legend(bbox_to_anchor=(1.0, 2),loc="upper left")
        plt.savefig(self.output_dir + "scatterplot", bbox_inches='tight')  
            