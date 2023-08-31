    def analyze_results(self, tocomparedir, tocomparefiles, titles, xlabels):
        filedata = {}
  
        for f in tocomparefiles:
            
            filedata[f] = {
                "index" : [],
                "xlabels" : [], 
                "avgs" : [], 
                "pairs" : {}
            }
            
            index = 0
            for directory in tocomparedir: 
                index += 1
                fi = open("./correlation_reports/" + directory + f, 'r') 
                lines = fi.readlines()
                dirname = directory.split("/")[-2] 
                filedata[f]["xlabels"].append(xlabels[dirname])
                filedata[f]["index"].append(index)
                avgs = []
                filedata[f]["pairs"][dirname] = {}
                for line in lines: 
                    if any(str(pair) in line for pair in self.target_pairs):
                        words = line.split(" ")
                        num1 = int(words[0].replace("(","").replace(",",""))
                        num2 = int(words[1].replace(")",""))
                        t = (num1, num2)
                        avg = float(words[10])
                        filedata[f]["pairs"][dirname][t] = avg
                        avgs.append(avg)

                filedata[f]["avgs"].append(avgs)
            
            self.box_and_whisker(filedata, f, titles)
        
        return filedata
    
    def box_and_whisker(self, data, f, titles): 
        plt.clf()
        plt.boxplot(data[f]["avgs"])
        plt.xticks(data[f]["index"], data[f]["xlabels"], rotation=80)
        plt.suptitle(titles[f])
        fname = f.replace(".txt", "")
        plt.savefig(fname, bbox_inches='tight')  
              
    
    def scatter_plot(self, tocomparefiles, dirs, titles, markers, data):
        plt.clf()
        pairdata = {dirs[0] : {}, dirs[1] : {}}
        tempdata = {dirs[0] : [], dirs[1] : []}
    
        for i, d in enumerate(dirs):
            simavg = {}
            diffavg = {}
            for f in tocomparefiles: 
                for tup, avg in data[f]["pairs"][d].items():
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
                plt.plot(v, diff, marker=markers[d], label=k)

       
        plt.xlabel("Pearson Correlation for Similarity Test Cases")

        plt.ylabel("Pearson Correlation for Differences Test Cases")
        # handles, labels = plt.gca().get_legend_handles_labels()
        # by_label = dict(zip(labels, handles))
        plt.legend(bbox_to_anchor=(1.0, 2),loc="upper left")
        plt.savefig("scatterplottest2", bbox_inches='tight')  
            