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
            