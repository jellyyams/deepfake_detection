from driver_config import *
from video_processing.mp_extractor import MPFeatureExtractor 
from signal_processing.signalprocessor import SignalProcessor
from signal_comparison.signalcomparator import SignalComparator
from pipeline_comparison.pipelinecomparator import PipelineComparator
from landmarks import Landmarks
import os
import csv
import pandas as pd
import shutil 

class Driver: 
    '''
    A class that contains the main functions for running pipeline

    Attributes
    ----------
    vidp_settings: dict
        A dict of configuration settings for running the video processing stage of pipeline
    sigp_settings: dict
        A dict of configuration settings for running the signal processing stage of pipeline
    sigc_settings: dict
        A dict of configuration settings for running the signal comparison stage of pipeline
    pipelinec_settings: dict
        A dict of configuration settings for running the pipeline comparison stage of pipeline
    landmark_settings: dict
        A dict of configuration settings for landmarks
    dirs: dict
        A dict of root directory names
    landmarks: Landmarks object 
        A Landmarks class object that contains attributes and methods for naming, sorting, and selecting landmarks
    output_dir: string
        The name of the root directory where all output files from one run will be stored 
    root: string
        The name of root directory where all run dirs are stored 

    Methods
    -------
    run_vid_processing(): Runs video processing for each video that's specified in config
    run_signal_processing(): reads from video processing output data, runs SignalProcessor on raw data, and stores results in csv files
    run_signal_comparison(): reads from either raw video processing output or signal processing data and runs test cases and/or makes plots
    run_pipeline_comparison(): reads report.txt files generated during signal comparison test cases and creates box and whisker plot comparing pipelines 
    write_data_to_csv(data, dirname, filename, rootfolder): Writes data into csv file between pipeline stages
    read_data_from_csv(dirname, fname, rootfolder): Reads csv files into pandas dataframes and returns the dataframe
    
    '''
    def __init__(self, vid_processing_settings, sig_processing_settings, sig_comparison_settings, directories, landmark_settings, pipeline_comparison_settings): 
        self.vidp_settings =  vid_processing_settings
        self.sigp_settings = sig_processing_settings
        self.sigc_settings = sig_comparison_settings
        self.pipelinec_settings = pipeline_comparison_settings
        self.landmark_settings = landmark_settings
        self.dirs = directories
        self.landmarks = Landmarks()
        self.root = directories["root"]
        self.output_dir = self.root + "run_number_" + str(Run["number"]) + "/"
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        shutil.copyfile("./driver_config.py", self.output_dir+"config_copy.py")

    def run_vid_processing(self): 
        """
        runs video processing on each video and stores output data in csv files
        """
        videos = self.vidp_settings["videos"]
        root_path = self.dirs["video input root"]
        landmarks = self.landmarks.get_landmarks_by_keyword(self.landmark_settings["key regions"])
        pairs_to_analyze = self.landmarks.generate_landmark_pairs(landmarks)

        for vid in videos: 
            input_path = root_path + vid + ".mp4"
            app = MPFeatureExtractor(
                input_path,
                draw_all_landmarks= self.vidp_settings["draw all landmarks?"],
                initial_detect = self.vidp_settings["initial detect?"],
                anchor_landmark = self.vidp_settings["anchor landmark"],
                target_landmarks = landmarks,
                pairs_to_analyze = pairs_to_analyze,
                pairs_to_plot = self.landmark_settings["lkeys to plot"],
                generate_video = self.vidp_settings["generate video?"],
                norm_approach = self.vidp_settings["norm by"],
                draw_landmark_nums = self.vidp_settings["draw landmark nums?"], 
                display_dim = self.vidp_settings["display dim"], 
                output_data = self.vidp_settings["output data"],
                draw_anchor_target_connector = self.vidp_settings["draw anchor target connector?"],
                output_directory = self.output_dir + "videos/")

            landmark_coords, landmark_single, landmark_groups = app.run()
            pathsplit = vid.split('/')
            self.write_data_to_csv(landmark_coords, pathsplit[1], "coords", self.dirs["vid processing output"])
            self.write_data_to_csv(landmark_single, pathsplit[1], "landmark_single", self.dirs["vid processing output"])
            self.write_data_to_csv(landmark_groups, pathsplit[1], "landmark_groups", self.dirs["vid processing output"])

    def run_signal_processing(self): 
        """
        runs signal processing on video data
        """
        output_dir = self.output_dir + self.dirs["sig processing output"] 
        
        signalprocessor = SignalProcessor(
            self.sigp_settings["make plots?"], 
            self.sigp_settings["should filter?"], 
            self.sigp_settings["pipeline"], 
            self.sigp_settings["moving avg window"], 
            self.sigp_settings["butter settings"], 
            output_dir, 
            self.landmark_settings["lkeys to avg"],
            self.landmark_settings["lkeys to plot"])
 
        for dirname in self.sigp_settings["files"]:
    
            dframe = self.read_data_from_csv(dirname, self.sigp_settings["datatype"], self.dirs["vid processing output"])
            processed, normed = signalprocessor.run(dframe, dirname, self.sigp_settings["datatype"])    
            path = dirname.replace("/","")
            self.write_data_to_csv(processed, path, "processed", self.dirs["sig processing output"]) 
            self.write_data_to_csv(normed, path, "normed", self.dirs["sig processing output"]) 
            print("completed signal processing on file: " + dirname)
    
    def run_signal_comparison(self): 
        """
        runs signal comparison on either raw video data or processed signal data
        """
   
        pdata = {} #processed data
        odata = {} #original, unprocessed data
        ndata = {} #normed data 
        for tc_name, testset in self.sigc_settings["test sets"].items():
            for k, v in testset.items():
                for x in v: 
                    if x not in odata: 
                        odata[x] = self.read_data_from_csv(x, "landmark_groups", self.dirs["vid processing output"])
                        if os.path.exists(self.output_dir + self.dirs["sig processing output"]): #if signal processing was run before this, then this dir exists 
                            pdata[x] = self.read_data_from_csv(x, "processed", self.dirs["sig processing output"])
                            ndata[x] = self.read_data_from_csv(x, "normed", self.dirs["sig processing output"])

        output_dir = self.output_dir + self.dirs["sig comparison output"] 

        signalcomparator = SignalComparator(
            odata, 
            pdata, 
            ndata,
            output_dir, 
            self.sigc_settings["test sets"], 
            self.sigc_settings["top cutoff"], 
            self.sigc_settings["bottom cutoff"], 
            self.sigc_settings["top num"], 
            self.sigc_settings["bottom num"], 
            self.sigc_settings["make plots?"], 
            self.sigc_settings["run test sets?"],
            self.sigc_settings["use processed?"], 
            self.landmark_settings["lkeys to plot"], 
            self.sigc_settings["vids to plot"], 
            self.sigc_settings["plot labels"]
        )

        signalcomparator.run()
        
    def run_pipeline_comparison(self): 
        """
        runs pipeline comparison across different run folders
        """
        output_dir = self.root + self.dirs["pipeline comparison output"]
        pipelinecomparator = PipelineComparator(
            self.pipelinec_settings["baw directories"], 
            self.pipelinec_settings["fpaths"], 
            self.root, 
            output_dir, 
            self.pipelinec_settings["baw xlabels"], 
            self.pipelinec_settings["baw titles"], 
            self.landmark_settings["lkeys to plot"], 
            self.pipelinec_settings["scatter dirs"], 
            self.pipelinec_settings["scatter markers"], 

        )
        pipelinecomparator.run()

    def write_data_to_csv(self, data, dirname, fname, rootfolder):
        """
        writes dict into csv format
        """
        
        directory = self.output_dir + rootfolder + str(dirname) + "/"

        if not os.path.exists(directory):
            os.makedirs(directory)
        
        with open(directory + fname + ".csv", 'w') as f:
            writer = csv.DictWriter(f, data.keys())
            writer.writeheader()
            writer.writerow(data)

    def read_data_from_csv(self, dirname, fname, rootfolder):
        """
        reads csv data and returns a dataframe
        """
    
        full_path = self.output_dir + rootfolder + dirname + fname + ".csv"
        df = pd.read_csv(full_path, header=None)
        df = df.rename({0:"Landmark_key", 1:"Data"}, axis="index")
        df = df.T

        return df
     
def main(): 
    driver = Driver(VideoProcessing, SignalProcessing, SignalComparison, Directories, LandmarkSettings, PipelineComparison)
    if Run["video processing?"]: driver.run_vid_processing()
    if Run["signal processing?"]: driver.run_signal_processing()
    if Run["signal comparison?"]: driver.run_signal_comparison()
    if Run["pipeline comparison?"]: driver.run_pipeline_comparison()

if __name__ == "__main__":
    main()