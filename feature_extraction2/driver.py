from driver_config import *
from video_processing.mp_extractor import MPFeatureExtractor 
from signal_processing.signalprocessor import SignalProcessor
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
    dirs: dict
        A dict of root directory names
    landmarks: Landmarks object 
        A Landmarks class object that contains attributes and methods for naming, sorting, and selecting landmarks
    output_dir: string
        The name of the root directory where all output files from one run will be stored 

    Methods
    -------
    run_vid_processing(): Runs video processing for each video that's specified in config
    write_data_to_csv(data, dirname, filename): Writes data into csv file between pipeline stages

    
    '''
    def __init__(self, vid_processing_settings, sig_processing_settings, directories): 
        self.vidp_settings =  vid_processing_settings
        self.sigp_settings = sig_processing_settings
        self.dirs = directories
        self.landmarks = Landmarks()
        self.output_dir = directories["output root"] + str(Run["number"]) + "/"
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        shutil.copyfile("./driver_config.py", self.output_dir+"config_copy.py")

    def run_vid_processing(self): 
        """
        runs video processing on each video and stores output data in csv files
        """
        videos = self.vidp_settings["videos"]
        root_path = self.dirs["video input root"]
        landmarks = self.landmarks.get_landmarks_by_keyword(self.vidp_settings["key regions"])
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
                pairs_to_plot = self.vidp_settings["pairs to plot"],
                generate_video = self.vidp_settings["generate video?"],
                norm_approach = self.vidp_settings["norm by"],
                draw_landmark_nums = self.vidp_settings["draw landmark nums?"], 
                display_dim = self.vidp_settings["display dim"], 
                output_data = self.vidp_settings["output data"],
                draw_anchor_target_connector = self.vidp_settings["draw anchor target connector?"],
                output_directory = self.output_dir + "videos/")

            landmark_coords, landmark_single, landmark_groups = app.run()
            pathsplit = vid.split('/')
            self.write_data_to_csv(landmark_coords, pathsplit[1], "coords", self.dirs["vid data output"])
            self.write_data_to_csv(landmark_single, pathsplit[1], "landmark_single", self.dirs["vid data output"])
            self.write_data_to_csv(landmark_groups, pathsplit[1], "landmark_groups", self.dirs["vid data output"])

    def run_signal_processing(self): 
        output_dir = self.output_dir + self.dirs["sig processing data output"] 
        
        signalprocessor = SignalProcessor(
            self.sigp_settings["make plots for"], 
            self.sigp_settings["should filter?"], 
            self.sigp_settings["pipeline"], 
            self.sigp_settings["moving avg window"], 
            self.sigp_settings["butter settings"], 
            output_dir, 
            self.sigp_settings["pairs to avg"])
 
        for dirname in self.sigp_settings["files"]:
    
            dframe = self.read_data_from_csv(dirname, self.sigp_settings["datatype"], self.dirs["vid data output"])
            processed, normed = signalprocessor.run(dframe, dirname, self.sigp_settings["datatype"])    
            path = dirname.replace("/","")
            self.write_data_to_csv(processed, path, "processed", self.dirs["sig processing data output"]) 
            self.write_data_to_csv(processed, path, "normed", self.dirs["sig processing data output"]) 


    
    def run_signal_comparison(self): 
        for sim in self.sim_test_cases: 
            best_sim_list, best_sim_pairs, sim_total_count, sim_total_score = self.run_test_cases(sim, self.top_cutoff, self.top_num, reverse=True)
        
        for diff in self.diff_test_cases: 
            best_diff_list, best_diff_pairs, diff_total_count, diff_total_score = self.run_test_cases(diff, self.bottom_cutoff, self.bottom_num, reverse=False) 
    
    def run_pipeline_comparison(self): 
        pass
    
    def find_best_pairs(self): 
        pass 

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
    
        full_path = self.output_dir + rootfolder + dirname + fname + ".csv"
        df = pd.read_csv(full_path, header=None)
        df = df.rename({0:"Landmark_key", 1:"Raw_data"}, axis="index")
        df = df.T

        return df
        
  
     
    
def main(): 
    driver = Driver(VideoProcessing, SignalProcessing, Directories)
    if Run["video processing?"]: driver.run_vid_processing()
    if Run["signal processing?"]: driver.run_signal_processing()
    if Run["signal comparison?"]: driver.run_signal_comparison()
    if Run["pipeline comparison?"]: driver.run_pipeline_comparison()
    if Run["find best pairs?"]: driver.find_best_pairs()

if __name__ == "__main__":
    main()