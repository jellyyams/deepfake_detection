Run = {
   "number" : 1, 
   "video processing?" : False,
   "signal processing?" : True,
   "signal comparison?" : True,
   "pipeline comparison?" : True, 
}

LandmarkSettings = {
   "lkeys to plot": [
      (336, 384), 
      (296, 385), 
      (255, 257),
      (258, 327), 
      (25, 159), 
      (61, 160), 
      (24, 159),
      (287, 327), 
      (98, 292), 
      (334, 386), 
      (293, 387), 
      (13, 321), 
      (375, 409), 
      (311, 375), 
      (270, 321), 
      (81, 181), 
      (82, 321), 
      (0, 405)
   ], 
   "lkeys to avg" : {
      (336, 384) : [(336, 385)], 
      (296, 385) : [(296, 386)], 
      (255, 257) : [(257, 339)],
      (258, 327) : [(257, 327)], 
      (25, 159) : [(110, 159)], 
      (61, 160): [(61, 159)], 
      (24, 159) : [(110, 159)],
      (287, 327) : [(291, 327)], 
      (98, 292) : [(98, 308)], 
      (334, 386) : [(334, 387)], 
      (293, 387) : [(293, 388)], 
      (13, 321) : [(312, 321)], 
      (375, 409) : [(321, 409)], 
      (311, 375) : [(311, 321)], 
      (270, 321) : [(269, 321)], 
      (81, 181) : [(82, 181)], 
      (82, 321) : [(82, 405)], 
      (0, 405) : [(0, 321)]
   }

}

Directories = {
   "video input root" : "../../../Desktop/Deepfake_Detection/Test_Videos/", 
   "root" : "./results/", 
   "vid processing output" : "vidprocessing_data/", 
   "sig processing output" : "sigprocessing_data/", 
   "sig comparison output" : "sigcomparison_data/",
   "pipeline comparison output" : "pipelinecomp_data/"
}

VideoProcessing = {
   "initial detect?" : True,
   "draw all landmarks?" : True,
   "generate video?": False,
   "output data" : ["pairwise_distances", "anchor_distances"],
   "anchor landmark" : 5, 
   "display dim" : 800, 
   "draw landmark nums?" : False, 
   "draw anchor target connector?" : False,
   "key regions" : ["Inner", "Outer", "1", "Corner", "0", "Eyebrow"],
   "norm by" : "first_upper_lower_bbox", #vs "first_quarters_bbox" vs "face_bbox" vs "none" vs "upper_lower_bbox" vs "first_face_bbox" vs "first_upper_lower_bbox"
   "videos" : [
      "Kelly_Front/kelly_front_s1_v2",
      "Kelly_Low/kelly_low_s1_v2",
      "Kelly_Right_Threequarter/kelly_right_threequarter_s1_v2",
      "Kelly_Left_Threequarter/kelly_left_threequarter_s1_v2",
      "Kelly_Front/kelly_front_s2_v2",
      "Kelly_Low/kelly_low_s2_v2",
      "Kelly_Right_Threequarter/kelly_right_threequarter_s2_v2",
      "Kelly_Left_Threequarter/kelly_left_threequarter_s2_v2",
      "Kelly_Front/kelly_front_s3_v2",
      "Kelly_Low/kelly_low_s3_v2",
      "Kelly_Right_Threequarter/kelly_right_threequarter_s3_v2",
      "Kelly_Left_Threequarter/kelly_left_threequarter_s3_v2",
      "Kelly_Front/kelly_front_s4_v2",
      "Kelly_Low/kelly_low_s4_v2",
      "Kelly_Right_Threequarter/kelly_right_threequarter_s4_v2",
      "Kelly_Left_Threequarter/kelly_left_threequarter_s4_v2",
      "Kelly_Front/kelly_front_s27_v2",
      "Kelly_Low/kelly_low_s27_v2",
      "Kelly_Right_Threequarter/kelly_right_threequarter_s27_v2",
      "Kelly_Left_Threequarter/kelly_left_threequarter_s27_v2",
      "Kelly_Front/kelly_front_s28_v2",
      "Kelly_Low/kelly_low_s28_v2",
      "Kelly_Right_Threequarter/kelly_right_threequarter_s28_v2",
      "Kelly_Left_Threequarter/kelly_left_threequarter_s28_v2",
      "Hadleigh_Front/hadleigh_front_s1_v2",
      "Hadleigh_Low/hadleigh_low_s1_v2",
      "Hadleigh_Right_Threequarter/hadleigh_right_threequarter_s1_v2",
      "Hadleigh_Left_Threequarter/hadleigh_left_threequarter_s1_v2",
      "Hadleigh_Front/hadleigh_front_s2_v2",
      "Hadleigh_Low/hadleigh_low_s2_v2",
      "Hadleigh_Right_Threequarter/hadleigh_right_threequarter_s2_v2",
      "Hadleigh_Left_Threequarter/hadleigh_left_threequarter_s2_v2",
      "Hadleigh_Front/hadleigh_front_s3_v2",
      "Hadleigh_Low/hadleigh_low_s3_v2",
      "Hadleigh_Right_Threequarter/hadleigh_right_threequarter_s3_v2",
      "Hadleigh_Left_Threequarter/hadleigh_left_threequarter_s3_v2",
      "Hadleigh_Front/hadleigh_front_s4_v2",
      "Hadleigh_Low/hadleigh_low_s4_v2",
      "Hadleigh_Right_Threequarter/hadleigh_right_threequarter_s4_v2",
      "Hadleigh_Left_Threequarter/hadleigh_left_threequarter_s4_v2",
      "Hadleigh_Front/hadleigh_front_s27_v2",
      "Hadleigh_Low/hadleigh_low_s27_v2",
      "Hadleigh_Right_Threequarter/hadleigh_right_threequarter_s27_v2",
      "Hadleigh_Left_Threequarter/hadleigh_left_threequarter_s27_v2",
      "Hadleigh_Front/hadleigh_front_s28_v2",
      "Hadleigh_Low/hadleigh_low_s28_v2",
      "Hadleigh_Right_Threequarter/hadleigh_right_threequarter_s28_v2",
      "Hadleigh_Left_Threequarter/hadleigh_left_threequarter_s28_v2",
      
   ]
}

SignalProcessing = {
   "files" : [
      "kelly_front_s1_v2/",
      "kelly_low_s1_v2/",
      "kelly_right_threequarter_s1_v2/",
      "kelly_left_threequarter_s1_v2/",
      "kelly_front_s2_v2/",
      "kelly_low_s2_v2/",
      "kelly_right_threequarter_s2_v2/",
      "kelly_left_threequarter_s2_v2/",
      "kelly_front_s3_v2/",
      "kelly_low_s3_v2/",
      "kelly_right_threequarter_s3_v2/",
      "kelly_left_threequarter_s3_v2/",
      "kelly_front_s4_v2/",
      "kelly_low_s4_v2/",
      "kelly_right_threequarter_s4_v2/",
      "kelly_left_threequarter_s4_v2/",
      "kelly_front_s27_v2/",
      "kelly_low_s27_v2/",
      "kelly_right_threequarter_s27_v2/",
      "kelly_left_threequarter_s27_v2/",
      "kelly_front_s28_v2/",
      "kelly_low_s28_v2/",
      "kelly_right_threequarter_s28_v2/",
      "kelly_left_threequarter_s28_v2/",
      "hadleigh_front_s1_v2/",
      "hadleigh_low_s1_v2/",
      "hadleigh_right_threequarter_s1_v2/",
      "hadleigh_left_threequarter_s1_v2/",
      "hadleigh_front_s2_v2/",
      "hadleigh_low_s2_v2/",
      "hadleigh_right_threequarter_s2_v2/",
      "hadleigh_left_threequarter_s2_v2/",
      "hadleigh_front_s3_v2/",
      "hadleigh_low_s3_v2/",
      "hadleigh_right_threequarter_s3_v2/",
      "hadleigh_left_threequarter_s3_v2/",
      "hadleigh_front_s4_v2/",
      "hadleigh_low_s4_v2/",
      "hadleigh_right_threequarter_s4_v2/",
      "hadleigh_left_threequarter_s4_v2/",
      "hadleigh_front_s27_v2/",
      "hadleigh_low_s27_v2/",
      "hadleigh_right_threequarter_s27_v2/",
      "hadleigh_left_threequarter_s27_v2/",
      "hadleigh_front_s28_v2/",
      "hadleigh_low_s28_v2/",
      "hadleigh_right_threequarter_s28_v2/",
      "hadleigh_left_threequarter_s28_v2/",
   ], 
   "datatype" : "landmark_groups",  
   "pipeline" : ["avg_across_signals", "moving_average"], #moving_average, avg_across_signals, 
   "moving avg window" : 5, 
   "butter settings" : {
      "fs" : 30, 
      "fc" : 4, 
      "type" : "low", 
      "cutoff" : 5
   },
   "make plots?": False,
   "should filter?" : True
}

SignalComparison = {
   "use processed?" : True, #set to false if you want to run the test cases on unproessed data 
   "top cutoff" : 0.6, #range from -1 to 1, setting to -1 means nothing is cut off. Filters out all lkeys with sim scores lower than this cutoff
   "bottom cutoff" : 0.3, #range from 0 to 1, setting to 1 means nothing is cut off. Filters out all lkeys with diff scores greater than this cutoff
   "top num" : 150, #if using pipeline to find best performing landmarks, this filters out the top 150 performing pairs for similarity test cases
   "bottom num" : 2000, #if using pipeline to find best performing landmarks, this filters out the bottom 2000 performing pairs for similarity test cases
   "vids to plot" : [
      "kelly_front_s1_v2/", 
      "kelly_low_s1_v2/", 
      "kelly_right_threequarter_s1_v2/", 
      "kelly_left_threequarter_s1_v2/"
   ], 
   "plot labels" : {
        "kelly_front_s1_v2/" : "Speaker 1, front angle",
        "kelly_low_s1_v2/": "Speaker 1, low angle",
        "kelly_right_threequarter_s1_v2/": "Speaker 1, right angle",
        "kelly_left_threequarter_s1_v2/": "Speaker 1, left angle", 
    },
   "make plots?" : True,
   "run test sets?": False, 
   "test sets" : {
      "angle_test_cases" : 
      {
         "kelly_angles1" : ["kelly_front_s1_v2/", "kelly_low_s1_v2/", "kelly_right_threequarter_s1_v2/", "kelly_left_threequarter_s1_v2/"],
         "kelly_angles2" : ["kelly_front_s2_v2/", "kelly_low_s2_v2/", "kelly_right_threequarter_s2_v2/", "kelly_left_threequarter_s2_v2/"],
         "kelly_angles3" : ["kelly_front_s3_v2/", "kelly_low_s3_v2/", "kelly_right_threequarter_s3_v2/", "kelly_left_threequarter_s3_v2/"],
         "kelly_angles4" : ["kelly_front_s4_v2/", "kelly_low_s4_v2/", "kelly_right_threequarter_s4_v2/", "kelly_left_threequarter_s4_v2/"],
         "hadleigh_angles1" : ["hadleigh_front_s1_v2/", "hadleigh_low_s1_v2/", "hadleigh_right_threequarter_s1_v2/", "hadleigh_left_threequarter_s1_v2/"],
         "hadleigh_angles2" : ["hadleigh_front_s2_v2/", "hadleigh_low_s2_v2/", "hadleigh_right_threequarter_s2_v2/", "hadleigh_left_threequarter_s2_v2/"],
         "hadleigh_angles3" : ["hadleigh_front_s3_v2/", "hadleigh_low_s3_v2/", "hadleigh_right_threequarter_s3_v2/", "hadleigh_left_threequarter_s3_v2/"],
         "hadleigh_angles4" : ["hadleigh_front_s4_v2/", "hadleigh_low_s4_v2/", "hadleigh_right_threequarter_s4_v2/", "hadleigh_left_threequarter_s4_v2/"],
        
      },
      "identity_test_cases" :
      {
         "identities1" : ["kelly_front_s1_v2/", "hadleigh_front_s1_v2/"],
         "identities2" : ["kelly_front_s2_v2/", "hadleigh_front_s2_v2/"],
         "identities3" : ["kelly_front_s3_v2/", "hadleigh_front_s3_v2/"],
         "identities4" : ["kelly_front_s4_v2/", "hadleigh_front_s4_v2/"],
      },
      "utterance_test_cases":  
      {
            "utterances1" : ["kelly_front_s28_v2/", "kelly_front_s27_v2/"],
            "utterances2" : ["kelly_low_s28_v2/", "kelly_low_s27_v2/"],
            "utterances3" : ["kelly_right_threequarter_s28_v2/", "kelly_right_threequarter_s27_v2/"],
            "utterances4" : ["hadleigh_front_s28_v2/", "hadleigh_front_s27_v2/"],
        
      }
   }
}

PipelineComparison = {
   "to run directories" : ["run_number_1/", "run_number_2"], 
   "fpaths" : [
      "sigcomparison_data/angle_test_cases/report.txt", 
      "sigcomparison_data/identity_test_cases/report.txt", 
      "sigcomparison_data/utterance_test_cases/report.txt"],
   "baw titles" : {
      "sigcomparison_data/angle_test_cases/report.txt" : "Pearson Correlation Distribution Across Angle Test Cases", 
      "sigcomparison_data/identity_test_cases/report.txt" : "Pearson Correlation Distribution Across Identity Test Cases", 
      "sigcomparison_data/utterance_test_cases/report.txt" : "Pearson Correlation Distribution Across Utterance Test Cases"
      },
   "baw xlabels" : {
      "run_number_1" : "Average across 2 pairs, window size 5", 
      "run_number_2" : "No smoothing or averaging",  
   }, 
   "scatter_dirs" : ["Avg3ThenSmooth4", "NoSmoothing"], 
   "scatter_markers" : {"Avg3ThenSmooth4" : "o", "NoSmoothing" : "x"}

}