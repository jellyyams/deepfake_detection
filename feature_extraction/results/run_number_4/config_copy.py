'''
A copy of this file is made into the corresponding "run_number_x" folder every single time driver.py is run 
'''

Run = {
   "number" : 4, #the number used to identify which run this is. Changing to new number will create a new directory for all the output in "./results"
   "video processing?" : True, 
   "signal processing?" : False,
   "signal comparison?" : False,
   "pipeline comparison?" : False, 
}

LandmarkSettings = {
   "key regions" : ["Inner", "Outer", "1", "Corner", "0", "Eyebrow"], #the keywords used to determine which landmarks are included in analysis
   "lkeys to plot": [ #the lkeys that will be plotted at each stage, if plotting is turned on 
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
   # "lkeys to avg" : { #the lkeys that will be averaged during signal processing is "avg across signals" is included in pipeline
   #    (336, 384) : [(336, 385)], 
   #    (296, 385) : [(296, 386)], 
   #    (255, 257) : [(257, 339)],
   #    (258, 327) : [(257, 327)], 
   #    (25, 159) : [(110, 159)], 
   #    (61, 160): [(61, 159)], 
   #    (24, 159) : [(110, 159)],
   #    (287, 327) : [(291, 327)], 
   #    (98, 292) : [(98, 308)], 
   #    (334, 386) : [(334, 387)], 
   #    (293, 387) : [(293, 388)], 
   #    (13, 321) : [(312, 321)], 
   #    (375, 409) : [(321, 409)], 
   #    (311, 375) : [(311, 321)], 
   #    (270, 321) : [(269, 321)], 
   #    (81, 181) : [(82, 181)], 
   #    (82, 321) : [(82, 405)], 
   #    (0, 405) : [(0, 321)]
   # }
   "lkeys to avg" : 
    {
        (336, 384) : [(336, 385), (296, 384)], 
        (296, 385) : [(296, 386), (334, 386)], 
        (255, 257) : [(257, 339), (269, 339)],
        (258, 327) : [(257, 327), (259, 327)], 
        (25, 159) : [(110, 159), (25, 160)], 
        (61, 160): [(61, 159), (76, 159)], 
        (24, 159) : [(110, 159), (24, 160)],
        (287, 327) : [(291, 327), (306, 327)], 
        (98, 292) : [(98, 308), (98, 306)], 
        (334, 386) : [(334, 387), (293, 387)], 
        (293, 387) : [(293, 388), (300, 387)], 
        (13, 321) : [(312, 321), (312, 402)], 
        (375, 409) : [(321, 409), (270, 321)], 
        (311, 375) : [(311, 321), (310, 321)], 
        (270, 321) : [(269, 321), (270, 405)], 
        (81, 181) : [(82, 181), (81, 84)], 
        (82, 321) : [(82, 405), (13, 321)], 
        (0, 405) : [(0, 321), (267, 321) ]
    }, 
    # "lkeys to avg" : 
    # {
    #     (336, 384) : [(336, 385), (296, 384), (296, 385)], 
    #     (296, 385) : [(296, 386), (334, 386), (334, 385)], 
    #     (255, 257) : [(257, 339), (269, 339), (255, 269)],
    #     (258, 327) : [(257, 327), (259, 327), (260, 321)], 
    #     (25, 159) : [(110, 159), (25, 160), (110, 160)], 
    #     (61, 160): [(61, 159), (76, 159), (76, 160)], 
    #     (24, 159) : [(110, 159), (24, 160), (110, 160)],
    #     (287, 327) : [(291, 327), (306, 327), (292, 306)], 
    #     (98, 292) : [(98, 308), (98, 306), (98, 291) ], 
    #     (334, 386) : [(334, 387), (293, 387), (293, 386)], 
    #     (293, 387) : [(293, 388), (300, 387), (300, 388)], 
    #     (13, 321) : [(312, 321), (312, 402), (13, 402)], 
    #     (375, 409) : [(321, 409), (270, 321), (270, 409)], 
    #     (311, 375) : [(311, 321), (310, 321), (310, 375)], 
    #     (270, 321) : [(269, 321), (270, 405), (269, 405)], 
    #     (81, 181) : [(82, 181), (81, 84), (82, 84)], 
    #     (82, 321) : [(82, 405), (13, 321), (13, 405)], 
    #     (0, 405) : [(0, 321), (267, 321), (267, 405) ]
    # }


}

Directories = {
   "video input root" : "../../../Desktop/Deepfake_Detection/Test_Videos/", #where all the videos are stored
   "root" : "./results/", #root output folder for anything generated 
   "vid processing output" : "vidprocessing_data/", #name of dir for vid processing output
   "sig processing output" : "sigprocessing_data/", #name of dir for sig processing ouput
   "sig comparison output" : "sigcomparison_data/", #name of dir for sig comparison output
   "pipeline comparison output" : "pipelinecomp_data/" #name of dir for pipeline comparison output
}

VideoProcessing = {
   "initial detect?" : True, #initial detect of face for cropping
   "draw all landmarks?" : True, #draw all landmarks in annotated frame of output video 
   "generate video?": True, #generate a video for every video that's processed. Set to false for faster run time
   "output data" : ["pairwise_distances", "anchor_distances"], #the data that's being extracted during video processing
   "anchor landmark" : 5, #to be used in calculating single landmark distances (all landmarks to this anchor)
   "display dim" : 800, 
   "draw landmark nums?" : False, #draw landmark numbers on the annotated frame of output video
   "draw anchor target connector?" : False,#draw a line between the anchor and landmarks being analyzed
   "norm by" : "first_upper_lower_bbox", #vs "first_quarters_bbox" vs "face_bbox" vs "none" vs "upper_lower_bbox" vs "first_face_bbox" vs "first_upper_lower_bbox"
   "videos" : [ #all the video files being processed
      "Kelly_Front/kelly_front_s1_v2",
      "Kelly_Low/kelly_low_s1_v2",
      "Kelly_Right_Threequarter/kelly_right_threequarter_s1_v2",
      # "Kelly_Left_Threequarter/kelly_left_threequarter_s1_v2",
      # "Kelly_Front/kelly_front_s2_v2",
      # "Kelly_Low/kelly_low_s2_v2",
      # "Kelly_Right_Threequarter/kelly_right_threequarter_s2_v2",
      # "Kelly_Left_Threequarter/kelly_left_threequarter_s2_v2",
      # "Kelly_Front/kelly_front_s3_v2",
      # "Kelly_Low/kelly_low_s3_v2",
      # "Kelly_Right_Threequarter/kelly_right_threequarter_s3_v2",
      # "Kelly_Left_Threequarter/kelly_left_threequarter_s3_v2",
      # "Kelly_Front/kelly_front_s4_v2",
      # "Kelly_Low/kelly_low_s4_v2",
      # "Kelly_Right_Threequarter/kelly_right_threequarter_s4_v2",
      # "Kelly_Left_Threequarter/kelly_left_threequarter_s4_v2",
      # "Kelly_Front/kelly_front_s27_v2",
      # "Kelly_Low/kelly_low_s27_v2",
      # "Kelly_Right_Threequarter/kelly_right_threequarter_s27_v2",
      # "Kelly_Left_Threequarter/kelly_left_threequarter_s27_v2",
      # "Kelly_Front/kelly_front_s28_v2",
      # "Kelly_Low/kelly_low_s28_v2",
      # "Kelly_Right_Threequarter/kelly_right_threequarter_s28_v2",
      # "Kelly_Left_Threequarter/kelly_left_threequarter_s28_v2",
      # "Hadleigh_Front/hadleigh_front_s1_v2",
      # "Hadleigh_Low/hadleigh_low_s1_v2",
      # "Hadleigh_Right_Threequarter/hadleigh_right_threequarter_s1_v2",
      # "Hadleigh_Left_Threequarter/hadleigh_left_threequarter_s1_v2",
      # "Hadleigh_Front/hadleigh_front_s2_v2",
      # "Hadleigh_Low/hadleigh_low_s2_v2",
      # "Hadleigh_Right_Threequarter/hadleigh_right_threequarter_s2_v2",
      # "Hadleigh_Left_Threequarter/hadleigh_left_threequarter_s2_v2",
      # "Hadleigh_Front/hadleigh_front_s3_v2",
      # "Hadleigh_Low/hadleigh_low_s3_v2",
      # "Hadleigh_Right_Threequarter/hadleigh_right_threequarter_s3_v2",
      # "Hadleigh_Left_Threequarter/hadleigh_left_threequarter_s3_v2",
      # "Hadleigh_Front/hadleigh_front_s4_v2",
      # "Hadleigh_Low/hadleigh_low_s4_v2",
      # "Hadleigh_Right_Threequarter/hadleigh_right_threequarter_s4_v2",
      # "Hadleigh_Left_Threequarter/hadleigh_left_threequarter_s4_v2",
      # "Hadleigh_Front/hadleigh_front_s27_v2",
      # "Hadleigh_Low/hadleigh_low_s27_v2",
      # "Hadleigh_Right_Threequarter/hadleigh_right_threequarter_s27_v2",
      # "Hadleigh_Left_Threequarter/hadleigh_left_threequarter_s27_v2",
      # "Hadleigh_Front/hadleigh_front_s28_v2",
      # "Hadleigh_Low/hadleigh_low_s28_v2",
      # "Hadleigh_Right_Threequarter/hadleigh_right_threequarter_s28_v2",
      # "Hadleigh_Left_Threequarter/hadleigh_left_threequarter_s28_v2",
      
   ]
}

SignalProcessing = {
   "files" : [ #videos that will be included in signal processing
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
   "datatype" : "landmark_groups",  #the name of the file from video processing that will be used in signal processing
   "pipeline" : ["avg_across_signals", "moving_average"], #"moving_average", "avg_across_signals", "normalize", or "butterworth"
   "moving avg window" : 5, #the moving average window to be used if "moving_average" is included in "pipeline"
   "butter settings" : { #butterworth filering settings to be used if "butterworth" is included in "pipeline"
      "fs" : 30, 
      "fc" : 4, 
      "type" : "low", 
      "cutoff" : 5
   },
   "make plots?": False, #set to False for faster run time. note that every lkey in "lkeys to plot" will generate a seaprate plot for each video in "files", so setting to "True" may generate a LOT of files
   "should filter?" : True #set to True for faster run time. Set to false when running the pipeline for finding best performing lkeys
}

SignalComparison = {
   "use processed?" : True, #set to false if you want to run the test cases on unproessed data 
   "top cutoff" : 0.6, #range from -1 to 1, setting to -1 means nothing is cut off. Filters out all lkeys with sim scores lower than this cutoff
   "bottom cutoff" : 0.3, #range from 0 to 1, setting to 1 means nothing is cut off. Filters out all lkeys with diff scores greater than this cutoff
   "top num" : 150, #if using pipeline to find best performing landmarks, this filters out the top 150 performing pairs for similarity test cases
   "bottom num" : 2000, #if using pipeline to find best performing landmarks, this filters out the bottom 2000 performing pairs for similarity test cases
   "vids to plot" : [#videos to plot in same plot
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
   "make plots?" : False, #can be run independently of test sets
   "run test sets?": True, 
   "test sets" : {
      "angle_test_cases" : #similarity test cases, ensuring consistency across angles
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
      "identity_test_cases" : #difference test case, ensuring we can differentiate between different speakers 
      {
         "identities1" : ["kelly_front_s1_v2/", "hadleigh_front_s1_v2/"],
         "identities2" : ["kelly_front_s2_v2/", "hadleigh_front_s2_v2/"],
         "identities3" : ["kelly_front_s3_v2/", "hadleigh_front_s3_v2/"],
         "identities4" : ["kelly_front_s4_v2/", "hadleigh_front_s4_v2/"],
      },
      "utterance_test_cases":  #difference test cases, ensuring we can differentiate between different sentences being spoken
      {
            "utterances1" : ["kelly_front_s28_v2/", "kelly_front_s27_v2/"], #sentences 27 and 28 are very similar
            "utterances2" : ["kelly_low_s28_v2/", "kelly_low_s27_v2/"],
            "utterances3" : ["kelly_right_threequarter_s28_v2/", "kelly_right_threequarter_s27_v2/"],
            "utterances4" : ["hadleigh_front_s28_v2/", "hadleigh_front_s27_v2/"],
        
      }
   }
}

PipelineComparison = {
   "baw directories" : ["run_number_3/", "run_number_1/", "run_number_2/"], #enter at least two directories to compare in box and whisker plot
   "fpaths" : [ #the report files being compared 
      "sigcomparison_data/angle_test_cases/report.txt", 
      "sigcomparison_data/identity_test_cases/report.txt", 
      "sigcomparison_data/utterance_test_cases/report.txt"],
   "baw titles" : { #box and whisker titles
      "sigcomparison_data/angle_test_cases/report.txt" : "Pearson Correlation Distribution Across Angle Test Cases", 
      "sigcomparison_data/identity_test_cases/report.txt" : "Pearson Correlation Distribution Across Identity Test Cases", 
      "sigcomparison_data/utterance_test_cases/report.txt" : "Pearson Correlation Distribution Across Utterance Test Cases"
      },
   "baw xlabels" : { #box and whisker xlabels
      "run_number_1" : "Average across 2 pairs, then smooth with window size 5", 
      "run_number_2" : "No smoothing or averaging", 
      "run_number_3" : "Average across 3 pairs, then smooth with window size 5" 
   }, 
   "scatter dirs" : ["run_number_3/", "run_number_2/"], 
   "scatter markers" : {"run_number_3" : "o", "run_number_2" : "x"}
}