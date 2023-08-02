ExtractionSettings = {
   "initial_detect" : True,
   "draw_all_landmarks" : True,
   "generate_video": True,
   "output_data" : ["pairwise_distances", "anchor_distances"],
   "anchor_landmark" : 5, 
   "key_regions" : ["Inner", "Outer", "1", "Corner", "0", "Eyebrow"],
   "root_video_path" : "../../../Desktop/Deepfake_Detection/Test_Videos", 
   "norm_approach" : "first_upper_lower_bbox", #vs "first_quarters_bbox" vs "face_bbox" vs "none" vs "upper_lower_bbox" vs "first_face_bbox" vs "first_upper_lower_bbox"
   "video_paths" : [
    #    "/Hadleigh_Low/hadleigh_low_s31_v2",
    #    "/Hadleigh_Right_Threequarter/hadleigh_right_threequarter_s31_v2",
    #    "/Hadleigh_Left_Threequarter/hadleigh_left_threequarter_s31_v2",
    #    "/Hadleigh_Front/hadleigh_front_s31_v2",
    #    "/Hadleigh_Low/hadleigh_low_s27_v2",
    #    "/Hadleigh_Right_Threequarter/hadleigh_right_threequarter_s27_v2",
    #    "/Hadleigh_Left_Threequarter/hadleigh_left_threequarter_s27_v2",
    #    "/Hadleigh_Front/hadleigh_front_s27_v2",
    #    "/Hadleigh_Low/hadleigh_low_s28_v2",
    #    "/Hadleigh_Right_Threequarter/hadleigh_right_threequarter_s28_v2",
    #    "/Hadleigh_Left_Threequarter/hadleigh_left_threequarter_s28_v2",
    #    "/Hadleigh_Front/hadleigh_front_s28_v2",
    #    "/Kelly_Low/kelly_low_s31_v2",
    #    "/Kelly_Right_Threequarter/kelly_right_threequarter_s31_v2",
    #    "/Kelly_Left_Threequarter/kelly_left_threequarter_s31_v2",
    #    "/Kelly_Front/kelly_front_s31_v2",
    #    "/Kelly_Low/kelly_low_s32_v2",
    #    "/Kelly_Right_Threequarter/kelly_right_threequarter_s32_v2",
    #    "/Kelly_Left_Threequarter/kelly_left_threequarter_s32_v2",
    #    "/Kelly_Front/kelly_front_s32_v2",
    #    "/Kelly_Low/kelly_low_s27_v2",
    #    "/Kelly_Right_Threequarter/kelly_right_threequarter_s27_v2",
    #    "/Kelly_Left_Threequarter/kelly_left_threequarter_s27_v2",
    #    "/Kelly_Front/kelly_front_s27_v2",
    #    "/Kelly_Low/kelly_low_s28_v2",
    #    "/Kelly_Right_Threequarter/kelly_right_threequarter_s28_v2",
    #    "/Kelly_Left_Threequarter/kelly_left_threequarter_s28_v2",
    #    "/Kelly_Front/kelly_front_s28_v2",
    #    "/Hadleigh_Low/hadleigh_low_s1_v2",
    #    "/Hadleigh_Right_Threequarter/hadleigh_right_threequarter_s1_v2",
    #    "/Hadleigh_Left_Threequarter/hadleigh_left_threequarter_s1_v2",
    #    "/Hadleigh_Front/hadleigh_front_s1_v2",
    #    "/Hadleigh_Low/hadleigh_low_s2_v2",
    #    "/Hadleigh_Right_Threequarter/hadleigh_right_threequarter_s2_v2",
    #    "/Hadleigh_Left_Threequarter/hadleigh_left_threequarter_s2_v2",
    #    "/Hadleigh_Front/hadleigh_front_s2_v2",
    #    "/Hadleigh_Low/hadleigh_low_s3_v2",
    #    "/Hadleigh_Right_Threequarter/hadleigh_right_threequarter_s3_v2",
    #    "/Hadleigh_Left_Threequarter/hadleigh_left_threequarter_s3_v2",
    #    "/Hadleigh_Front/hadleigh_front_s3_v2",
    #    "/Hadleigh_Low/hadleigh_low_s4_v2",
    #    "/Hadleigh_Right_Threequarter/hadleigh_right_threequarter_s4_v2",
    #    "/Hadleigh_Left_Threequarter/hadleigh_left_threequarter_s4_v2",
    #    "/Hadleigh_Front/hadleigh_front_s4_v2",
    #    "/Hadleigh_Low/hadleigh_low_s5_v2",
    #    "/Hadleigh_Right_Threequarter/hadleigh_right_threequarter_s5_v2",
    #    "/Hadleigh_Left_Threequarter/hadleigh_left_threequarter_s5_v2",
    #    "/Hadleigh_Front/hadleigh_front_s5_v2",
    #    "/Hadleigh_Low/hadleigh_low_s6_v2",
    #    "/Hadleigh_Right_Threequarter/hadleigh_right_threequarter_s6_v2",
    #    "/Hadleigh_Left_Threequarter/hadleigh_left_threequarter_s6_v2",
       "/Krithika_Front/krithika_front_s2_v1",
       "/Kelly_Front/kelly_front_s2_v1",
    #    "/Kelly_Low/kelly_low_s1_v2",
    #    "/Kelly_Right_Threequarter/kelly_right_threequarter_s1_v2",
    #    "/Kelly_Left_Threequarter/kelly_left_threequarter_s1_v2",
    #    "/Kelly_Front/kelly_front_s1_v2",
    #    "/Kelly_Low/kelly_low_s2_v2",
    #    "/Kelly_Right_Threequarter/kelly_right_threequarter_s2_v2",
    #    "/Kelly_Left_Threequarter/kelly_left_threequarter_s2_v2",
    #    "/Kelly_Front/kelly_front_s2_v2",
    #    "/Kelly_Low/kelly_low_s3_v2",
    #    "/Kelly_Right_Threequarter/kelly_right_threequarter_s3_v2",
    #    "/Kelly_Left_Threequarter/kelly_left_threequarter_s3_v2",
    #    "/Kelly_Front/kelly_front_s3_v2",
    #    "/Kelly_Low/kelly_low_s4_v2",
    #    "/Kelly_Right_Threequarter/kelly_right_threequarter_s4_v2",
    #    "/Kelly_Left_Threequarter/kelly_left_threequarter_s4_v2",
    #    "/Kelly_Front/kelly_front_s4_v2",
    #    "/Kelly_Low/kelly_low_s5_v2",
    #    "/Kelly_Right_Threequarter/kelly_right_threequarter_s5_v2",
    #    "/Kelly_Left_Threequarter/kelly_left_threequarter_s5_v2",
    #    "/Kelly_Front/kelly_front_s5_v2",
    #    "/Kelly_Low/kelly_low_s6_v2",
    #    "/Kelly_Right_Threequarter/kelly_right_threequarter_s6_v2",
    #    "/Kelly_Left_Threequarter/kelly_left_threequarter_s6_v2",
    #    "/Kelly_Front/kelly_front_s6_v2"
   ]
}

PlottingSettings = {
    "videos_for_plotting" : [
        "kelly_front_s2_v2/",
        "hadleigh_front_s2_v2/",
        "krithika_front_s2_v1/",
        "kelly_front_s2_v1/", 
    ], 
    "labels" : {
        "kelly_front_s2_v2/" : "Speaker 1, Sentence 2",
        "hadleigh_front_s2_v2/": "Speaker 2, Sentence 2",
        "krithika_front_s2_v1/": "Speaker 3, Sentence 2",
        "kelly_front_s2_v1/": "Speaker 4, Sentence 2", 
    },
    "singlevid" : "kelly_front_s2_v2/"
}

ComparisonSettings = {
    "top_cutoff" : -1, #range from -1 to 1, setting to -1 means nothing is cut off
    "bottom_cutoff" : 1, #range from -1 to 1, setting to 1 means nothing is cut off
    "top_num" : 150, #if using pipeline to find best performing landmarks, this filters out the top 150 performing pairs for similarity test cases
    "bottom_num" : 2000, #if using pipeline to find best performing landmarks, this filters out the bottom 2000 performing pairs for similarity test cases
    "filterpairs?" : True #To speed up processing filter out pairs that aren't being analyzed
}

CoordProcessingSettings = {
    "type" : "area", #vs "pairwise_distance" vs "anchor_distance" vs "angle" vs "velocity"
    "target_pairs" : [ #should match the keys in "pairs_for_avg"
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
    # "pairs_for_avg" : 
    # {
    #     (336, 384) : [(336, 385)], 
    #     (296, 385) : [(296, 386)], 
    #     (255, 257) : [(257, 339)],
    #     (258, 327) : [(257, 327)], 
    #     (25, 159) : [(110, 159)], 
    #     (61, 160): [(61, 159)], 
    #     (24, 159) : [(110, 159)],
    #     (287, 327) : [(291, 327)], 
    #     (98, 292) : [(98, 308)], 
    #     (334, 386) : [(334, 387)], 
    #     (293, 387) : [(293, 388)], 
    #     (13, 321) : [(312, 321)], 
    #     (375, 409) : [(321, 409)], 
    #     (311, 375) : [(311, 321)], 
    #     (270, 321) : [(269, 321)], 
    #     (81, 181) : [(82, 181)], 
    #     (82, 321) : [(82, 405)], 
    #     (0, 405) : [(0, 321)]
    # }, 
    "pairs_for_avg" : 
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
    # "pairs_for_avg" : 
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

SignalProcessingSettings = {
    "preavg_process" : "none",
    "averagepairs?" : True, 
    "moving_avg_window" : 4, 
    "postavg_process" : "moving_average",
}

TestCases = {
   "angle_test_cases" : {
        "name" : "angle_test_cases", 
        "data" : {
            "kelly_angles1" : ["kelly_front_s1_v2/", "kelly_low_s1_v2/", "kelly_right_threequarter_s1_v2/", "kelly_left_threequarter_s1_v2/"],
            "kelly_angles2" : ["kelly_front_s2_v2/", "kelly_low_s2_v2/", "kelly_right_threequarter_s2_v2/", "kelly_left_threequarter_s2_v2/"],
            "kelly_angles3" : ["kelly_front_s3_v2/", "kelly_low_s3_v2/", "kelly_right_threequarter_s3_v2/", "kelly_left_threequarter_s3_v2/"],
            "kelly_angles4" : ["kelly_front_s4_v2/", "kelly_low_s4_v2/", "kelly_right_threequarter_s4_v2/", "kelly_left_threequarter_s4_v2/"],
            "kelly_angles5" : ["kelly_front_s5_v2/", "kelly_low_s5_v2/", "kelly_right_threequarter_s5_v2/", "kelly_left_threequarter_s5_v2/"],
            "kelly_angles6" : ["kelly_front_s6_v2/", "kelly_low_s6_v2/", "kelly_right_threequarter_s6_v2/", "kelly_left_threequarter_s6_v2/"],
            "hadleigh_angles1" : ["hadleigh_front_s1_v2/", "hadleigh_low_s1_v2/", "hadleigh_right_threequarter_s1_v2/", "hadleigh_left_threequarter_s1_v2/"],
            "hadleigh_angles2" : ["hadleigh_front_s2_v2/", "hadleigh_low_s2_v2/", "hadleigh_right_threequarter_s2_v2/", "hadleigh_left_threequarter_s2_v2/"],
            "hadleigh_angles3" : ["hadleigh_front_s3_v2/", "hadleigh_low_s3_v2/", "hadleigh_right_threequarter_s3_v2/", "hadleigh_left_threequarter_s3_v2/"],
            "hadleigh_angles4" : ["hadleigh_front_s4_v2/", "hadleigh_low_s4_v2/", "hadleigh_right_threequarter_s4_v2/", "hadleigh_left_threequarter_s4_v2/"],
            "hadleigh_angles5" : ["hadleigh_front_s5_v2/", "hadleigh_low_s5_v2/", "hadleigh_right_threequarter_s5_v2/", "hadleigh_left_threequarter_s5_v2/"],
            "hadleigh_angles6" : ["hadleigh_front_s6_v2/", "hadleigh_low_s6_v2/", "hadleigh_right_threequarter_s6_v2/", "hadleigh_left_threequarter_s6_v2/"]
        }
    },
   "identity_test_cases" : {
        "name" : "identity_test_cases", 
        "data" : {
            "identities1" : ["kelly_front_s1_v2/", "hadleigh_front_s1_v2/"],
            "identities2" : ["kelly_front_s2_v2/", "hadleigh_front_s2_v2/"],
            "identities3" : ["kelly_front_s3_v2/", "hadleigh_front_s3_v2/"],
            "identities4" : ["kelly_front_s4_v2/", "hadleigh_front_s4_v2/"],
            "identities5" : ["kelly_low_s1_v2/", "hadleigh_low_s1_v2/"],
            "identities6" : ["kelly_low_s2_v2/", "hadleigh_low_s2_v2/"],
            "identities7" : ["kelly_low_s3_v2/", "hadleigh_low_s3_v2/"],
            "identities8" : ["kelly_low_s4_v2/", "hadleigh_low_s4_v2/"],
        }
   }, 
   "utterance_test_cases" : {
        "name" : "utterance_test_cases", 
        "data" : {
            "utterances1" : ["kelly_front_s28_v2/", "kelly_front_s27_v2/"],
            "utterances2" : ["kelly_low_s28_v2/", "kelly_low_s27_v2/"],
            "utterances3" : ["kelly_right_threequarter_s28_v2/", "kelly_right_threequarter_s27_v2/"],
            "utterances4" : ["hadleigh_front_s28_v2/", "hadleigh_front_s27_v2/"],
            "utterances5" : ["hadleigh_low_s28_v2/", "hadleigh_low_s27_v2/"],
            "utterances6" : ["hadleigh_right_threequarter_s28_v2/", "hadleigh_right_threequarter_s27_v2/"],
            "utterances7" : ["kelly_front_s31_v2/", "kelly_front_s32_v2/"],
            "utterances8" : ["kelly_low_s31_v2/", "kelly_low_s32_v2/"]
        }
       
   }
}

CompareResults = {
    # "directories" : ["pairs2_v2/Avg4ThenSmooth4/", "pairs2_v2/Avg3ThenSmooth4/",  "pairs2_v2/Avg2ThenSmooth4/", "pairs2_v2/MovingAvg4/", "pairs2_v2/Avg4Pairs/", "pairs2_v2/Avg3Pairs/", "pairs2_v2/Avg2Pairs/", "pairs2_v2/NoSmoothing/"], 
    "directories" : ["pairs2_v2/Avg3ThenSmooth4/", "pairs2_v2/Avg3Butterworth4/",  "pairs2_v2/Avg2Butterworth4/", "pairs2_v2/Butterworth4/","pairs2_v2/Butterworth5/", "pairs2_v2/NoSmoothing/"], 
    "files" : ["angle_test_casesreport.txt", "identity_test_casesreport.txt", "utterance_test_casesreport.txt"],
    "titles" : {
        "angle_test_casesreport.txt" : "Pearson Correlation Distribution Across Angle Test Cases", 
        "identity_test_casesreport.txt" : "Pearson Correlation Distribution Across Identity Test Cases", 
        "utterance_test_casesreport.txt" : "Pearson Correlation Distribution Across Utterance Test Cases"
        },
    "xlabels" : {
        "Avg4ThenSmooth4" : "Average across 4 pairs, window size 4", 
        "Avg3ThenSmooth4" : "Average across 3 pairs, window size 4",  
        "Avg2ThenSmooth4" : "Average across 2 pairs, window size 4", 
        "MovingAvg4" : "Mpving average of window size 4", 
        "Avg4Pairs" : "Average across 4 pairs, no smoothing", 
        "Avg3Pairs" : "Average across 3 pairs, no smoothing", 
        "Avg2Pairs" : "Average across 2 pairs, no smoothing",
        "NoSmoothing" : "No smoothing or averaging", 
        "Butterworth5" : "Butterworth low pass cutoff of 5 hz", 
        "Butterworth4" : "Butterworth low pass cutoff of 4 hz", 
        "Avg3Butterworth4" : "Average across 3 pairs, 4 hz cutoff",
        "Avg2Butterworth4" : "Average across 2 pairs, 4 hz cutoff",
    }, 
    "scatter_dirs" : ["Avg3ThenSmooth4", "NoSmoothing"], 
    "scatter_markers" : {"Avg3ThenSmooth4" : "o", "NoSmoothing" : "x"}
}


OutputDir = "./correlation_reports/pairs2_v2/Avg3ThenSmooth4/"

Run = {
    "video extractions": True, 
    "test cases": False, 
    "pair plotting" : True, 
    "correlation comparison": True
}




