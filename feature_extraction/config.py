ExtractionSettings = {
   "initial_detect" : True,
   "draw_all_landmarks" : True,
   "generate_video": False,
   "analysis_types" : ["landmark_pairs", "landmark_to_anchor"],
   "anchor_landmark" : 57,
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
       "/Hadleigh_Low/hadleigh_low_s6_v2",
       "/Hadleigh_Right_Threequarter/hadleigh_right_threequarter_s6_v2",
       "/Hadleigh_Left_Threequarter/hadleigh_left_threequarter_s6_v2",
       "/Hadleigh_Front/hadleigh_front_s6_v2",
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
       "/Kelly_Low/kelly_low_s6_v2",
       "/Kelly_Right_Threequarter/kelly_right_threequarter_s6_v2",
       "/Kelly_Left_Threequarter/kelly_left_threequarter_s6_v2",
       "/Kelly_Front/kelly_front_s6_v2"
   ]
}

PlottingSettings = {
    "videos_for_plotting" : [
        "hadleigh_front_s3_v2/",
        "hadleigh_low_s3_v2/",
        "hadleigh_left_threequarter_s3_v2/",
        "hadleigh_right_threequarter_s3_v2/"
    ], 
    "pairs_for_plotting" : [
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
    ]
}

ComparisonSettings = {
    "top_cutoff" : -1, 
    "bottom_cutoff" : 1, 
    "top_num" : 150, 
    "bottom_num" : 2000, 
    "filterpairs?" : True
}

ProcessingSettings = {
    "type" : "normalize",
    "averagepairs?" : True, 
    "window" : 5, 
    "postavg_process" : "none",
    "pairs_for_avg" : 
    {
        (336, 384) : (336, 385), 
        (296, 385) : (296, 386), 
        (255, 257) : (257, 339),
        (258, 327) : (257, 327), 
        (25, 159) : (110, 159), 
        (61, 160): (61, 159), 
        (24, 159) : (110, 159),
        (287, 327) : (291, 327), 
        (98, 292) : (98, 308), 
        (334, 386) : (334, 387), 
        (293, 387) : (293, 388), 
        (13, 321) : (312, 321), 
        (375, 409) : (321, 409), 
        (311, 375) : (311, 321), 
        (270, 321) : (269, 321), 
        (81, 181) : (82, 181), 
        (82, 321) : (82, 405), 
        (0, 405) : (0, 321)
    }
}

TestCases = {
   "angle_test_cases" : {
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
   },
   "identity_test_cases" : {
       "identities1" : ["kelly_front_s1_v2/", "hadleigh_front_s1_v2/"],
       "identities2" : ["kelly_front_s2_v2/", "hadleigh_front_s2_v2/"],
       "identities3" : ["kelly_front_s3_v2/", "hadleigh_front_s3_v2/"],
       "identities4" : ["kelly_front_s4_v2/", "hadleigh_front_s4_v2/"],
       "identities5" : ["kelly_low_s1_v2/", "hadleigh_low_s1_v2/"],
       "identities6" : ["kelly_low_s2_v2/", "hadleigh_low_s2_v2/"],
       "identities7" : ["kelly_low_s3_v2/", "hadleigh_low_s3_v2/"],
       "identities8" : ["kelly_low_s4_v2/", "hadleigh_low_s4_v2/"],
   }, 
   "utterance_test_cases" : {
       "utterances1" : ["kelly_front_s28_v2/", "kelly_front_s27_v2/"],
       "utterances2" : ["kelly_low_s28_v2/", "kelly_low_s27_v2/"],
       "utterances3" : ["kelly_right_threequarter_s28_v2/", "kelly_right_threequarter_s27_v2/"],
       "utterances4" : ["hadleigh_front_s28_v2/", "hadleigh_front_s27_v2/"],
       "utterances5" : ["hadleigh_low_s28_v2/", "hadleigh_low_s27_v2/"],
       "utterances6" : ["hadleigh_right_threequarter_s28_v2/", "hadleigh_right_threequarter_s27_v2/"],
       "utterances7" : ["kelly_front_s31_v2/", "kelly_front_s32_v2/"],
       "utterances8" : ["kelly_low_s31_v2/", "kelly_low_s32_v2/"],
   }


}

CompareResults = {
    "directories" : ["pairs2/Avg2ThenSmooth/", "pairs2/Avg2Pairs/", "pairs2/NoSmoothing/"], 
    "files" : ["sim_then_diff_report.txt", "sim_report.txt"],
    "titles" : {
        "sim_then_diff_report.txt" : "Pearson Correlation Distribution Across Difference Test Cases", 
        "sim_report.txt" : "Pearson Correlation Distribution Across Similarity Test Cases"
        }
    
}



OutputDir = "./correlation_reports/pairs2/Avg2Pairs/"




