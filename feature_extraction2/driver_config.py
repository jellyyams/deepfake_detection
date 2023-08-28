Run = {
   "number" : 1, 
   "video processing?" : True,
   "signal processing?" : True,
   "signal comparison?" : False,
   "pipeline comparison?" : False, 
   "find best pairs?" : False, 
}

Directories = {
   "video input root" : "../../../Desktop/Deepfake_Detection/Test_Videos/", 
   "output root" : "./results/run_number_", 
   "vid data output" : "vidprocessing_data/", 
   "sig processing data output" : "sigprocessing_data/"
}

VideoProcessing = {
   "initial detect?" : True,
   "draw all landmarks?" : True,
   "generate video?": True,
   "output data" : ["pairwise_distances", "anchor_distances"],
   "anchor landmark" : 5, 
   "display dim" : 800, 
   "draw landmark nums?" : False, 
   "draw anchor target connector?" : False,
   "key regions" : ["Inner", "Outer", "1", "Corner", "0", "Eyebrow"],
   "norm by" : "first_upper_lower_bbox", #vs "first_quarters_bbox" vs "face_bbox" vs "none" vs "upper_lower_bbox" vs "first_face_bbox" vs "first_upper_lower_bbox"
   "videos" : [
      "Krithika_Front/krithika_front_s2_v1",
      "Kelly_Front/kelly_front_s2_v1",
   ], 
   "pairs to plot": [
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

SignalProcessing = {
   "files" : ["kelly_front_s2_v1/", "krithika_front_s2_v1/"], 
   "datatype" : ["landmark_groups", "landmark_single", "coords"], 
   "make plots?" : True
}

SignalComparison = {}

PipelineComparison = {}