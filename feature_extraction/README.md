All feature extraction is handled by the FeatureExtractor class, located in feature_extractor.py. The MPFeatureExtractor and FANFeatureExtractor (located in mp.py and fan.py, respectively) inherit from FeatureExtractor and correspond to specific implementations of the FeatureExtractor based on MediaPipe (MP) facial landmark extraction and FAN facial landmark extraction. 

To perform feature extraction on a video, create a MPFeatureExtractor or FANFeatureExtractor class and run it like below:

```
from mp import MPFeatureExtractor
fe = MPFeatureExtractor(<INPUT VIDEO PATH>, <DESIRED OUTPUT DIRECTORY PATH>, draw_all_landmarks=True initial_detect=True)
fe.run_extraction()
```

See feature_extractor.py for a detailed description of parameters that can be used when initializing the feature extractor. The example above initializes a feature extractor that draws all landmarks in the output video and performs initial face detection. 

To visualize output videos side by side (e.g., if you want to compare the results of feature extraction on video of a scene captured from different camera distances/poses), you can use stack_vids (located in ../common/stack_vids.py). Use it like so:

```
    stack_vids(<INPUT VIDEO 1 PATH>, <INPUT VIDEO 2 PATH>, <OUTPUT VIDEO PATH>, vertical=True, out_fps = 30)
```

This will vertically stack two input videos and generate an output with a framerate of 30 fps. 

