# Feature extraction

## Getting started
1. Deep learning models often have a lot of dependencies that can quickly become incompatible with each other. It’s often hard to go back and correct these incompatibilities once they’ve arisen, which can really mess up your environment. To avoid this, I’ve created a conda environment for each model. Create the conda environment containing everything needed to run MediaPipe-based feature extraction from mp_env.yml (can be found at deepfake_detection/misc)
2. Download mobilenet0.25_Final.pth and Resnet50_Final.pth here and put in common/weights. These are the weights for the RetinaFace face detection model. 
3. Download MediaPipe model bundle. These are the weights/configurations for the facial landmark extraction. 
Enter common/weights and enter !wget -O face_landmarker_v2_with_blendshapes.task -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

## Usage
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

