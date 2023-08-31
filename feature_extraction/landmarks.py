from itertools import combinations

class Landmarks:
    '''
    A class that contains lists of landmarks for each region of the face

    Attributes
    ----------
    num_landmarks: int
        the total number of landmarks, likely either 468 or 478. MP identifies 478 but the mp_alignment file can only return 468 at the moment
    landmarks : dict
       a dict of each region of the face and the landmarks they contain in list format
    upper_keywords: list
        words that are used to identify which regions of the face are considered the top half 
    bottom_keywords: list
        words that are used to identify which regions of the face are considered the bottom half
    all: list 
        landmarks from 0 to the number of landmarks there are 
    mouth_landmarks: list
        list of all the landmarks on the mouth 
    upper_landmarks: list 
        landmarks in the upper half of the face
    lower_landmarks: list 
        landmarks in the lower half of the face
    left_landmarks: list 
        landmarks in the left half of the face
    right_landmarks: list
        landmarks in the right half of the face
    upper_right_landmarks: list
        landmakrs in the upper right quarter of the face
    upper_left_landmarks: list
        landmakrs in the upper left quarter of the face
    lower_right_landmarks: list
        landmakrs in the lower right quarter of the face
    lower_left_landmarks: list
        landmakrs in the lower left quarter of the face

    Methods
    ----------
    set_mouth_landmarks(): finds all the landmarks that are on the mouth and sets mouth_landmarks to that list
    set_upper_lower(): separates all landmarks into upper and lower lists
    set_left_right(): separates all landmarks into left and right lists
    set_quarters(): separates all landmarks into upper right, upper left, lower left, and lower right lists
    get_landmarks_by_keyword(keywords): returns all landmarks that are labeled with one of the keywords
    generate_landmark_pairs(landmarks): returns a list of landmarks pairs in sorted order (smaller landmark first and larger landmark second)

    '''
    def __init__(self): 
        self.num_landmarks = 468

        self.landmarks = {
            "silhouetteLeft" : [10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152],
            "slihouetteRight" : [109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148], 
            "silhouetteUpper": [ 93, 234, 127, 162, 21, 54, 109, 67, 103, 10, 338, 297, 332, 284, 251, 389, 356, 454, 323], 
            "silhouetteLower" :  [ 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132], 
            "lipsUpperOuterRight" : [185, 40, 39, 37], 
            "lipsUpperOuterLeft" : [0, 267, 269, 270, 409], 
            "lipsLowerOuterRight" : [146, 91, 181, 84], 
            "lipsLowerOuterLeft": [17, 314, 405, 321, 375], 
            "lipsUpperInnerRight": [191, 80, 81, 82], 
            "lipsUpperInnerLeft" : [13, 312, 311, 310, 415],
            "lipsUpperLeft" : [ 11, 302, 303, 304, 408, 12, 268, 271, 272, 407],
            "lipsUpperRight" : [72, 38, 73, 41, 74, 42, 184, 183],
            "lipsLowerRight" : [86, 179, 89, 96, 77, 90, 180, 85],
            "lipsLowerLeft" : [15, 16, 315, 316, 403, 404, 319, 320, 325, 307],
            "lipsLowerInnerRight" : [95, 88, 178, 87],
            "lipsLowerInnerLeft" : [14, 317, 402, 318, 324],
            "lipsAboveRight" : [ 167, 165, 92, 186],
            "lipsAboveLeft" : [164, 393, 391, 322, 410],
            "lipsRightCorner" : [57, 62, 76, 61, 78],
            "lipsLeftCorner" : [287, 292, 306, 308, 291],
            "lipsBelowLeft" : [18, 313, 406, 335, 273],
            "lipsBelowRight" : [83, 182, 106, 43],
            "rightEyeUpper0" : [246, 161, 160, 159, 158, 157, 173],
            "rightEyeLower0" : [33, 7, 163, 144, 145, 153, 154, 155, 133],
            "rightEyeUpper1" : [247, 30, 29, 27, 28, 56, 190],
            "rightEyeLower1" : [130, 25, 110, 24, 23, 22, 26, 112, 243],
            "rightEyeUpper2" : [113, 225, 224, 223, 222, 221, 189],
            "rightEyeLower2" : [226, 31, 228, 229, 230, 231, 232, 233, 244],
            "rightEyeLower3" : [143, 111, 117, 118, 119, 120, 121, 128, 245],
            "rightEyebrowUpper" : [156, 70, 63, 105, 66, 107, 55, 193],
            "rightEyebrowLower" : [35, 124, 46, 53, 52, 65],
            "rightEyeIris" : [],
            # "rightEyeIris" : [473, 474, 475, 476, 477],

            "leftEyeUpper0" : [466, 388, 387, 386, 385, 384, 398],
            "leftEyeLower0" : [263, 249, 390, 373, 374, 380, 381, 382, 362],
            "leftEyeUpper1" : [467, 260, 259, 257, 258, 286, 414],
            "leftEyeLower1" : [359, 255, 339, 254, 253, 252, 256, 341, 463],
            "leftEyeUpper2" : [342, 445, 444, 443, 442, 441, 413],
            "leftEyeLower2" : [446, 261, 448, 449, 450, 451, 452, 453, 464],
            "leftEyeLower3" : [372, 340, 346, 347, 348, 349, 350, 357, 465],

            "leftEyebrowUpper" : [383, 300, 293, 334, 296, 336, 285, 417],
            "leftEyebrowLower" : [265, 353, 276, 283, 282, 295],
            "leftEyeIris" : [],
            # "leftEyeIris" : [468, 469, 470, 471, 472],

            "topOfBridgeRight" : [122],
            "topOfBridgeLeft" : [6, 351],

            "midwayBetweenEyes" : [8, 9, 168],
            "foreheadRight" : [108, 69, 104, 68,  71, 139, 34 ],
            "foreheadLeft" : [151, 337, 299, 333, 298, 301, 368, 264],

            "noseTipRight" : [141, 125, 241, 242, 44 ],
            "noseTipLeft" : [94, 370, 462, 354, 461, 19, 1, 274],
            "noseBridgeLeft" : [419, 248, 281, 275, 5, 4, 195, 197],
            "noseBridgeRight" : [45, 51, 3, 196],
            "rightNostril" : [166, 59, 75, 60, 20, 238, 239, 79],
            "leftNostril" : [250, 290, 305, 289, 392, 309, 459, 458],

            "noseBottomLeft" : [2, 326, 328],
            "noseBottomRight" : [97, 99],
            "leftAla" : [438, 439, 455, 460, 457, 344, 278, 294, 440, 363, 360],
            "rightAla" : [218, 219, 235, 240, 64, 48, 115, 220, 237, 134, 131],

            "noseRightCorner" : [98],
            "noseLeftCorner" : [327],

            "chinRight" : [32, 194, 135, 169, 138, 170, 140, 171, 428, 201, 202, 204, 208, 210, 211, 212, 214],
            "chinLeft" : [200, 199, 175, 426, 421, 396, 418, 262, 369, 424, 431, 395, 422, 430, 394, 432, 434, 364, 367 ],
            "leftUpperCheek" : [345, 447, 366, 352, 280, 330, 329, 277, 343, 412],
            "leftLowerCheek" : [399, 437, 456, 420, 355, 429, 279, 331, 358, 371, 266, 425, 423, 436, 427, 411, 376, 433, 416, 401, 435],
            "rightUpperCheek" : [227, 137, 116, 123, 50, 101, 100, 47, 114, 188],
            "rightLowerCheek" : [174, 217, 236, 198, 209, 126, 142, 129, 102, 49, 36, 203, 205, 206, 187, 207, 216, 147, 213, 192, 177, 215]

        }
        self.upper_keywords = ["UpperCheek", "topOf", "forehead", "Eye", "silhouetteUpper", "midway"]
        self.lower_keywords = ["lips", "LowerCheek", "chin", "Ala", "Nostril", "nose", "silhouetteLower"]
        self.all = list(range(0, self.num_landmarks))

        self.upper_landmarks = []
        self.lower_landmarks = [] 
        self.right_landmarks = []
        self.left_landmarks = [] 
        self.mouth_landmarks = []

        self.upper_right_landmarks = []
        self.upper_left_landmarks = []
        self.lower_left_landmarks = []
        self.lower_right_landmarks = []

        self.set_upper_lower()
        self.set_left_right()
        self.set_quarters()


    def set_mouth_landmarks(self):
        for k, v in self.landmarks.items():
            if "lip" in k:
                self.mouth_landmarks += v

    def set_upper_lower(self):
        for k, v in self.landmarks.items():

            if any(sub in k for sub in self.upper_keywords):
                self.upper_landmarks += v
            elif any(sub in k for sub in self.lower_keywords):
                self.lower_landmarks += v 
    
    def set_left_right(self):
        for k, v in self.landmarks.items():
            if any(sub in k for sub in ["right", "Right"]):
                self.right_landmarks += v
            elif any(sub in k for sub in ["left", "Left", "midway"]):
                self.left_landmarks += v 
        

    def set_quarters(self):
        self.upper_right_landmarks = list(set(self.upper_landmarks) & set(self.right_landmarks))
        self.upper_left_landmarks = list(set(self.upper_landmarks) & set(self.left_landmarks))
        self.lower_left_landmarks = list(set(self.lower_landmarks) & set(self.left_landmarks))
        self.lower_right_landmarks = list(set(self.lower_landmarks) & set(self.right_landmarks))
    
    def get_landmarks_by_keyword(self, keywords): 
        res = []
        for k, v in self.landmarks.items():
            if any(sub in k for sub in keywords):
                res += v
        
        return res

    def generate_landmark_pairs(self, landmarks):
        c = list(combinations(landmarks, 2))
        sorted_list = []
        for pair in c:
            if pair[0] > pair[1]:
                sorted_list.append((pair[1], pair[0]))
            else:
                sorted_list.append(pair)
        
        return sorted_list
        

    
        

        
    


       