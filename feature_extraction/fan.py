import numpy as np, face_alignment
from video_app import VideoApp
import fan_alignment

class FANApp(VideoApp):

    def __init__(self, input_video, win_size=60, filter=50, bbox_norm=True, draw_all=False, show_num=False, show_dist=True, three_d=False):
        VideoApp.__init__(self, input_video, 'fan_output', 51, [56, 58], win_size, filter, bbox_norm, draw_all, show_num, show_dist, three_d, model='fan')
        self.detector = face_alignment.FaceAlignment((face_alignment.LandmarksType._3D), face_detector='blazeface', flip_input=True, device='cuda')

    def detect(self, frame):
        face_landmarks_list = self.detector.get_landmarks(frame)
        if face_landmarks_list is None:
            return []
        face_landmarks_list = face_landmarks_list[-1]
        landmark_coords = []
        for i in range(face_landmarks_list.shape[0]):
            landmark_coords.append([face_landmarks_list[i][0], face_landmarks_list[i][1], face_landmarks_list[i][2]])
        else:
            return landmark_coords

    def align_landmarks(self, landmark_list):
        landmark_array = np.array(landmark_list)
        transformed = fan_alignment.align(landmark_array)
        transformed = (transformed - transformed.min()) / (transformed.max() - transformed.min())
        transformed[:, 0] = transformed[:, 0] * int(self.W / 2) + int(self.W / 4)
        transformed[:, 1] = transformed[:, 1] * int(self.H / 2) + int(self.H / 4)
        return (
         transformed[:, :2], transformed)

    def update_points(self, landmark_coords=None):
        if landmark_coords is None:
            for i in range(68):
                if i not in self.points:
                    self.points[i] = [
                     np.nan]
                else:
                    self.points[i].append(np.nan)
                if i in self.target_points:
                    if i not in self.diffs:
                        self.diffs[i] = [
                         np.nan]
                    else:
                        self.diffs[i].append(np.nan)
                return

        anchor_coord = landmark_coords[self.anchor_point]
        for i, l in enumerate(landmark_coords):
            if i not in self.points:
                self.points[i] = [
                 l]
            else:
                self.points[i].append(l)
            if i in self.target_points:
                x_diff = anchor_coord[0] - l[0]
                y_diff = anchor_coord[1] - l[1]
                z_diff = anchor_coord[2] - l[2]
                if self.bbox_norm:
                    cx_min = self.W
                    cy_min = self.H
                    cz_min = self.W
                    cx_max = cy_max = cz_max = 0
                    for id, lm in enumerate(landmark_coords):
                        cx, cy, cz = lm
                        if cx < cx_min:
                            cx_min = cx
                        if cy < cy_min:
                            cy_min = cy
                        if cz < cz_min:
                            cz_min = cz
                        if cx > cx_max:
                            cx_max = cx
                        if cy > cy_max:
                            cy_max = cy
                        if cz > cz_max:
                            cz_max = cz
                        bbox = [(cx_min, cy_min, cz_min), (cx_max, cy_max, cz_max)]
                        bbox_W = bbox[1][0] - bbox[0][0]
                        bbox_H = bbox[1][1] - bbox[0][1]
                        bbox_D = bbox[1][2] - bbox[0][2]
                        x_diff /= bbox_W
                        y_diff /= bbox_H
                        z_diff /= bbox_D

                elif self.three_d:
                    dist = np.sqrt(x_diff ** 2 + y_diff ** 2 + z_diff ** 2)
                else:
                    dist = np.sqrt(x_diff ** 2 + y_diff ** 2)
                if i not in self.diffs:
                    self.diffs[i] = [
                     dist]
                else:
                    self.diffs[i].append(dist)