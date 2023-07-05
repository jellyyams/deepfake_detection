
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import numpy as np

def draw_landmarks_on_image(rgb_image, detection_result):
  """
  From Google's Mediapipe Facemesh example
  """
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
  
  return annotated_image


def draw_landmark_pairs(target_landmark_pairs, output_name = 'annotated_mp_sample.jpg'):
  base_options = python.BaseOptions(model_asset_path='weights/face_landmarker_v2_with_blendshapes.task')
  options = vision.FaceLandmarkerOptions(base_options=base_options,num_faces=1, min_face_detection_confidence=.25, min_face_presence_confidence=.25, min_tracking_confidence=.25, output_face_blendshapes=True, output_facial_transformation_matrixes=True)
  detector = vision.FaceLandmarker.create_from_options(options)

  image = mp.Image.create_from_file("mp_sample.jpg")
  detection_result = detector.detect(image)

  annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
  annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

  face_landmarks_list = detection_result.face_landmarks
  face_landmarks = face_landmarks_list[0] 
  # scale x,y,z by image dimensions
  H, W, c = annotated_image.shape #not the same as self.H, self.W if initial face cropping is being used!
  landmark_coords = [(landmark.x * W, landmark.y * H, landmark.z) for landmark in face_landmarks] #un-normalize
  for pair in target_landmark_pairs:
    p1_x, p1_y = landmark_coords[pair[0]][:2]
    p2_x, p2_y = landmark_coords[pair[1]][:2]
    color = (0, 0, 255)
    cv2.circle(annotated_image, (int(p1_x), int(p1_y)), 2, color=color, thickness=-1) #draw landmarks
    cv2.circle(annotated_image, (int(p2_x), int(p2_y)), 2, color=color, thickness=-1) 
    cv2.putText(annotated_image, str(pair[0]), (int(p1_x) + 5, int(p1_y) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(annotated_image, str(pair[1]), (int(p2_x) + 5, int(p2_y) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(annotated_image, (int(p1_x), int(p1_y)), (int(p2_x), int(p2_y)), color, 1)
  cv2.imwrite(output_name, annotated_image)

draw_landmark_pairs([(13, 321), (375, 409), (311, 375), (312, 321), (312, 375), 
               (270, 321), (311, 321), (310, 405), (13, 405), (270, 375), (81, 181), (311, 314)],
               output_name='annotated_mp_sample1.jpg')
draw_landmark_pairs([(78, 146), (40, 84), (98, 402), (0, 375), (91, 185), 
               (37, 375), (181, 191), (98, 318), (98, 321), (269, 375), (267, 375)], 
               output_name='annotated_mp_sample2.jpg')