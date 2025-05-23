# Typing
from typing import Union, List
# Base Recognition
from .base_recognition import BaseRecognition, FaceDetection, FacialKeyPoints
# Other component
import cv2
import mediapipe as mp
import numpy as np

# Define face mesh
mp_face_mesh = mp.solutions.face_mesh

# Key landmark
KEY_LANDMARKS = {
    "nose": 1,
    "chin": 152,
    "left_eye": 263,
    "right_eye": 33,
    "left_mouth": 287,
    "right_mouth": 57,
}

class MediapipeDetection(BaseRecognition):
    def __init__(self,
                 model_selection :int = 1,
                 min_detection_confidence: float = 0.5):
        super().__init__()
        self._face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                                max_num_faces=1)

    def detect_faces(self,
                     image: Union[str, np.ndarray],
                     limit: int = 1) -> List[FaceDetection]:
        """Predict the face keypoint from input image"""
        # Get the face information
        h, w = image.shape[:2]
        # Convert color
        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Process the images
        prediction = self._face_mesh.process(rgb_frame)

        # When no face detection
        if not prediction.multi_face_landmarks:
            return None
        # Get the first face *** Should select largest ***
        face_landmarks = prediction.multi_face_landmarks[0]  # Currently select the largest one
        # Get landmark
        landmarks = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]

        # Get bounding box coordinates
        x_coords = [pt[0] for pt in landmarks]
        y_coords = [pt[1] for pt in landmarks]
        # Get coordination of bounding box
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # Define image points
        image_points = np.array([
            (landmarks[i][0] * w, landmarks[i][1] * h) for i in list(KEY_LANDMARKS.values())
        ], dtype="double")
        # Define key point
        keypoint = FacialKeyPoints(nose = image_points[0],
                                   chin = image_points[1],
                                   left_eye = image_points[3],
                                   right_eye = image_points[2],
                                   left_mouth = image_points[5],
                                   right_mouth = image_points[4])
        # Return
        return [FaceDetection(box = [int(x_min * w),int(y_min * h),int(x_max * w),int(y_max * h)],
                              keypoints = keypoint)]
