# Typing
from typing import Union, List, Literal
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
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 max_num_faces :int = 3):
        super().__init__()
        self._face_mesh = mp_face_mesh.FaceMesh(static_image_mode = False,
                                                min_detection_confidence = min_detection_confidence,
                                                min_tracking_confidence = min_tracking_confidence,
                                                max_num_faces = max_num_faces)

    @staticmethod
    def _get_central_face(image :np.ndarray,
                          prediction :list):
        img_h, img_w, _ = image.shape
        target_point = int(img_w/2), int(img_h/2)

        # Store the closest face info
        min_distance = float('inf')
        closest_face_landmarks = None

        for face_landmarks in prediction.multi_face_landmarks:
            # Compute the mean x, y of all landmarks in the face
            xs = [lm.x * img_w for lm in face_landmarks.landmark]
            ys = [lm.y * img_h for lm in face_landmarks.landmark]
            face_center_x = np.mean(xs)
            face_center_y = np.mean(ys)

            distance = np.sqrt((face_center_x - target_point[0]) ** 2 + (face_center_y - target_point[1]) ** 2)

            if distance < min_distance:
                min_distance = distance
                closest_face_landmarks = face_landmarks

        return closest_face_landmarks

    def detect_faces(self,
                     image: Union[str, np.ndarray],
                     limit: int = 1,
                     mode :Literal["central","biggest"] = "central") -> List[FaceDetection]:
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
        # Get face with most central position compare to the image
        face_landmarks = self._get_central_face(image = image, prediction = prediction) if mode == 'central' else None  # Currently select the largest one
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
