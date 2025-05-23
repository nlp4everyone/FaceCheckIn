# Typing
from typing import Union, Literal
# Base point
from app.utils.face.recognition import FacialKeyPoints
# Other component
import mediapipe as mp
import numpy as np
import cv2

# Define face mesh
mp_face_mesh = mp.solutions.face_mesh
# Sample facial point
model_points = np.array([
    (0.0, 0.0, 0.0),           # Nose tip
    (0.0, -330.0, -65.0),      # Chin
    (-225.0, 170.0, -135.0),   # Left eye left corner
    (225.0, 170.0, -135.0),    # Right eye right corner
    (-150.0, -150.0, -125.0),  # Left Mouth corner
    (150.0, -150.0, -125.0)    # Right mouth corner
])

# Key landmark
KEY_LANDMARKS = {
    "nose": 1,
    "chin": 152,
    "left_eye": 263,
    "right_eye": 33,
    "left_mouth": 287,
    "right_mouth": 57,
}

class MediapipeMetric:
    def __init__(self):
        self._face_mesh = mp_face_mesh.FaceMesh(static_image_mode = False,
                                                max_num_faces = 1)

    @staticmethod
    def _get_head_pose(image_points,
                       image_width,
                       image_height):
        """Estimate head point over image points"""
        focal_length = image_width
        center = (image_height / 2, image_height / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        pose_mat = cv2.hconcat((rotation_matrix, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

        pitch, yaw, roll = euler_angles.flatten()
        return pitch, yaw, roll

    def _predict(self,
                 frame :np.ndarray,
                 response_type :Literal["numpy","keypoints"] = "keypoints"):
        """Predict the face keypoint from input image"""
        # Get the face information
        h, w = frame.shape[:2]
        # Convert color
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the images
        prediction = self._face_mesh.process(rgb_frame)

        # When no face detection
        if not prediction.multi_face_landmarks:
            return None
        # Get the first face *** Should select largest ***
        face_landmarks = prediction.multi_face_landmarks[0]  # Currently select the largest one
        # Get landmark
        landmarks = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]

        # Define image points
        image_points = np.array([
            (landmarks[i][0] * w, landmarks[i][1] * h) for i in list(KEY_LANDMARKS.values())
        ], dtype="double")

        # Check type
        if response_type == "numpy": return image_points
        # Return keypoint
        return FacialKeyPoints(nose = image_points[0],
                               chin = image_points[1],
                               left_eye = image_points[2],
                               right_eye = image_points[3],
                               left_mouth = image_points[4],
                               right_mouth = image_points[5])

    def calculate_frontalness_score(self,
                                    frame :np.ndarray) -> Union[float,None]:
        """
        Calculate score to measure how frontal of a face is.
        :param frame: Input image which has a face ( Numpy Array).
        :return: score (float): Higher is better (more frontal)
        """
        # Get the face information
        h, w = frame.shape[:2]
        image_points = self._predict(frame, response_type = "numpy")
        # Calculate pitch, yaw, roll value
        pitch, yaw, roll = self._get_head_pose(image_points,
                                               image_width = w,
                                               image_height = h)
        # *** Add more strategy like accumulate and weighted metrics
        return -(abs(pitch) + abs(yaw))


