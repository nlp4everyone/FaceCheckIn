from app.utils.face.recognition import FaceDetection
import numpy as np

class RankingMetrics:
    @staticmethod
    def calculate_face_tilt(detection :FaceDetection):
        """Roll is the head tilt to one side."""
        # Get position
        right_eye_pos = detection.keypoints.get("right_eye")
        left_eye_pos = detection.keypoints.get("left_eye")
        # Calculate Dx/Dy
        dx = right_eye_pos[0] - left_eye_pos[1]
        dy = right_eye_pos[1] - left_eye_pos[1]
        # Angle
        return np.degrees(np.arctan2(dy,dx))

    @staticmethod
    def calculate_vertical_ratio(detection :FaceDetection):
        """Pitch is head looking up or down"""
        # Get position
        right_eye_pos = detection.keypoints.get("right_eye")
        left_eye_pos = detection.keypoints.get("left_eye")
        mouth_left_pos = detection.keypoints.get("mouth_left")
        mouth_right_pos = detection.keypoints.get("mouth_right")
        nose_pos = detection.keypoints.get("nose")

        # Calculate average position of 5 main key points
        eye_avg_y = (left_eye_pos[1] + right_eye_pos[1]) / 2
        mouth_avg_y = (mouth_left_pos[1] + mouth_right_pos[1]) / 2
        d_eye_nose = nose_pos[1] - eye_avg_y
        d_nose_mouth = mouth_avg_y - nose_pos[1]
        # Return ratio
        return d_eye_nose / d_nose_mouth