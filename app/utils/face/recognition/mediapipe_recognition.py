# Typing
from typing import Union, List
# Base Recognition
from .base_recognition import BaseRecognition, FaceDetection
# Other component
import cv2
import mediapipe as mp
import numpy as np

# Define face mesh
mp_face_detection = mp.solutions.face_detection

class MediapipeDetection(BaseRecognition):
    def __init__(self,
                 model_selection :int = 1,
                 min_detection_confidence: float = 0.5):
        super().__init__()
        self._face_detection = mp_face_detection.FaceDetection(model_selection = model_selection,
                                                               min_detection_confidence = min_detection_confidence)

    def detect_faces(self,
                     image: Union[str, np.ndarray],
                     limit: int = 1) -> List[FaceDetection]:
        # Convert the image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape

        results = self._face_detection.process(image)

        predictions = []
        for detection in results.detections:
            # Get confidence score
            score = detection.score[0]

            # Get normalized bounding box (values between 0 and 1)
            bbox = detection.location_data.relative_bounding_box
            x_min = bbox.xmin
            y_min = bbox.ymin
            box_width = bbox.width
            box_height = bbox.height

            # Convert to pixel coordinates
            x1 = int(x_min * width)
            y1 = int(y_min * height)
            x2 = int((x_min + box_width) * width)
            y2 = int((y_min + box_height) * height)
            predictions.append(FaceDetection(box = [x1,y1,x2,y2],
                                             confidence = score))

        return predictions
