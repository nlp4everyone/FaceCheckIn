from facenet_pytorch import MTCNN
# Base Recognition
from .base_recognition import BaseRecognition, FaceDetection, FacialKeyPoints
# Other component
from typing import Literal, Union, List
import numpy as np
# Default threshold
DEFAULT_THRESHOLD = [0.7, 0.7, 0.8]

class MTCNNRecognition(BaseRecognition):
    def __init__(self,
                 image_size :int = 160,
                 device :Union[Literal["cpu","cuda:0"],str] = "cpu",
                 post_process :bool = True,
                 keep_all :bool = False,
                 select_largest :bool = False):
        super().__init__()
        # Define MTCNN
        self._detector = MTCNN(thresholds = DEFAULT_THRESHOLD,
                               image_size = image_size,
                               keep_all = keep_all,
                               device = device,
                               select_largest = select_largest,
                               post_process = post_process)

    def detect_faces(self,
                     image: Union[str, np.ndarray],
                     limit: int = 1) -> List[FaceDetection]:
        # Detect
        boxes, probs, landmarks = self._detector.detect(image,
                                                        landmarks = True,)
        # Raise exceptions while not found face
        if boxes is None or probs is None:
            return []

        predictions = []
        # Iterate each
        for (box, prob, landmark) in zip(boxes, probs, landmarks):
            # Convert ndarray to list of int
            box = [int(element) for element in box.tolist()]
            landmark = [[int(x), int(y)] for x, y in landmark.tolist()]

            # Append values
            predictions.append(FaceDetection(box=box,
                                             confidence = prob,
                                             keypoints = FacialKeyPoints(left_eye = list(landmark[0]),
                                                                         right_eye = list(landmark[1]),
                                                                         nose = list(landmark[2]),
                                                                         left_mouth = list(landmark[3]),
                                                                         right_mouth = list(landmark[4]))))
        return predictions

    def batch_detect_faces(self,
                           images: List[Union[str, np.ndarray]]) -> List[List[FaceDetection]]:
        if isinstance(images, np.ndarray): images = [images]
        # Return
        return [self.detect_faces(image) for image in images]