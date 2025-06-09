import numpy as np
from typing import Tuple

def intersection_area(boxA, boxB):
    xA = max(boxA[0], boxB[0])  # max of x_min
    yA = max(boxA[1], boxB[1])  # max of y_min
    xB = min(boxA[2], boxB[2])  # min of x_max
    yB = min(boxA[3], boxB[3])  # min of y_max

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)

    return inter_width * inter_height

def calculate_iou(boxA, boxB):
    # Compute intersection
    inter_area = intersection_area(boxA, boxB)

    # Compute area of both rectangles
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute union
    union = areaA + areaB - inter_area

    if union == 0:
        return 0

    return inter_area / union

def is_standard_image(frame :np.ndarray,
                      face_bbox :Tuple[int,int,int,int],
                      rect_width :int,
                      rect_height :int,
                      accepted_threshold :float = 0.25) -> bool:
    """
    Function  for deciding image with face is good enough for processing.
    Its must achieves minimum threshold for assuring quality.
    :param frame: Image input (np.ndarray)
    :param face_bbox: Bbox contains the coordination of a face
    :param rect_width:
    :param rect_height:
    :param accepted_threshold:
    :return:
    """
    # Get the face information
    h, w = frame.shape[:2]
    # Centered point of image
    centered_h, centered_w = int(h/2), int(w/2)
    # Centre box
    centered_bbox = (int(centered_w - rect_width/2),
                     int(centered_h - rect_height/2),
                     int(centered_w + rect_width/2),
                     int(centered_w + rect_height/2))
    # Calculate IOU
    iou = calculate_iou(centered_bbox,face_bbox)
    return True if iou > accepted_threshold else False


