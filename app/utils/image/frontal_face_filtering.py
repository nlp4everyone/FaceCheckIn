from typing import List, Literal
import numpy as np
from app.utils.face.frontal_metrics import MediapipeMetric


class FrontalFaceFiltering:
    def __init__(self,
                 backends :Literal["mediapipe"] = "mediapipe"):
        if backends == "mediapipe":
            self._backend_metric = MediapipeMetric()
        else:
            raise NotImplementedError()

    def select_frames(self,
                      frames :List[np.ndarray],
                      top_k :int = 1):
        # Score for evaluating frontal
        frontal_scores = [self._backend_metric.calculate_frontalness_score(frame) for frame in frames]
        # Add index and remove None value
        indexed_scores = [(i,score) for (i, score) in enumerate(frontal_scores) if score is not None]
        # Sort value based in score descendingly:
        sorted_indexed_scores = sorted(indexed_scores,key = lambda x: x[1], reverse= True)

        # Select top-k element with highest score ( Most frontal)
        selected_scores = sorted_indexed_scores[:top_k]
        # Return frame with highest score
        return [frames[index] for (index, _) in selected_scores]