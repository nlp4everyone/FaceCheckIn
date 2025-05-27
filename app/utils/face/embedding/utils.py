import torch
import torch.nn.functional as F
from typing import Literal, Union, List
import numpy as np

def calculate_similarity(source_vector :Union[torch.Tensor,List[float]],
                         ref_vector :Union[torch.Tensor,List[float]],
                         metrics :Literal["cosine"] = "cosine") -> torch.Tensor:
    # Validate input
    if isinstance(source_vector, list):
        # Convert to tensor
        source_vector = torch.tensor(source_vector)
        # Check if source vector must be 2D array
        if source_vector.ndim != 2:
            raise ValueError(f"Source vector must be 2D-array")

    # Validate input
    if isinstance(ref_vector, list):
        # Convert to tensor
        ref_vector = torch.tensor(ref_vector)
        # Check if source vector must be 2D array
        if ref_vector.ndim != 2:
            raise ValueError(f"Reference vector must be 2D-array")

    return F.cosine_similarity(source_vector, ref_vector, dim=1)