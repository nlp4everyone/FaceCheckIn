from typing import Union, List
import numpy as np
import torch

class BaseEmbedding:
    @property
    def model_name(self):
        raise NotImplementedError()

    def embed(self,
              images :Union[np.ndarray, List[np.ndarray]]) -> torch.Tensor:
        raise NotImplementedError()