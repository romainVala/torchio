import warnings
from typing import Tuple, Union

import torch

from ....data.subject import Subject
from ....torchio import DATA, TypeCallable
from . import NormalizationTransform


class ApplyMask(NormalizationTransform):
    """
    """
    def __init__(
            self,
            masking_method: Union[str, TypeCallable, None] = None,
            p: float = 1,
            **kwargs
            ):
        super().__init__(masking_method=masking_method, p=p, **kwargs)
        self.args_names = 'masking_method',

    def apply_normalization(
            self,
            sample: Subject,
            image_name: str,
            mask: torch.Tensor,
            ) -> None:
        image_dict = sample[image_name]
        image_dict[DATA][mask == 0] = torch.tensor(0).float()

