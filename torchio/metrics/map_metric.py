from .base_metric import Metric
from ..data.subject import Subject
from ..torchio import DATA
from typing import Sequence, Union
from abc import ABC
import numpy as np
import torch
import warnings

average_func_dict = {"mean": torch.mean,
                     "sum": torch.sum,
                     }


class MapMetric(Metric, ABC):

    def __init__(self, mask_keys: Sequence = [], average_method: str = None):
        super(MapMetric, self).__init__()
        self.mask_keys = mask_keys
        self.average_method = self._get_average_method(average_method)

    @staticmethod
    def _get_average_method(average_method: str):
        if average_method is None:
            return None
        if average_method in average_func_dict.keys():
            return average_func_dict[average_method]
        else:
            warnings.warn("Unfound specified averaging method. Averaging methods available are: {}\nUsing mean function")
            return average_func_dict["mean"]

    def _apply_masks_and_averaging(self, sample: Subject, metric_map: Union[torch.Tensor, np.ndarray]):
        masked_map = metric_map
        if self.mask_keys:
            mask_keys_in_sample = sample.keys() & set(self.mask_keys)

            if len(mask_keys_in_sample) == 0:
                warnings.warn("None of the given masks {} found for the sample".format(self.mask_keys))
            else:

                for mask_key in mask_keys_in_sample:
                    mask_data = sample[mask_key][DATA]
                    masked_map = torch.mul(mask_data, metric_map)

        if self.average_method is not None:
            masked_map = self.average_method(masked_map)
        return masked_map
