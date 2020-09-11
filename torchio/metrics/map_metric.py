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

    def __init__(self, metric_name: str, mask_keys: Sequence = [], average_method: str = None, select_key: str = None):
        super(MapMetric, self).__init__(metric_name=metric_name)
        self.mask_keys = mask_keys
        self.average_method = self._get_average_method(average_method)
        self.select_key = select_key

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
        if metric_map.ndim == 5:
            #We assume the first dimension is batch and the second is channel. We keep the batch dimension and discard
            # the channel dimension
            metric_map = metric_map[:, 0]
        #Dictionary with all maps+average
        masked_maps = dict()
        #Adding the original metric map (without mask)
        orig_map = metric_map
        if self.average_method is not None:
            orig_map = self.average_method(orig_map)
        masked_maps["no_mask"] = orig_map
        #Check if masks
        if self.mask_keys:
            mask_keys_in_sample = sample.keys() & set(self.mask_keys)

            if len(mask_keys_in_sample) == 0:
                warnings.warn("None of the given masks {} found for the sample".format(self.mask_keys))
            else:
                for mask_key in mask_keys_in_sample:
                    mask_data = sample[mask_key][DATA]
                    masked_map = metric_map[mask_data > 0]
                    if self.average_method is not None:
                        masked_map = self.average_method(masked_map)
                    masked_maps[mask_key] = masked_map
        return masked_maps
