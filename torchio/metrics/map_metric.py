from .base_metric import Metric
from ..data.subject import Subject
from ..constants import DATA
from typing import Sequence, Union
from abc import ABC
import numpy as np
import torch
import warnings

average_func_dict = {"mean": torch.mean,
                     "sum": torch.sum,
                     }


class MapMetric(Metric, ABC):

    def __init__(self, metric_name: str, mask_keys: Sequence = [], average_method: str = None, select_key: str = None,
                 scale_metric: float = 1.0, save_in_subject_keys: bool = False):
        super(MapMetric, self).__init__(metric_name=metric_name, select_key=select_key, scale_metric=scale_metric,
                                        save_in_subject_keys=save_in_subject_keys)
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
        masked_maps[self.metric_name] = orig_map
        #Check if masks
        if self.mask_keys:
            mask_keys_in_sample = sample.keys() & set(self.mask_keys)

            if len(mask_keys_in_sample) == 0:
                warnings.warn("None of the given masks {} found for the sample".format(self.mask_keys))
            else:
                for mask_key in mask_keys_in_sample:
                    mask_data = sample[mask_key][DATA]
                    if mask_data.shape[0] > metric_map.shape[0] : #case of a 4D mask, make all label
                        if self.average_method is None: #for global metrics without average
                            for num_label, label_mask in enumerate(mask_data):
                                masked_map = metric_map[label_mask.unsqueeze(0) > 0] #todo there should be a threshold to get mask data
                                masked_maps[self.metric_name + "_" + mask_key + f'_L{num_label}'] = masked_map
                        else: #make a weighted mean Warning do not take the average_method
                            weighted_map = metric_map.repeat((mask_data.shape[0], 1, 1, 1))* mask_data
                            weighted_means = weighted_map.sum(dim=(1,2,3)) /  mask_data.sum(dim=(1,2,3))
                            for num_label in range(0, mask_data.shape[0]):
                                masked_maps[self.metric_name + "_" + mask_key + f'_L{num_label}'] = weighted_means[num_label]
                    else:
                        masked_map = metric_map[mask_data > 0]
                        if self.average_method is not None:
                            masked_map = self.average_method(masked_map)
                        masked_maps[self.metric_name + "_" + mask_key] = masked_map
        return masked_maps
