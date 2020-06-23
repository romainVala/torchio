from ..torchio import DATA
from .base_metric import Metric
import torch


class MetricWrapper(Metric):

    def __init__(self, metric_name, metric_func, use_mask=False, mask_key=None):
        self.metric_name = metric_name
        self.metric_func = metric_func
        self.use_mask = use_mask
        self.mask_key = mask_key

    def apply_metric(self, sample1, sample2):
        common_keys = sample1.keys() & sample2.keys()
        for sample_key in common_keys:
            if sample_key is self.mask_key:
                continue
            data1 = sample1[sample_key][DATA]
            data2 = sample2[sample_key][DATA]

            if self.use_mask and self.mask_key is not None:
                mask_data1, mask_data2 = sample1[self.mask_key][DATA], sample2[self.mask_key][DATA]
                data1 = torch.mul(data1, mask_data1)
                data2 = torch.mul(data2, mask_data2)
            #Apply mask if mask
            """
            Calculer la map puis moyenner
            """
            """
            if self.use_mask and self.mask_key is not None:
                mask_data = sample[self.mask_key][DATA]
                orig_data = torch.mul(orig_data, mask_data)
                transformed_data = torch.mul(transformed_data, mask_data)
            """
            #Compute metric
            if "metrics" not in sample2[sample_key].keys():
                sample2[sample_key]["metrics"] = dict()
            sample2[sample_key]["metrics"][self.metric_name] = self.metric_func(data1, data2)
