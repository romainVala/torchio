from ..torchio import DATA
from .base_metric import Metric
import torch


class MetricWrapper(Metric):

    def __init__(self, metric_name, metric_func, use_mask=False, mask_key=None, select_key=None):
        self.metric_name = metric_name
        self.metric_func = metric_func
        self.use_mask = use_mask
        self.mask_key = mask_key
        if isinstance(select_key, str):
            select_key = [select_key]
        self.select_key = select_key

    def apply_metric(self, sample1, sample2):
        if self.select_key is not None:
            common_keys = self.select_key
        else:
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

            #Compute metric
            if "metrics" not in sample2[sample_key].keys():
                sample2[sample_key]["metrics"] = dict()
            result = self.metric_func(data1, data2)
            if isinstance(result, dict):
                for key_metric, value_metric in result.items():
                    sample2[sample_key]["metrics"][self.metric_name+"_"+key_metric] = value_metric
            else:
                sample2[sample_key]["metrics"][self.metric_name] = result
