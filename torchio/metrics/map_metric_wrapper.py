from .map_metric import MapMetric
from ..data import Subject
from ..torchio import DATA


class MapMetricWrapper(MapMetric):

    def __init__(self, metric_name: str, metric_func, **kwargs):
        super(MapMetricWrapper, self).__init__(**kwargs)
        self.metric_name = metric_name
        self.metric_func = metric_func

    def apply_metric(self, sample1: Subject, sample2: Subject):
        common_keys = sample1.keys() & sample2.keys()
        for sample_key in common_keys:
            if sample_key in self.mask_keys:
                continue
            data1 = sample1[sample_key][DATA]
            data2 = sample2[sample_key][DATA]

            if "metrics" not in sample2[sample_key].keys():
                sample2[sample_key]["metrics"] = dict()
            metric_map = self.metric_func(data1, data2)

            metric_map = self._apply_masks_and_averaging(sample2, metric_map=metric_map)
            for mask_name, masked_metric in metric_map.items():
                if mask_name is "no_mask":
                    sample2[sample_key]["metrics"][self.metric_name] = masked_metric
                else:
                    sample2[sample_key]["metrics"][self.metric_name+"_"+mask_name] = masked_metric
