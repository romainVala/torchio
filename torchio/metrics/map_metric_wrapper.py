from .map_metric import MapMetric
from ..data import Subject
from ..torchio import DATA


class MapMetricWrapper(MapMetric):

    def __init__(self, metric_name: str, metric_func, select_key=None, scale_metric=1, average_method=None,
                 save_in_subject_keys: bool = False):
        super(MapMetricWrapper, self).__init__(metric_name=metric_name, select_key=select_key,
                                               average_method=average_method, save_in_subject_keys=save_in_subject_keys)
        self.metric_func = metric_func
        self.scale_metric = scale_metric

    def apply_metric(self, sample1: Subject, sample2: Subject):
        common_keys = self.get_common_intensity_keys(sample1=sample1, sample2=sample2)
        computed_metrics = dict()
        for sample_key in common_keys:
            if sample_key in self.mask_keys:
                continue
            data1 = sample1[sample_key][DATA]
            data2 = sample2[sample_key][DATA]
            computed_metrics[sample_key] = dict()
            """
            if "metrics" not in sample2[sample_key].keys():
                sample2[sample_key]["metrics"] = dict()
            """
            metric_map = self.metric_func(data1, data2)

            metric_map = self._apply_masks_and_averaging(sample2, metric_map=metric_map)
            computed_metrics[sample_key] = metric_map
            """
            for mask_name, masked_metric in metric_map.items():
                if mask_name is "no_mask":
                    computed_metrics[self.metric_name] = masked_metric
                else:
                    computed_metrics[self.metric_name+"_"+mask_name] = masked_metric
            """
        return computed_metrics
