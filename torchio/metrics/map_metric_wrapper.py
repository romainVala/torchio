from .map_metric import MapMetric
from ..data import Subject
from ..torchio import DATA
from typing import Callable, Union, List


class MapMetricWrapper(MapMetric):

    def __init__(self, metric_name: str, metric_func: Callable, select_key: Union[List, str] = None,
                 scale_metric: float = 1, average_method: str = None, save_in_subject_keys: bool = False,
                 metric_kwargs: dict = None,  **kwargs):
        super(MapMetricWrapper, self).__init__(metric_name=metric_name, select_key=select_key,
                                               average_method=average_method, save_in_subject_keys=save_in_subject_keys,
                                               **kwargs)
        self.metric_func = metric_func
        self.scale_metric = scale_metric
        self.metric_kwargs = metric_kwargs

    def apply_metric(self, sample1: Subject, sample2: Subject):
        common_keys = self.get_common_intensity_keys(sample1=sample1, sample2=sample2)
        computed_metrics = dict()
        for sample_key in common_keys:
            if sample_key in self.mask_keys:
                continue
            data1 = sample1[sample_key][DATA]
            data2 = sample2[sample_key][DATA]
            computed_metrics[sample_key] = dict()
            metric_map = self.metric_func(data1, data2, **self.metric_kwargs) if self.metric_kwargs \
                else self.metric_func(data1, data2)

            metric_map = self._apply_masks_and_averaging(sample2, metric_map=metric_map)
            computed_metrics[sample_key] = metric_map

        return computed_metrics
