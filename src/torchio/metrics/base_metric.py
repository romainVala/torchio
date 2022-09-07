from typing import Union, List
from abc import ABC, abstractmethod
from ..data.subject import Subject


class Metric(ABC):
    def __init__(self, metric_name: str, select_key: Union[str, List] = None, scale_metric: float = 1.0,
                 save_in_subject_keys: bool = False):
        self.metric_name = metric_name
        if isinstance(select_key, str):
            select_key = [select_key]
        self.select_key = select_key
        self.scale_metric = scale_metric
        self.save_in_subject_keys = save_in_subject_keys

    def __call__(self, sample1: Subject, sample2: Subject):
        computed_metrics = self.apply_metric(sample1=sample1, sample2=sample2)
        for sample_key in computed_metrics.keys():
            computed_metrics[sample_key] = {k: (v*self.scale_metric).tolist() for k, v in computed_metrics[sample_key].items()}

        if self.save_in_subject_keys:
            for sample_key, metrics_sample in computed_metrics.items():
                    sample2[sample_key]["metrics"] = metrics_sample

        return computed_metrics

    @abstractmethod
    def apply_metric(self, sample1: Subject, sample2: Subject):
        raise NotImplementedError

    def get_common_intensity_keys(self, sample1: Subject, sample2: Subject):
        intensities1 = sample1.get_images_dict(intensity_only=True)
        intensities2 = sample2.get_images_dict(intensity_only=True)
        common_keys = intensities1.keys() & intensities2.keys()
        if self.select_key:
            return [key for key in self.select_key if key in common_keys]
        return common_keys

