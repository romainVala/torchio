from abc import ABC, abstractmethod
from ..data.subject import Subject


class Metric(ABC):
    def __init__(self, metric_name: str):
        self.metric_name = metric_name

    def __call__(self, sample1: Subject, sample2: Subject):
        return self.apply_metric(sample1=sample1, sample2=sample2)

    @abstractmethod
    def apply_metric(self, sample1: Subject, sample2: Subject):
        raise NotImplementedError

    @staticmethod
    def get_common_intensity_keys(sample1: Subject, sample2: Subject):
        intensities1 = sample1.get_images_dict(intensity_only=True)
        intensities2 = sample2.get_images_dict(intensity_only=True)
        return intensities1.keys() & intensities2.keys()

