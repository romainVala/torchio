from abc import ABC, abstractmethod
from ..data.subject import Subject


class Metric(ABC):

    def __call__(self, sample1: Subject, sample2: Subject):
        return self.apply_metric(sample1=sample1, sample2=sample2)

    @abstractmethod
    def apply_metric(self, sample1: Subject, sample2: Subject):
        raise NotImplementedError
