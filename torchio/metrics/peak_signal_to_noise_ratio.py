import torch
from .map_metric import MapMetric
from ..data import Subject
from ..torchio import DATA


class PSNR(MapMetric):

    def __init__(self, metric_name="PSNR", **kwargs):
        super(PSNR, self).__init__(metric_name=metric_name, **kwargs)

    def apply_metric(self, sample1: Subject, sample2: Subject):
        common_keys = self.get_common_intensity_keys(sample1=sample1, sample2=sample2)

        for sample_key in common_keys:
            if sample_key in self.mask_keys:
                continue
            data1 = sample1[sample_key][DATA]
            data2 = sample2[sample_key][DATA]

            if "metrics" not in sample2[sample_key].keys():
                sample2[sample_key]["metrics"] = dict()
            psnr = _psnr(data1, data2)
            sample2[sample_key]["metrics"]["{}".format(self.metric_name)] = psnr


def _psnr(input, target):
    """
    This function is a torch implementation of skimage.metrics.compare_psnr
    Parameters
    ----------
    input : torch.Tensor
    target : torch.Tensor

    Returns
    -------
    torch.Tensor
    """

    input_view = input.reshape(-1)
    target_view = target.reshape(-1)
    maximum_value = torch.max(input_view)

    mean_square_error = torch.mean((input_view - target_view) ** 2)
    psnrs = 20.0 * torch.log10(maximum_value) - 10.0 * torch.log10(mean_square_error)
    return psnrs
