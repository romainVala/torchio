import torch
from .map_metric import MapMetric
from .utils import spatial_filter_nd, gauss_kernel_3d
from ..data import Subject
from ..torchio import DATA


class NCC(MapMetric):

    def __init__(self, metric_name="NCC", **kwargs):
        super(NCC, self).__init__(metric_name=metric_name, **kwargs)

    def apply_metric(self, sample1: Subject, sample2: Subject):
        common_keys = self.get_common_intensity_keys(sample1=sample1, sample2=sample2)
        computed_metrics = dict()
        for sample_key in common_keys:
            if sample_key in self.mask_keys:
                continue
            data1 = sample1[sample_key][DATA]
            data2 = sample2[sample_key][DATA]
            computed_metrics[sample_key] = dict()
            if "metrics" not in sample2[sample_key].keys():
                sample2[sample_key]["metrics"] = dict()
            psnr_map = th_pearsonr(data1, data2)
            computed_metrics[sample_key]["metrics"]["{}".format(self.metric_name)] = psnr_map
        return computed_metrics


def _ncc(x, y):
    """
    This function is a torch implementation of the normalized cross correlation
    Parameters
    ----------
    x : torch.Tensor
    y : torch.Tensor

    Returns
    -------
    torch.Tensor
    """
    x_reshape, y_reshape = x.reshape(-1), y.reshape(-1)
    x_sub, y_sub = x_reshape - x_reshape.mean(), y_reshape - y_reshape.mean()
    x_normed, y_normed = x_sub/torch.norm(x), y_sub/torch.norm(y)
    return x_normed.dot(y_normed)


def inner_prod_ncc(x, y):
    x_normed, y_normed = (x - x.mean())/torch.norm(x), (y - y.mean())/torch.norm(y)
    x_normed, y_normed = x_normed.reshape(-1), y_normed.reshape(-1)
    return torch.dot(x_normed, y_normed).clamp(0.0, 1.0)


def th_pearsonr(x, y):
    """
    mimics scipy.stats.pearsonr
    """
    x = torch.flatten(x)
    y = torch.flatten(y)

    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val
