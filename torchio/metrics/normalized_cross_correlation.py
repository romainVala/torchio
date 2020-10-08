import torch
from typing import Union, List, Callable
from .map_metric_wrapper import MapMetricWrapper


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


class NCC(MapMetricWrapper):

    def __init__(self, metric_name: str = "NCC", metric_func: Callable = th_pearsonr,  select_key: Union[List, str] = None, scale_metric: float = 1,
                 average_method: str = None, save_in_subject_keys: bool = False, metric_kwargs: dict = None,
                 **kwargs):
        super(NCC, self).__init__(metric_name=metric_name, metric_func=metric_func, select_key=select_key,
                                  scale_metric=scale_metric, average_method=average_method,
                                  save_in_subject_keys=save_in_subject_keys, metric_kwargs=metric_kwargs, **kwargs)

