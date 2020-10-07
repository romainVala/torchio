import torch
from typing import Callable, List, Union
from .map_metric import MapMetric
from .utils import spatial_filter_nd, gauss_kernel_3d


def _vif(x, y, kernel="gaussian", sigma=3.0, truncate=4.0):
    sigma_nsq = 2
    num = 0.0
    den = 0.0

    orig_ndim_x, orig_ndim_y = x.ndim, y.ndim
    eps = 1e-10

    if orig_ndim_x == 4:
        x = x.unsqueeze(0)

    if orig_ndim_y == 4:
        y = y.unsqueeze(0)

    if kernel == "gaussian":
        kernel_params = gauss_kernel_3d(sigma=sigma, truncate=truncate)
        kernel_params = kernel_params.reshape(1, 1, *kernel_params.shape).float()
    else:
        kernel_params = torch.ones((1, 1, 3, 3, 3))
        kernel_params /= 27

    mu_x, mu_y = spatial_filter_nd(x, kernel_params), spatial_filter_nd(y, kernel_params)
    x_sub, y_sub = x - mu_x, y - mu_y

    std_x, std_y = spatial_filter_nd(x_sub.pow(2), kernel=kernel_params).sqrt(), \
                   spatial_filter_nd(y_sub.pow(2), kernel=kernel_params).sqrt()

    std_xy = spatial_filter_nd(torch.mul(x_sub, y_sub), torch.ones_like(kernel_params))
    std_xy /= (kernel_params.numel() - 1)

    g = std_xy / (std_x**2 + eps)
    sv_sq = std_y**2 - g * std_xy

    num += torch.sum(torch.log10(1 + g * g * std_x**2 / (sv_sq + sigma_nsq)))
    den += torch.sum(torch.log10(1 + std_x**2 / sigma_nsq))

    vifp = num/den

    return vifp


class VIF(MapMetric):

    def __init__(self, metric_name: str = "VIF", metric_func: Callable = _vif, select_key: Union[List, str] = None, scale_metric: float = 1,
                 average_method: str="mean", save_in_subject_keys: bool = False, metric_kwargs: dict =
                 {"kernel": "gaussian", "truncate": 4.0, "sigma": 3.0}, **kwargs):
        super(VIF, self).__init__(metric_name=metric_name, metric_func=metric_func, select_key=select_key,
                                   scale_metric=scale_metric, average_method=average_method,
                                   save_in_subject_keys=save_in_subject_keys, metric_kwargs=metric_kwargs, **kwargs)

