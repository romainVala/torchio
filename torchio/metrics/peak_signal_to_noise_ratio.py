import torch
from typing import Union, List, Callable
from .map_metric_wrapper import MapMetricWrapper


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


class PSNR(MapMetricWrapper):

    def __init__(self, metric_name: str = "PSNR", metric_func: Callable = _psnr, select_key: Union[List, str] = None, scale_metric: float = 1,
                 average_method: str = None, save_in_subject_keys: bool = False, metric_kwargs: dict = None, **kwargs):
        super(PSNR, self).__init__(metric_name=metric_name, metric_func=metric_func, select_key=select_key,
                                   scale_metric=scale_metric, average_method=average_method,
                                   save_in_subject_keys=save_in_subject_keys, metric_kwargs=metric_kwargs, **kwargs)
