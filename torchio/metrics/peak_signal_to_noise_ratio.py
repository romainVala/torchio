import torch
from typing import Union, List, Callable
from .map_metric_wrapper import MapMetricWrapper
import torch.nn.functional as F

def _nrmse(input, target, normalization='euclidean'):
    '''
    A Pytorch version of scikit-image's implementation of normalized_root_mse
    https://scikit-image.org/docs/dev/api/skimage.metrics.html#skimage.metrics.normalized_root_mse
    Compute the normalized root mean-squared error (NRMSE) between two
    images.
    Parameters
    ----------
    image_true : ndarray
        Ground-truth image, same shape as im_test.
    image_test : ndarray
        Test image.
    normalization : {'euclidean', 'min-max', 'mean'}, optional
        Controls the normalization method to use in the denominator of the
        NRMSE.  There is no standard method of normalization across the
        literature [1]_.  The methods available here are as follows:
        - 'euclidean' : normalize by the averaged Euclidean norm of
          ``im_true``::
              NRMSE = RMSE * sqrt(N) / || im_true ||
          where || . || denotes the Frobenius norm and ``N = im_true.size``.
          This result is equivalent to::
              NRMSE = || im_true - im_test || / || im_true ||.
        - 'min-max'   : normalize by the intensity range of ``im_true``.
        - 'mean'      : normalize by the mean of ``im_true``

    Returns
    -------
    nrmse : float
        The NRMSE metric.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Root-mean-square_deviation

    '''

    if normalization == "min-max":
        denom = input.max() - input.min()
    elif normalization == "mean":
        denom = input.mean()
    else:
        if normalization != "euclidean":
            raise Warning("Unsupported norm type. Found {}.\nUsing euclidean by default".format(self.normalization))
        denom = torch.sqrt(torch.mean(input ** 2))

    return (F.mse_loss(input, target).sqrt()) / denom


def _psnr(input, target, normalization='max'):
    """
    This function is a torch implementation of skimage.metrics.compare_psnr
    Parameters
    ----------
    input : torch.Tensor
    target : torch.Tensor
    norm : either max or mean

    Returns
    -------
    torch.Tensor
    """

    input_view = input.reshape(-1)
    target_view = target.reshape(-1)
    if normalization == 'mean':
        maximum_value = torch.mean(input_view)
    else:
        maximum_value = torch.max(input_view)

    mean_square_error = torch.mean((input_view - target_view) ** 2)
    psnrs = 20.0 * torch.log10(maximum_value) - 10.0 * torch.log10(mean_square_error)
    return psnrs


class PSNR(MapMetricWrapper):

    def __init__(self, metric_name: str = "PSNR", metric_func: Callable = _psnr, select_key: Union[List, str] = None, scale_metric: float = 1,
                 average_method: str = None, save_in_subject_keys: bool = False, metric_kwargs: dict = {"normalization":"max"}, **kwargs):
        super(PSNR, self).__init__(metric_name=metric_name, metric_func=metric_func, select_key=select_key,
                                   scale_metric=scale_metric, average_method=average_method,
                                   save_in_subject_keys=save_in_subject_keys, metric_kwargs=metric_kwargs, **kwargs)

class NRMSE(MapMetricWrapper):

    def __init__(self, metric_name: str = "NRMSE", metric_func: Callable = _nrmse, select_key: Union[List, str] = None, scale_metric: float = 1,
                 average_method: str = None, save_in_subject_keys: bool = False, metric_kwargs: dict = {"normalization":"euclidean"}, **kwargs):
        super(NRMSE, self).__init__(metric_name=metric_name, metric_func=metric_func, select_key=select_key,
                                   scale_metric=scale_metric, average_method=average_method,
                                   save_in_subject_keys=save_in_subject_keys, metric_kwargs=metric_kwargs, **kwargs)
