import torch
from typing import Union, List, Callable
from .map_metric_wrapper import MapMetricWrapper
import torch.nn.functional as F
import numpy as np
import math


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

def normalize_cc(x, y, zero_mean=True):
    """
    very similar as th_perasonr, but ... different ...
    """
    x = torch.flatten(x)
    y = torch.flatten(y)

    if zero_mean:
        x = x.sub(x.mean())
        y = y.sub(y.mean())

    r_num = x.dot(y) / x.shape[0]
    r_den = torch.std(x) * torch.std(y)
    r_val = r_num / r_den
    return r_val


class NCC(MapMetricWrapper):

    def __init__(self, metric_name: str = "NCC", metric_func: Callable = normalize_cc,  select_key: Union[List, str] = None, scale_metric: float = 1,
                 average_method: str = None, save_in_subject_keys: bool = False, metric_kwargs: dict = None,
                 **kwargs):
        super(NCC, self).__init__(metric_name=metric_name, metric_func=metric_func, select_key=select_key,
                                  scale_metric=scale_metric, average_method=average_method,
                                  save_in_subject_keys=save_in_subject_keys, metric_kwargs=metric_kwargs, **kwargs)


def _ncc_conv(y_true, y_pred, win=None):
    #je comprend pas cette implementation, car les regions avec des valeurs nulle, vont donner un cc de zero
    #long a calculer en cpu ... bof bof bof
    I = y_true.unsqueeze(0)
    J = y_pred.unsqueeze(0)

    # get dimension of volume
    # assumes I, J are sized [batch_size, *vol_shape    #rrr , nb_feats]
    ndims = len(list(I.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

    # set window size
    win = [9] * ndims if win is None else win

    # compute filters
    sum_filt = torch.ones([1, 1, *win]) #.to("cuda") ##rr modif only dim #rrr make conv complain, ... may be the data are not in cuda TODO

    padding = math.floor(win[0]/2)  #I guess to have the same ouput size for cc, but since the mean
    stride = 1

    # get convolution function
    conv_fn = getattr(F, 'conv%dd' % ndims)

    # compute CC squares
#for test def CC(I,J, conv_fn=F.conv3d,stride=1, padding=4, sum_filt = torch.ones([1, 1, 9,9,9]), win=[9,9,9]):
    I2 = I * I
    J2 = J * J
    IJ = I * J

    I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
    J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
    I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
    J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
    IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    cc = cross * cross / (I_var * J_var + 1e-5)
    return cc
    #return torch.mean(cc)


class NCC_conv(MapMetricWrapper):
    """
    Local (over window) normalized cross correlation loss.
    """
    def __init__(self, metric_name: str = "NCC_conv", metric_func: Callable = _ncc_conv, select_key: Union[List, str] = None,
                 scale_metric: float = 1,
                 average_method: str = None, save_in_subject_keys: bool = False, metric_kwargs: dict =
                 {"win": None}, **kwargs):

        super(NCC_conv, self).__init__(metric_name=metric_name, metric_func=metric_func, select_key=select_key,
                                   scale_metric=scale_metric, average_method=average_method,
                                   save_in_subject_keys=save_in_subject_keys, metric_kwargs=metric_kwargs, **kwargs)


