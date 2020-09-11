import torch
import math
import numpy as np
from torch.nn import Parameter
import torch.nn.functional as F
"""
Most of the functions (if not all of them) are taken from : https://github.com/yuta-hi/pytorch_similarity
"""

_func_conv_nd_table = {
    1: F.conv1d,
    2: F.conv2d,
    3: F.conv3d
}

def _gauss_1d(x, mu, sigma):
    return 1./(sigma * math.sqrt(2 * math.pi)) * torch.exp( - (x - mu)**2 / (2 * sigma**2) )

def gauss_kernel_1d(sigma, truncate=4.0):
    sd = float(sigma)
    lw = int(truncate * sd + 0.5)
    x = torch.arange(-lw, lw+1)
    kernel_1d = _gauss_1d(x, 0., sigma)
    return kernel_1d / kernel_1d.sum()

def gauss_kernel_2d(sigma, truncate=4.0):
    sd = float(sigma)
    lw = int(truncate * sd + 0.5)
    x = y = torch.arange(-lw, lw+1)
    X, Y = torch.meshgrid(x, y)
    kernel_2d = _gauss_1d(X, 0., sigma) \
              * _gauss_1d(Y, 0., sigma)
    return kernel_2d / kernel_2d.sum()

def gauss_kernel_3d(sigma, truncate=4.0):
    sd = float(sigma)
    lw = int(truncate * sd + 0.5)
    x = y = z = torch.arange(-lw, lw+1)
    X, Y, Z = torch.meshgrid(x, y, z)
    kernel_3d = _gauss_1d(X, 0., sigma) \
              * _gauss_1d(Y, 0., sigma) \
              * _gauss_1d(Z, 0., sigma)
    return kernel_3d / kernel_3d.sum()


# NOTE: Average kernel
def _average_kernel_nd(ndim, kernel_size):

    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * ndim

    kernel_nd = torch.ones(kernel_size)
    kernel_nd /= torch.sum(kernel_nd)

    return kernel_nd

def average_kernel_1d(kernel_size):
    return _average_kernel_nd(1, kernel_size)

def average_kernel_2d(kernel_size):
    return _average_kernel_nd(2, kernel_size)

def average_kernel_3d(kernel_size):
    return _average_kernel_nd(3, kernel_size)


# NOTE: Gradient kernel
def gradient_kernel_1d(method='default'):

    if method == 'default':
        kernel_1d = np.array([-1,0,+1])
    else:
        raise ValueError('unsupported method (got {})'.format(method))

    return kernel_1d

def gradient_kernel_2d(method='default', axis=0):

    if method == 'default':
        kernel_2d = np.array([[0,-1,0],
                              [0,0,0],
                              [0,+1,0]])
    elif method == 'sobel':
        kernel_2d = np.array([[-1,-2,-1],
                              [0,0,0],
                              [+1,+2,+1]])
    elif method == 'prewitt':
        kernel_2d = np.array([[-1,-1,-1],
                              [0,0,0],
                              [+1,+1,+1]])
    elif method == 'isotropic':
        kernel_2d = np.array([[-1,-np.sqrt(2),-1],
                              [0,0,0],
                              [+1,+np.sqrt(2),+1]])
    else:
        raise ValueError('unsupported method (got {})'.format(method))

    return np.moveaxis(kernel_2d, 0, axis)

def gradient_kernel_3d(method='default', axis=0):

    if method == 'default':
        kernel_3d = np.array([[[0, 0, 0],
                               [0, -1, 0],
                               [0, 0, 0]],
                              [[0, 0, 0],
                               [0, 0, 0],
                               [0, 0, 0]],
                              [[0, 0, 0],
                               [0, +1, 0],
                               [0, 0, 0]]])
    elif method == 'sobel':
        kernel_3d = np.array([[[-1, -3, -1],
                               [-3, -6, -3],
                               [-1, -3, -1]],
                              [[0, 0, 0],
                               [0, 0, 0],
                               [0, 0, 0]],
                              [[+1, +3, +1],
                               [+3, +6, +3],
                               [+1, +3, +1]]])
    elif method == 'prewitt':
        kernel_3d = np.array([[[-1, -1, -1],
                               [-1, -1, -1],
                               [-1, -1, -1]],
                              [[0, 0, 0],
                               [0, 0, 0],
                               [0, 0, 0]],
                              [[+1, +1, +1],
                               [+1, +1, +1],
                               [+1, +1, +1]]])
    elif method == 'isotropic':
        kernel_3d = np.array([[[-1, -1, -1],
                               [-1, -np.sqrt(2), -1],
                               [-1, -1, -1]],
                              [[0, 0, 0],
                               [0, 0, 0],
                               [0, 0, 0]],
                              [[+1, +1, +1],
                               [+1, +np.sqrt(2), +1],
                               [+1, +1, +1]]])
    else:
        raise ValueError('unsupported method (got {})'.format(method))

    return np.moveaxis(kernel_3d, 0, axis)


def spatial_filter_nd(x, kernel, mode='replicate'):
    """ N-dimensional spatial filter with padding.
    Args:
        x (~torch.Tensor): Input tensor.
        kernel (~torch.Tensor): Weight tensor (e.g., Gaussain kernel).
        mode (str, optional): Padding mode. Defaults to 'replicate'.
    Returns:
        ~torch.Tensor: Output tensor
    """

    n_dim = x.dim() - 2
    conv = _func_conv_nd_table[n_dim]

    pad = [None,None]*n_dim
    pad[0::2] = kernel.shape[2:]
    pad[1::2] = kernel.shape[2:]
    pad = [k//2 for k in pad]
    res = conv(F.pad(x, pad=pad, mode=mode), kernel)
    return res


def _grad_param(ndim, method, axis):

    if ndim == 1:
        kernel = gradient_kernel_1d(method)
    elif ndim == 2:
        kernel = gradient_kernel_2d(method, axis)
    elif ndim == 3:
        kernel = gradient_kernel_3d(method, axis)
    else:
        raise NotImplementedError

    kernel = kernel.reshape(1, 1, *kernel.shape)
    return Parameter(torch.Tensor(kernel).float())


def _gauss_param(ndim, sigma, truncate):

    if ndim == 1:
        kernel = gauss_kernel_1d(sigma, truncate)
    elif ndim == 2:
        kernel = gauss_kernel_2d(sigma, truncate)
    elif ndim == 3:
        kernel = gauss_kernel_3d(sigma, truncate)
    else:
        raise NotImplementedError

    kernel = kernel.reshape(1, 1, *kernel.shape)
    return Parameter(torch.Tensor(kernel).float())
