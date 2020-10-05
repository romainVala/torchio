import torch
from .map_metric import MapMetric
from .utils import spatial_filter_nd, gauss_kernel_3d
from ..data import Subject
from ..torchio import DATA


class SSIM3D(MapMetric):

    def __init__(self, metric_name="SSIM", k1=.001, k2=.001, k3=.001, L=None, alpha=1, beta=1, gamma=1, kernel="uniform", sigma=3.0,
                 truncate=4.0, **kwargs):
        super(SSIM3D, self).__init__(metric_name=metric_name, **kwargs)
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.L = L
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.kernel = kernel.lower()
        if self.kernel == "gaussian":
            self.sigma = sigma
            self.truncate = truncate
        else:
            self.sigma, self.truncate = None, None

    def apply_metric(self, sample1: Subject, sample2: Subject):
        common_keys = self.get_common_intensity_keys(sample1=sample1, sample2=sample2)
        computed_metrics_res = dict()
        for sample_key in common_keys:
            if sample_key in self.mask_keys:
                continue
            computed_metrics_res[sample_key] = dict()
            data1 = sample1[sample_key][DATA]
            data2 = sample2[sample_key][DATA]

            if "metrics" not in sample2[sample_key].keys():
                sample2[sample_key]["metrics"] = dict()
            computed_metrics = functional_ssim(data1, data2, k1=self.k1, k2=self.k2, k3=self.k3,
                                                    L=self.L, alpha=self.alpha, beta=self.beta, gamma=self.gamma,
                                                    kernel=self.kernel, sigma=self.sigma, truncate=self.truncate,
                                                    return_map=True)

            for m_name, m_map in computed_metrics.items():
                metric_dict = self._apply_masks_and_averaging(sample=sample2, metric_map=m_map)
                for mask_name, masked_metric in metric_dict.items():
                    if mask_name is "no_mask":
                        computed_metrics_res[sample_key]["{}_{}".format(self.metric_name, m_name)] = masked_metric
                    else:
                        computed_metrics_res[sample_key]["{}_{}_{}".format(self.metric_name, m_name, mask_name)] = masked_metric
        return computed_metrics_res


def functional_ssim(x, y, k1=.001, k2=.001, k3=.001, L=None, alpha=1, beta=1, gamma=1, kernel="uniform", sigma=3.0,
                    truncate=4.0, return_map=False):
    """
    Computes the structural similarity between x and y
    Args:
        x:
        y:
        k1:
        k2:
        k3:
        L:
        alpha:
        beta:
        gamma:
        kernel:
        sigma:
        truncate:
        return_map:

    Returns:

    """
    if not L:
        L = torch.max(torch.max(x), torch.max(y))

    orig_ndim_x, orig_ndim_y = x.ndim, y.ndim
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

    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    c3 = (k3 * L) ** 2

    mu_x, mu_y = spatial_filter_nd(x, kernel_params), spatial_filter_nd(y, kernel_params)
    x_sub, y_sub = x - mu_x, y - mu_y
    std_x, std_y = spatial_filter_nd(x_sub.pow(2), kernel=kernel_params).sqrt(), \
                   spatial_filter_nd(y_sub.pow(2), kernel=kernel_params).sqrt()
    std_xy = spatial_filter_nd(torch.mul(x_sub, y_sub), torch.ones_like(kernel_params))
    std_xy /= (kernel_params.numel() - 1)

    luminance = (2 * mu_x * mu_y + c1) / (mu_x ** 2 + mu_y ** 2 + c1)
    contrast = (2 * std_x * std_y + c2) / (std_x ** 2 + std_y ** 2 + c2)
    structure = (std_xy + c3) / ((std_x * std_y) + c3)
    #Keep values between 0.0 and 1.0
    luminance = luminance.clamp(0.0, 1.0)
    contrast = contrast.clamp(0.0, 1.0)
    structure = structure.clamp(0.0, 1.0)
    ssim = luminance ** alpha * contrast ** beta * structure ** gamma

    if not return_map:

        luminance = luminance.mean()
        contrast = contrast.mean()
        structure = structure.mean()
        ssim = ssim.mean()

    return {
        "luminance": luminance,
        "contrast": contrast,
        "structure": structure,
        "ssim": ssim
    }
