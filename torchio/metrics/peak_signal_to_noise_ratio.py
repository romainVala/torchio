import torch
from typing import Union, List, Callable
from .map_metric_wrapper import MapMetricWrapper
import torch.nn.functional as F
import numpy as np
from ..data import Subject
from ..constants import DATA
from scipy import ndimage

EPS = np.finfo(float).eps

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

def _entropy(x):
    #simple definition from autofocusing literature
    x = x.ravel()
    x = abs(x) #log need positiv number !
    x = x[abs(x)>1e-6]  #avoid 0 values
    bmax = np.square(np.sum(x**2)) #np.sum(x**2) #change le 9 02 2021
    proba = x / bmax
    return - np.sum( np.log(x)*x )

def nmi(x, y, **kwargs):
    """ Mutual information for joint histogram Elina
    """
    # Convert bins counts to probability values
    bins = (256, 256)
    hist_inter, _, _ = np.histogram2d(x.ravel(), y.ravel(), bins=bins)
    hist1, _, _ = np.histogram2d(x.ravel(), x.ravel(), bins=bins)
    hist2, _, _ = np.histogram2d(y.ravel(), y.ravel(), bins=bins)

    return 2 * _mutual_information(hist_inter) / (
            _mutual_information(hist1) + _mutual_information(hist2))

def _mutual_information(hgram):
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1)  # marginal for x over y
    py = np.sum(pxy, axis=0)  # marginal for y over x
    px_py = px[:, None] * py[None, :]  # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0  # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def mutual_information_2d(x, y, sigma=1, normalized=True):
    """
    Computes (normalized) mutual information between two 1D variate from a
    joint histogram.
    Parameters
    ----------
    x : 1D array
        first variable
    y : 1D array
        second variable
    sigma: float
        sigma for Gaussian smoothing of the joint histogram
    Returns
    -------
    nmi: float
        the computed similariy measure
    """
    bins = (256, 256)

    jh = np.histogram2d(x.ravel(), y.ravel(), bins=bins)[0]

    # smooth the jh with a gaussian filter of given sigma
    ndimage.gaussian_filter(jh, sigma=sigma, mode='constant',
                                 output=jh)

    # compute marginal histograms
    jh = jh + EPS
    sh = np.sum(jh)
    jh = jh / sh
    s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
    s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))

    # Normalised Mutual Information of:
    # Studholme,  jhill & jhawkes (1998).
    # "A normalized entropy measure of 3-D medical image alignment".
    # in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.
    if normalized:
        mi = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2)))
                / np.sum(jh * np.log(jh))) - 1
    else:
        mi = ( np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1))
               - np.sum(s2 * np.log(s2)))

    return 2*mi


def _get_autocor(data, nb_off_center = 3):
    data = (data - torch.mean(data))
    data = data / torch.std(data)
    N = torch.prod(torch.tensor(data.shape)).numpy()
    data = data.numpy()

    tfi = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(data))).astype(np.complex128) / N
    freq_domain = tfi * np.conjugate(tfi)

    output = np.fft.ifftshift(np.fft.ifftn(freq_domain) * N)
    #plt.figure();    plt.plot(np.abs(output.flatten()))

    d = np.array(data.shape)
    d_center = d // 2
    dfcent = np.abs(
        output[d_center[0] - nb_off_center:d_center[0] + nb_off_center+1,
        d_center[1] - nb_off_center:d_center[1] + nb_off_center+1,
        d_center[2] - nb_off_center:d_center[2] + nb_off_center+1])
    aa = np.mgrid[-nb_off_center:nb_off_center+1, -nb_off_center:nb_off_center+1, -nb_off_center:nb_off_center+1]


    [i1, i2, i3] = np.meshgrid(np.arange(nb_off_center*2+1)-nb_off_center,  np.arange(nb_off_center*2+1)-nb_off_center,
                               np.arange(nb_off_center*2+1)-nb_off_center)
    dist_ind=np.sqrt(i1**2+i2**2+i3**2)
    dist_flat = dist_ind.flatten()
    unique_dist = np.unique(dist_flat)
    cor_coef = np.zeros_like(unique_dist)
    for iind, one_dist in enumerate(unique_dist):
        ii = dist_ind==one_dist
        cor_coef[iind] = np.mean(dfcent[ii])

    correlation_slope, b = np.polyfit(unique_dist[1:6], cor_coef[1:6], 1)
    corelation1 = cor_coef[unique_dist==1][0]
    corelation2 = cor_coef[unique_dist==2][0]
    corelation3 = cor_coef[unique_dist==3][0]
    return corelation1, corelation2, corelation3, correlation_slope

def _grad_ratio(input,target, do_scat=False, do_nmi=True, do_entropy=True, do_autocorr=True):
    #print(f' i shape {input.shape}')
     #not sure how to handel batch size (first dim) TODO
    input = input[0]
    target = target[0]

    grad_i = np.gradient(input)
    grad_t = np.gradient(target)

    grad_sum_i=np.zeros_like(grad_i[0])
    for gg in grad_i :
        grad_sum_i += np.abs(gg)

    grad_sum_t=np.zeros_like(grad_t[0])
    for gg in grad_t :
        grad_sum_t += np.abs(gg)

    #grad_sum_i = np.sum([np.sum(np.abs(gg)) for gg in grad_i])
    #grad_sum_t = np.sum([np.sum(np.abs(gg)) for gg in grad_t])
    grad_mean_i = np.mean(grad_sum_i)
    grad_mean_t = np.mean(grad_sum_t)

    #mean only on edge ... (like AES metric)
    grad_mean_edge_i = np.mean( grad_sum_i[grad_sum_i > 0.01])
    grad_mean_edge_t = np.mean( grad_sum_t[grad_sum_t > 0.01])

    res_dict=dict()
    res_dict['ratio'] = grad_mean_i/grad_mean_t
    res_dict['ratio_bin'] = grad_mean_edge_i/grad_mean_edge_t

    if do_scat:
        from kymatio import HarmonicScattering3D
        import time
        data =  dd= torch.stack([input,target])
        data_shape = data.shape[1:]
        print(data_shape)
        J = 2;        L = 2;        integral_powers = [1., 2.];        sigma_0 = 1
        scattering = HarmonicScattering3D(J, shape=data_shape, L=L, sigma_0=sigma_0)
        scattering.method = 'integral'
        scattering.integral_powers = integral_powers
        s=time.time()
        res = scattering(data)
        s=time.time()-s
        print(f'scat in {s}')
        res_dict['scat'] = torch.norm(res[0]-res[1])
    if do_nmi:
        res_dict['nMI1'] = nmi(input.numpy(), target.numpy())
        res_dict['nMI2'] = mutual_information_2d(input.numpy(), target.numpy())
    if do_entropy:
        res_dict['Eorig']  = _entropy(input.numpy())
        res_dict['Emot']   = _entropy(target.numpy())
        # res_dict['Emot2'] = nmi(target.numpy(), target.numpy()) #faudrait une version non normalise ...
        res_dict['Eratio'] = res_dict['Eorig'] / res_dict['Emot']
        entro_grad1 = np.sum([ _entropy(np.abs(gg)) for gg in grad_i])
        entro_grad2 = np.sum([ _entropy(np.abs(gg)) for gg in grad_t])
        res_dict['EGratio'] = entro_grad1 / entro_grad2
    if do_autocorr:
        c1, c2, c3, cdiff = _get_autocor(input, nb_off_center=3)
        c1m, c2m, c3m, cdiffm = _get_autocor(target, nb_off_center=3)
        res_dict['cor1_ratio'] = c1 / c1m
        res_dict['cor2_ratio'] = c2 / c2m
        res_dict['cor3_ratio'] = c3 / c3m
        res_dict['cor_diff_ratio'] = cdiffm / cdiff
        res_dict['cor1_orig'] = c1
        res_dict['cor2_orig'] = c2
        res_dict['cor3_orig'] = c3
        res_dict['cor_diff_orig'] = cdiff

    return res_dict

def _identity():
    pass

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
class GradRatio(MapMetricWrapper):

    def __init__(self, metric_name: str = "Gratio", metric_func: Callable = _grad_ratio, select_key: Union[List, str] = None, scale_metric: float = 1,
                 average_method: str = None, save_in_subject_keys: bool = False, metric_kwargs: dict = None, **kwargs):
        super(GradRatio, self).__init__(metric_name=metric_name, metric_func=metric_func, select_key=select_key,
                                   scale_metric=scale_metric, average_method=average_method,
                                   save_in_subject_keys=save_in_subject_keys, metric_kwargs=metric_kwargs, **kwargs)


    def apply_metric(self, sample1: Subject, sample2: Subject):
        common_keys = self.get_common_intensity_keys(sample1=sample1, sample2=sample2)
        computed_metrics_res = dict()
        for sample_key in common_keys:
            if sample_key in self.mask_keys:
                continue
            computed_metrics_res[sample_key] = dict()
            data1 = sample1[sample_key][DATA]
            data2 = sample2[sample_key][DATA]
            computed_metrics = self.metric_func(data1, data2)

            for m_name, m_map in computed_metrics.items():
                #metric_dict = self._apply_masks_and_averaging(sample=sample2, metric_map=m_map)
                #for mask_name, masked_metric in metric_dict.items():
                #    if mask_name == "no_mask":
                computed_metrics_res[sample_key]["{}_{}".format(self.metric_name, m_name)] = m_map#masked_metric
                #    else:
                #        computed_metrics_res[sample_key]["{}_{}".format( m_name, mask_name)] = masked_metric
        return computed_metrics_res


class LabelMetric(MapMetricWrapper):

    def __init__(self, metric_name: str = "lab", metric_func: Callable = _identity, select_key: Union[List, str] = None, scale_metric: float = 1,
                 average_method: str = None, save_in_subject_keys: bool = False, metric_kwargs: dict = None, **kwargs):
        super(LabelMetric, self).__init__(metric_name=metric_name, metric_func=metric_func, select_key=select_key,
                                   scale_metric=scale_metric, average_method=average_method,
                                   save_in_subject_keys=save_in_subject_keys, metric_kwargs=metric_kwargs, **kwargs)

    def apply_metric(self, sample1: Subject, sample2: Subject):
        common_keys = self.get_common_intensity_keys(sample1=sample1, sample2=sample2)
        computed_metrics_res = dict()
        for sample_key in common_keys:
            if sample_key in self.mask_keys:
                continue
            computed_metrics_res[sample_key] = dict()
            data1 = sample1[sample_key][DATA][0] #arg todo if batch size >1 ... ?
            data2 = sample2[sample_key][DATA][0]
            labels = sample1['label'][DATA]
            ind = labels>0.5 #binarize

            mean_ratio = [data1[li].mean()/data2[li].mean() for li in ind]
            std_ratio = [data1[li].std()/data2[li].std() for li in ind]

            for i, (one_mean, one_std) in enumerate(zip(mean_ratio,std_ratio)):
                computed_metrics_res[sample_key]["{}_mean{}".format(self.metric_name, i)] = one_mean
                computed_metrics_res[sample_key]["{}_std{}".format(self.metric_name, i)] = one_std

        return computed_metrics_res
