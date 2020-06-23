from numbers import Number
from typing import Tuple, Optional, List, Union
import torch
import numpy as np
import SimpleITK as sitk
from ....data.subject import Subject
from ....torchio import LABEL, DATA, AFFINE, TYPE, TypeRangeFloat, STEM
from .. import Interpolation, get_sitk_interpolator
from .. import RandomTransform


class RandomAffineFFT(RandomTransform):
    r"""Random affine transformation.

    Args:
        scales: Tuple :math:`(a, b)` defining the scaling
            magnitude. The scaling values along each dimension are
            :math:`(s_1, s_2, s_3)`, where :math:`s_i \sim \mathcal{U}(a, b)`.
            For example, using ``scales=(0.5, 0.5)`` will zoom out the image,
            making the objects inside look twice as small while preserving
            the physical size and position of the image.
        degrees: Tuple :math:`(a, b)` defining the rotation range in degrees.
            The rotation angles around each axis are
            :math:`(\theta_1, \theta_2, \theta_3)`,
            where :math:`\theta_i \sim \mathcal{U}(a, b)`.
            If only one value :math:`d` is provided,
            :math:`\theta_i \sim \mathcal{U}(-d, d)`.
        isotropic: If ``True``, the scaling factor along all dimensions is the
            same, i.e. :math:`s_1 = s_2 = s_3`.
        default_pad_value: As the image is rotated, some values near the
            borders will be undefined.
            If ``'minimum'``, the fill value will be the image minimum.
            If ``'mean'``, the fill value is the mean of the border values.
            If ``'otsu'``, the fill value is the mean of the values at the
            border that lie under an
            `Otsu threshold <https://ieeexplore.ieee.org/document/4310076>`_.
        p: Probability that this transform will be applied.
        seed: See :py:class:`~torchio.transforms.augmentation.RandomTransform`.

    .. note:: Rotations are performed around the center of the image.

    Example:
        >>> from torchio.transforms import RandomAffine, Interpolation
        >>> sample = images_dataset[0]  # instance of torchio.ImagesDataset
        >>> transform = RandomAffine(
        ...     scales=(0.9, 1.2),
        ...     degrees=(10),
        ...     isotropic=False,
        ...     default_pad_value='otsu',
        ... )
        >>> transformed = transform(sample)

    From the command line::

        $ torchio-transform t1.nii.gz RandomAffine --kwargs "degrees=30 default_pad_value=minimum" --seed 42 affine_min.nii.gz

    """
    def __init__(
            self,
            scales: Tuple[float, float] = (0.9, 1.1),
            degrees: TypeRangeFloat = 10,
            isotropic: bool = False,
            default_pad_value: Union[str, float] = 'otsu',
            p: float = 1,
            seed: Optional[int] = None,
            oversampling_pct = 0.2,
            **kwargs
            ):
        super().__init__(p=p, seed=seed, **kwargs)
        self.scales = scales
        self.degrees = self.parse_degrees(degrees)
        self.isotropic = isotropic
        self.default_pad_value = self.parse_default_value(default_pad_value)
        self.oversampling_pct = oversampling_pct

    @staticmethod
    def parse_default_value(value: Union[str, float]) -> Union[str, float]:
        if isinstance(value, Number) or value in ('minimum', 'otsu', 'mean'):
            return value
        message = (
            'Value for default_pad_value must be "minimum", "otsu", "mean"'
            ' or a number'
        )
        raise ValueError(message)

    def apply_transform(self, sample: Subject) -> dict:
        sample.check_consistent_shape()
        scaling_params, rotation_params = self.get_params(
            self.scales, self.degrees, self.isotropic)
        random_parameters_dict = {
            'scaling': scaling_params,
            'rotation': rotation_params,
            'oversampling' : self.oversampling_pct
        }
        for image_dict in sample.get_images(intensity_only=False):
            if image_dict[TYPE] == LABEL:
                padding_values = [0]
            else:
                padding_values = estimate_borders_mean_std(image_dict[DATA].numpy())
                add_name = image_dict[STEM] if image_dict[STEM] is not None else '' #because when called directly with tensor, it is not define hmmm...
                random_parameters_dict['noise_mean_' + add_name] = padding_values[0]
                random_parameters_dict['noise_std_' + add_name] = padding_values[1]

            image_dict[DATA] = self.apply_affine_transform(
                image_dict[DATA],
                scaling_params,
                rotation_params,
                padding_values,
            )
        sample.add_transform(self, random_parameters_dict)
        return sample

    @staticmethod
    def get_params(
            scales: Tuple[float, float],
            degrees: Tuple[float, float],
            isotropic: bool,
            ) -> Tuple[List[float], List[float]]:
        scaling_params = torch.FloatTensor(3).uniform_(*scales)
        if isotropic:
            scaling_params.fill_(scaling_params[0])
        rotation_params = torch.FloatTensor(3).uniform_(*degrees)
        return scaling_params.tolist(), rotation_params.tolist()

    def apply_affine_transform(
            self,
            tensor: torch.Tensor,
            scaling_params: List[float],
            rotation_params: List[float],
            padding_values: List[float]
            ) -> torch.Tensor:
        assert tensor.ndim == 4
        assert len(tensor) == 1

        from torchio.transforms.augmentation.intensity.random_motion_from_time_course import create_rotation_matrix_3d
        import math
        import finufftpy

        image = tensor[0]
        #noise_mean, nois_std = estimate_borders_mean_std(np.abs(image.numpy())) #random_noise gives negativ values ...
        noise_mean, nois_std = estimate_borders_mean_std(image.numpy())
        original_image_shape = image.shape
        if self.oversampling_pct > 0.0:
            if len(padding_values) == 2: #mean std
                padd_mode = 'random.normal'
            else:
                padd_mode = 'constant'

            image = self._oversample(image, self.oversampling_pct, padding_mode=padd_mode,
                                     padding_normal=padding_values)

        #im_freq_domain = (np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(image)))).astype(np.complex128)
        im_freq_domain = self._fft_im(image)
        #if self.oversampling_pct > 0.0:
        #    im_freq_domain = self._oversample(im_freq_domain, self.oversampling_pct,
        #                                      padding_mode='random.normal', padding_normal=(noise_mean, nois_std))


        rrrot = -np.radians(rotation_params); rrrot[1] = - rrrot[1] #to get the same as sitk ... hmmm
        rotation_matrices = create_rotation_matrix_3d(rrrot)
        scaling_matrices = np.eye(3) / np.array(scaling_params) #/ to have same convention as
        rotation_matrices = np.matmul(rotation_matrices, scaling_matrices)
        im_shape = im_freq_domain.shape

        center = [math.ceil((x - 1) / 2) for x in im_shape]

        [i1, i2, i3] = np.meshgrid(2*(np.arange(im_shape[0]) - center[0])/im_shape[0],
                                   2*(np.arange(im_shape[1]) - center[1])/im_shape[1],
                                   2*(np.arange(im_shape[2]) - center[2])/im_shape[2], indexing='ij')

        grid_coordinates = np.array([i1.flatten('F'), i2.flatten('F'), i3.flatten('F')])

        method='one_matrix'
        if method=='one_matrix':
            new_grid_coords = np.matmul(rotation_matrices, grid_coordinates)
        else: #grrr marche pas ! (inspirer de random_motion_from_time_course
            rotation_matrices = np.expand_dims(rotation_matrices, [2, 3, 4])
            rotation_matrices = np.tile(rotation_matrices, [1, 1] + list(im_shape))  # 3 x 3 x img_shape
            rotation_matrices = rotation_matrices.reshape([-1, 3, 3], order='F')

            # tile grid coordinates for vectorizing computation
            grid_coordinates_tiled = np.tile(grid_coordinates, [3, 1])
            grid_coordinates_tiled = grid_coordinates_tiled.reshape([3, -1], order='F').T
            rotation_matrices = rotation_matrices.reshape([-1, 3]) #reshape for matrix multiplication, so no order F

            new_grid_coords = (rotation_matrices * grid_coordinates_tiled).sum(axis=1)
            # reshape new grid coords back to 3 x nvoxels
            new_grid_coords = new_grid_coords.reshape([3, -1], order='F')

        # scale data between -pi and pi
        max_vals = [1, 1, 1]
        new_grid_coordinates_scaled = [(new_grid_coords[i, :] / max_vals[i]) * math.pi for i in [0, 1, 2]]

        # initialize array for nufft output
        f = np.zeros([len(new_grid_coordinates_scaled[0])], dtype=np.complex128, order='F')

        freq_domain_data_flat = np.asfortranarray(im_freq_domain.flatten(order='F'))
        iflag, eps = 1,  1E-7
        finufftpy.nufft3d1(new_grid_coordinates_scaled[0], new_grid_coordinates_scaled[1],
                           new_grid_coordinates_scaled[2], freq_domain_data_flat,
                           iflag, eps, im_shape[0], im_shape[1], im_shape[2],
                           f, debug=0, spread_debug=0, spread_sort=2, fftw=0, modeord=0,
                           chkbnds=0, upsampfac=1.25)  # upsampling at 1.25 saves time at low precisions
        im_out = f.reshape(im_shape, order='F')
        im_out = abs(im_out / im_out.size)

        if im_shape[0] - original_image_shape[0]:
            im_out = self.crop_volume(im_out, original_image_shape)

        #ov(im_out)

        tensor[0] = torch.from_numpy(im_out)

        return tensor


def estimate_borders_mean_std(array):

    borders_tuple = (
        array[ 0,  :,  :],
        array[-1,  :,  :],
        array[ :,  0,  :],
        array[ :, -1,  :],
        array[ :,  :,  0],
        array[ :,  :, -1],
    )
    borders_flat = np.hstack([border.ravel() for border in borders_tuple])
    borders_means = [border.ravel().mean() for border in borders_tuple]
    borders_std  = [border.ravel().std() for border in borders_tuple]

    #the borders with minimal mean is likely to have only noise ...
    ii = np.argmin(borders_means)

    return borders_means[ii], borders_std[ii]

