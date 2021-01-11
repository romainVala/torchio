import copy
import numbers
import warnings
from typing import Union, Tuple
from abc import ABC, abstractmethod
from contextlib import contextmanager

import torch
import numpy as np
import SimpleITK as sitk

from ..utils import to_tuple
from ..data.subject import Subject
from ..data.io import nib_to_sitk, sitk_to_nib
from ..data.image import LabelMap
from ..typing import (
    TypeKeys,
    TypeData,
    TypeNumber,
    TypeCallable,
    TypeTripletInt,
)
from .interpolation import Interpolation, get_sitk_interpolator
from .data_parser import DataParser, TypeTransformInput
from typing import Dict

TypeSixBounds = Tuple[int, int, int, int, int, int]
TypeBounds = Union[
    int,
    TypeTripletInt,
    TypeSixBounds,
]
TypeMaskingMethod = Union[str, TypeCallable, TypeBounds, None]
anat_axes = 'Left', 'Right', 'Anterior', 'Posterior', 'Inferior', 'Superior'


class Transform(ABC):
    """Abstract class for all TorchIO transforms.

    All subclasses must overwrite
    :meth:`Transform.apply_transform`,
    which takes data, applies some transformation and returns the result.

    The input can be an instance of
    :class:`torchio.Subject`,
    :class:`torchio.Image`,
    :class:`numpy.ndarray`,
    :class:`torch.Tensor`,
    :class:`SimpleITK.Image`,
    or :class:`dict`.

    Args:
        p: Probability that this transform will be applied.
        copy: Make a shallow copy of the input before applying the transform.
        include: Sequence of strings with the names of the only images to which
            the transform will be applied.
            Mandatory if the input is a :class:`dict`.
        exclude: Sequence of strings with the names of the images to which the
            the transform will not be applied, apart from the ones that are
            excluded because of the transform type.
            For example, if a subject includes an MRI, a CT and a label map,
            and the CT is added to the list of exclusions of an intensity
            transform such as :class:`~torchio.transforms.RandomBlur`,
            the transform will be only applied to the MRI, as the label map is
            excluded by default by spatial transforms.
    """
    def __init__(
            self,
            p: float = 1,
            copy: bool = True,
            include: TypeKeys = None,
            exclude: TypeKeys = None,
            keys: TypeKeys = None,
            metrics: Dict = None,
            keep_before = None
            ):
        self.probability = self.parse_probability(p)
        self.copy = copy
        self.keys = keys
        self.default_image_name = 'default_image_name'
        self.metrics = metrics
        self.keep_before = keep_before

        if keys is not None:
            message = (
                'The "keys" argument is deprecated and will be removed in the'
                ' future. Use "include" instead.'
            )
            warnings.warn(message, DeprecationWarning)
            include = keys
        self.include, self.exclude = self.parse_include_and_exclude(
            include, exclude)

    def __call__(
            self,
            data: TypeTransformInput,
            ) -> TypeTransformInput:
        """Transform data and return a result of the same type.

        Args:
            data: Instance of 1) :class:`~torchio.Subject`, 4D
                :class:`torch.Tensor` or :class:`numpy.ndarray` with dimensions
                :math:`(C, W, H, D)`, where :math:`C` is the number of channels
                and :math:`W, H, D` are the spatial dimensions. If the input is
                a tensor, the affine matrix will be set to identity. Other
                valid input types are a SimpleITK image, a
                :class:`torchio.Image`, a NiBabel Nifti1 image or a
                :class:`dict`. The output type is the same as the input type.
        """
        if torch.rand(1).item() > self.probability:
            return data
        if isinstance(data, list):
            return [self.__call__(ii) for ii in data]

        if self.metrics:
            self._metrics = dict()

        data_parser = DataParser(data, keys=self.include)
        subject = data_parser.get_subject()
        orig = subject #todo marche aussi si self.copy is false ?
        if self.copy:
            subject = copy.copy(subject)

        if self.keep_before:
            from ..data.image import LabelMap
            affine = subject[self.keep_before]['affine']
            new_key = 'before' + self.name + '_' + self.keep_before
            new_image = LabelMap(affine=affine, tensor=orig[self.keep_before]['data'])

            #self.exclude = [new_key] if self.exclude is None else self.exclude.append(new_key)

        with np.errstate(all='warn'):
            transformed = self.apply_transform(subject)

        if self.keep_before: #also add diff
            new_key2 = 'diff' + self.name + '_' + self.keep_before
            new_image2 = LabelMap(affine=affine, tensor=orig[self.keep_before]['data'] -  transformed[self.keep_before]['data'])
            transformed.add_image(new_image, new_key)
            transformed.add_image(new_image2, new_key2)

        if isinstance(transformed, list):
            return transformed

        if self.metrics:
            _metrics = [metric_func(orig, transformed) for metric_func in self.metrics.values()]

            for dict_metrics in _metrics:
                for sample_key, metric_vals in dict_metrics.items():
                    if sample_key not in self._metrics.keys():
                        self._metrics[sample_key] = dict()
                    for metric_name, metric_val in metric_vals.items():
                        self._metrics[sample_key][metric_name] = metric_val

        if hasattr(self, "_metrics"):
            transformed.add_metrics(self, self._metrics)

        self.add_transform_to_subject_history(transformed)

        for image in transformed.get_images(intensity_only=False):
            ndim = image.data.ndim
            assert ndim == 4, f'Output of {self.name} is {ndim}D'

        output = data_parser.get_output(transformed)
        return output

    def __repr__(self):
        if hasattr(self, 'args_names'):
            names = self.args_names
            args_strings = [f'{arg}={getattr(self, arg)}' for arg in names]
            if hasattr(self, 'invert_transform') and self.invert_transform:
                args_strings.append('invert=True')
            args_string = ', '.join(args_strings)
            return f'{self.name}({args_string})'
        else:
            return super().__repr__()

    @property
    def name(self):
        return self.__class__.__name__

    @abstractmethod
    def apply_transform(self, subject: Subject):
        raise NotImplementedError

    def add_transform_to_subject_history(self, subject):
        from .augmentation import RandomTransform
        from . import Compose, OneOf, CropOrPad, EnsureShapeMultiple
        from .preprocessing.label import SequentialLabels
        call_others = (
            RandomTransform,
            Compose,
            OneOf,
            CropOrPad,
            EnsureShapeMultiple,
            SequentialLabels,
        )
        if not isinstance(self, call_others):
            subject.add_transform(self, self._get_reproducing_arguments())

    @staticmethod
    def to_range(n, around):
        if around is None:
            return 0, n
        else:
            return around - n, around + n

    def parse_params(self, params, around, name, make_ranges=True, **kwargs):
        params = to_tuple(params)
        # d or (a, b)
        if len(params) == 1 or (len(params) == 2 and make_ranges):
            params *= 3  # (d, d, d) or (a, b, a, b, a, b)
        if len(params) == 3 and make_ranges:  # (a, b, c)
            items = [self.to_range(n, around) for n in params]
            # (-a, a, -b, b, -c, c) or (1-a, 1+a, 1-b, 1+b, 1-c, 1+c)
            params = [n for prange in items for n in prange]
        if make_ranges:
            if len(params) != 6:
                message = (
                    f'If "{name}" is a sequence, it must have length 2, 3 or'
                    f' 6, not {len(params)}'
                )
                raise ValueError(message)
            for param_range in zip(params[::2], params[1::2]):
                self._parse_range(param_range, name, **kwargs)
        return tuple(params)

    @staticmethod
    def _parse_range(
            nums_range: Union[TypeNumber, Tuple[TypeNumber, TypeNumber]],
            name: str,
            min_constraint: TypeNumber = None,
            max_constraint: TypeNumber = None,
            type_constraint: type = None,
            ) -> Tuple[TypeNumber, TypeNumber]:
        r"""Adapted from :class:`torchvision.transforms.RandomRotation`.

        Args:
            nums_range: Tuple of two numbers :math:`(n_{min}, n_{max})`,
                where :math:`n_{min} \leq n_{max}`.
                If a single positive number :math:`n` is provided,
                :math:`n_{min} = -n` and :math:`n_{max} = n`.
            name: Name of the parameter, so that an informative error message
                can be printed.
            min_constraint: Minimal value that :math:`n_{min}` can take,
                default is None, i.e. there is no minimal value.
            max_constraint: Maximal value that :math:`n_{max}` can take,
                default is None, i.e. there is no maximal value.
            type_constraint: Precise type that :math:`n_{max}` and
                :math:`n_{min}` must take.

        Returns:
            A tuple of two numbers :math:`(n_{min}, n_{max})`.

        Raises:
            ValueError: if :attr:`nums_range` is negative
            ValueError: if :math:`n_{max}` or :math:`n_{min}` is not a number
            ValueError: if :math:`n_{max} \lt n_{min}`
            ValueError: if :attr:`min_constraint` is not None and
                :math:`n_{min}` is smaller than :attr:`min_constraint`
            ValueError: if :attr:`max_constraint` is not None and
                :math:`n_{max}` is greater than :attr:`max_constraint`
            ValueError: if :attr:`type_constraint` is not None and
                :math:`n_{max}` and :math:`n_{max}` are not of type
                :attr:`type_constraint`.
        """
        if isinstance(nums_range, numbers.Number):  # single number given
            if nums_range < 0:
                raise ValueError(
                    f'If {name} is a single number,'
                    f' it must be positive, not {nums_range}')
            if min_constraint is not None and nums_range < min_constraint:
                raise ValueError(
                    f'If {name} is a single number, it must be greater'
                    f' than {min_constraint}, not {nums_range}'
                )
            if max_constraint is not None and nums_range > max_constraint:
                raise ValueError(
                    f'If {name} is a single number, it must be smaller'
                    f' than {max_constraint}, not {nums_range}'
                )
            if type_constraint is not None:
                if not isinstance(nums_range, type_constraint):
                    raise ValueError(
                        f'If {name} is a single number, it must be of'
                        f' type {type_constraint}, not {nums_range}'
                    )
            min_range = -nums_range if min_constraint is None else nums_range
            return (min_range, nums_range)

        try:
            min_value, max_value = nums_range
        except (TypeError, ValueError):
            raise ValueError(
                f'If {name} is not a single number, it must be'
                f' a sequence of len 2, not {nums_range}'
            )

        min_is_number = isinstance(min_value, numbers.Number)
        max_is_number = isinstance(max_value, numbers.Number)
        if not min_is_number or not max_is_number:
            message = (
                f'{name} values must be numbers, not {nums_range}')
            raise ValueError(message)

        if min_value > max_value:
            raise ValueError(
                f'If {name} is a sequence, the second value must be'
                f' equal or greater than the first, but it is {nums_range}')

        if min_constraint is not None and min_value < min_constraint:
            raise ValueError(
                f'If {name} is a sequence, the first value must be greater'
                f' than {min_constraint}, but it is {min_value}'
            )

        if max_constraint is not None and max_value > max_constraint:
            raise ValueError(
                f'If {name} is a sequence, the second value must be smaller'
                f' than {max_constraint}, but it is {max_value}'
            )

        if type_constraint is not None:
            min_type_ok = isinstance(min_value, type_constraint)
            max_type_ok = isinstance(max_value, type_constraint)
            if not min_type_ok or not max_type_ok:
                raise ValueError(
                    f'If "{name}" is a sequence, its values must be of'
                    f' type "{type_constraint}", not "{type(nums_range)}"'
                )
        return nums_range

    @staticmethod
    def parse_interpolation(interpolation: str) -> str:
        if not isinstance(interpolation, str):
            itype = type(interpolation)
            raise TypeError(f'Interpolation must be a string, not {itype}')
        interpolation = interpolation.lower()
        is_string = isinstance(interpolation, str)
        supported_values = [key.name.lower() for key in Interpolation]
        is_supported = interpolation.lower() in supported_values
        if is_string and is_supported:
            return interpolation
        message = (
            f'Interpolation "{interpolation}" of type {type(interpolation)}'
            f' must be a string among the supported values: {supported_values}'
        )
        raise ValueError(message)

    @staticmethod
    def parse_probability(probability: float) -> float:
        is_number = isinstance(probability, numbers.Number)
        if not (is_number and 0 <= probability <= 1):
            message = (
                'Probability must be a number in [0, 1],'
                f' not {probability}'
            )
            raise ValueError(message)
        return probability

    @staticmethod
    def parse_include_and_exclude(
            include: TypeKeys = None,
            exclude: TypeKeys = None,
            ) -> Tuple[TypeKeys, TypeKeys]:
        if include is not None and exclude is not None:
            raise ValueError('Include and exclude cannot both be specified')
        return include, exclude

    @staticmethod
    def nib_to_sitk(data: TypeData, affine: TypeData) -> sitk.Image:
        return nib_to_sitk(data, affine)

    @staticmethod
    def sitk_to_nib(image: sitk.Image) -> Tuple[torch.Tensor, np.ndarray]:
        return sitk_to_nib(image)

    def _get_reproducing_arguments(self):
        """
        Return a dictionary with the arguments that would be necessary to
        reproduce the transform exactly.
        """
        reproducing_arguments = {
            'include': self.include,
            'exclude': self.exclude,
            'copy': self.copy,
        }
        args_names = {name: getattr(self, name) for name in self.args_names}
        reproducing_arguments.update(args_names)
        return reproducing_arguments

    def is_invertible(self):
        return hasattr(self, 'invert_transform')

    def inverse(self):
        if not self.is_invertible():
            raise RuntimeError(f'{self.name} is not invertible')
        new = copy.deepcopy(self)
        new.invert_transform = not self.invert_transform
        return new

    @staticmethod
    @contextmanager
    def _use_seed(seed):
        """Perform an operation using a specific seed for the PyTorch RNG"""
        torch_rng_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        yield
        torch.random.set_rng_state(torch_rng_state)

    @staticmethod
    def _fft_im(image):
        output = (np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(image)))).astype(np.complex128)
        return output

    @staticmethod
    def _ifft_im(freq_domain):
        output = np.fft.ifftshift(np.fft.ifftn(freq_domain))
        return output

    @staticmethod
    def _oversample(data, perc_oversampling=.10, padding_mode='constant', padding_normal=None):
        """
        Oversamples data with a zero padding. Adds perc_oversampling percentage values
        :param data (ndarray): array to pad
        :param perc_oversampling (float): percentage of oversampling to add to data (based on its current shape)
        :return oversampled version of the data:
        """
        data_shape = list(data.shape)
        to_pad = np.ceil(np.asarray(data_shape) * perc_oversampling / 2) * 2
        # to force an even number if odd, this will shift the volume when croping
        # print("Pading at {}".format(to_pad))
        left_pad = np.floor(to_pad / 2).astype(int)
        right_pad = np.ceil(to_pad / 2).astype(int)

        if padding_mode == "random.normal":
            pad_data = np.pad(data, list(zip(left_pad, right_pad)))
            data_shape = list(pad_data.shape)

            # replace the padding values by random nois
            size_pad = left_pad[0] * data_shape[1] * data_shape[2]
            pad_data[:left_pad[0], :, :] = np.random.normal(padding_normal[0], padding_normal[1],
                                                            size_pad).reshape(left_pad[0], data_shape[1], data_shape[2])

            size_pad = left_pad[1] * data_shape[0] * data_shape[2]
            pad_data[:, :left_pad[1], :] = np.random.normal(padding_normal[0], padding_normal[1],
                                                            size_pad).reshape(data_shape[0], left_pad[1], data_shape[2])

            size_pad = left_pad[2] * data_shape[1] * data_shape[0]
            pad_data[:, :, :left_pad[2]] = np.random.normal(padding_normal[0], padding_normal[1],
                                                            size_pad).reshape(data_shape[0], data_shape[1], left_pad[2])

            size_pad = right_pad[0] * data_shape[1] * data_shape[2]
            pad_data[-right_pad[0]:, :, :] = np.random.normal(padding_normal[0], padding_normal[1],
                                                              size_pad).reshape(right_pad[0], data_shape[1],
                                                                                data_shape[2])

            size_pad = right_pad[1] * data_shape[0] * data_shape[2]
            pad_data[:, -right_pad[1]:, :] = np.random.normal(padding_normal[0], padding_normal[1],
                                                              size_pad).reshape(data_shape[0], right_pad[1],
                                                                                data_shape[2])

            size_pad = right_pad[2] * data_shape[1] * data_shape[0]
            pad_data[:, :, -right_pad[0]:] = np.random.normal(padding_normal[0], padding_normal[1],
                                                              size_pad).reshape(data_shape[0], data_shape[1],
                                                                                right_pad[2])
            # print('PADING with random nois {} {}'.format(padding_normal[0], padding_normal[1]))
        else:
            pad_data = np.pad(data, list(zip(left_pad, right_pad)), mode=padding_mode)

        return pad_data

    @staticmethod
    def crop_volume(data, cropping_shape):
        '''
        Cropping data to cropping_shape size. Cropping starts from center of the image
        '''
        vol_centers = (np.asarray(data.shape) / 2).astype(int)
        dim_ranges = np.ceil(np.asarray(cropping_shape) / 2).astype(int)
        slicing = [slice(dim_center - dim_range, dim_center + dim_range)
                   for dim_center, dim_range in zip(vol_centers, dim_ranges)]
        return data[tuple(slicing)]

    @property
    def name(self):
        return self.__class__.__name__

    @staticmethod
    def get_sitk_interpolator(interpolation: str) -> int:
        return get_sitk_interpolator(interpolation)

    @staticmethod
    def parse_bounds(bounds_parameters: TypeBounds) -> TypeSixBounds:
        try:
            bounds_parameters = tuple(bounds_parameters)
        except TypeError:
            bounds_parameters = (bounds_parameters,)

        # Check that numbers are integers
        for number in bounds_parameters:
            if not isinstance(number, (int, np.integer)) or number < 0:
                message = (
                    'Bounds values must be integers greater or equal to zero,'
                    f' not "{bounds_parameters}" of type {type(number)}'
                )
                raise ValueError(message)
        bounds_parameters = tuple(int(n) for n in bounds_parameters)
        bounds_parameters_length = len(bounds_parameters)
        if bounds_parameters_length == 6:
            return bounds_parameters
        if bounds_parameters_length == 1:
            return 6 * bounds_parameters
        if bounds_parameters_length == 3:
            return tuple(np.repeat(bounds_parameters, 2).tolist())
        message = (
            'Bounds parameter must be an integer or a tuple of'
            f' 3 or 6 integers, not {bounds_parameters}'
        )
        raise ValueError(message)

    @staticmethod
    def ones(tensor: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(tensor, dtype=torch.bool)

    @staticmethod
    def mean(tensor: torch.Tensor) -> torch.Tensor:
        mask = tensor > tensor.float().mean()
        return mask

    @staticmethod
    def get_mask_from_masking_method(
            masking_method: TypeMaskingMethod,
            subject: Subject,
            tensor: torch.Tensor,
            ) -> torch.Tensor:
        if masking_method is None:
            return Transform.ones(tensor)
        elif callable(masking_method):
            return masking_method(tensor)
        elif type(masking_method) is str:
            in_subject = masking_method in subject
            if in_subject and isinstance(subject[masking_method], LabelMap):
                return subject[masking_method].data.bool()
            masking_method = masking_method.capitalize()
            if masking_method in anat_axes:
                return Transform.get_mask_from_anatomical_label(
                    masking_method, tensor)
        elif type(masking_method) in (tuple, list, int):
            return Transform.get_mask_from_bounds(masking_method, tensor)
        message = (
            'Masking method parameter must be a function, a label map name,'
            f' an anatomical label: {anat_axes}, or a bounds parameter'
            ' (an int, tuple of 3 ints, or tuple of 6 ints),'
            f' not {masking_method} of type {type(masking_method)}'
        )
        raise ValueError(message)

    @staticmethod
    def get_mask_from_anatomical_label(
            anatomical_label: str,
            tensor: torch.Tensor,
            ) -> torch.Tensor:
        anatomical_label = anatomical_label.title()
        if anatomical_label.title() not in anat_axes:
            message = (
                f'Anatomical label must be one of {anat_axes}'
                f' not {anatomical_label}'
            )
            raise ValueError(message)
        mask = torch.zeros_like(tensor, dtype=torch.bool)
        _, width, height, depth = tensor.shape
        if anatomical_label == 'Right':
            mask[:, width // 2:] = True
        elif anatomical_label == 'Left':
            mask[:, :width // 2] = True
        elif anatomical_label == 'Anterior':
            mask[:, :, height // 2:] = True
        elif anatomical_label == 'Posterior':
            mask[:, :, :height // 2] = True
        elif anatomical_label == 'Superior':
            mask[:, :, :, depth // 2:] = True
        elif anatomical_label == 'Inferior':
            mask[:, :, :, :depth // 2] = True
        return mask

    @staticmethod
    def get_mask_from_bounds(
            bounds_parameters: TypeBounds,
            tensor: torch.Tensor,
            ) -> torch.Tensor:
        bounds_parameters = Transform.parse_bounds(bounds_parameters)
        low = bounds_parameters[::2]
        high = bounds_parameters[1::2]
        i0, j0, k0 = low
        i1, j1, k1 = np.array(tensor.shape[1:]) - high
        mask = torch.zeros_like(tensor, dtype=torch.bool)
        mask[:, i0:i1, j0:j1, k0:k1] = True
        return mask
