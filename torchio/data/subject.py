import copy
import pprint
from typing import Any, Dict, List, Tuple, Optional, Sequence

import numpy as np

from ..constants import TYPE, INTENSITY
from .image import Image
from ..utils import get_subclasses


class Subject(dict):
    """Class to store information about the images corresponding to a subject.

    Args:
        *args: If provided, a dictionary of items.
        **kwargs: Items that will be added to the subject sample.

    Example:

        >>> import torchio as tio
        >>> # One way:
        >>> subject = tio.Subject(
        ...     one_image=tio.ScalarImage('path_to_image.nii.gz'),
        ...     a_segmentation=tio.LabelMap('path_to_seg.nii.gz'),
        ...     age=45,
        ...     name='John Doe',
        ...     hospital='Hospital Juan Negrín',
        ... )
        >>> # If you want to create the mapping before, or have spaces in the keys:
        >>> subject_dict = {
        ...     'one image': tio.ScalarImage('path_to_image.nii.gz'),
        ...     'a segmentation': tio.LabelMap('path_to_seg.nii.gz'),
        ...     'age': 45,
        ...     'name': 'John Doe',
        ...     'hospital': 'Hospital Juan Negrín',
        ... }
        >>> subject = tio.Subject(subject_dict)

    """

    def __init__(self, *args, **kwargs: Dict[str, Any]):
        if args:
            if len(args) == 1 and isinstance(args[0], dict):
                kwargs.update(args[0])
            else:
                message = (
                    'Only one dictionary as positional argument is allowed')
                raise ValueError(message)
        super().__init__(**kwargs)
        self._parse_images(self.get_images(intensity_only=False))
        self.update_attributes()  # this allows me to do e.g. subject.t1
        self.applied_transforms = []
        self.transforms_metrics = []

    def __repr__(self):
        num_images = len(self.get_images(intensity_only=False))
        string = (
            f'{self.__class__.__name__}'
            f'(Keys: {tuple(self.keys())}; images: {num_images})'
        )
        return string

    def __copy__(self):
        result_dict = {}
        for key, value in self.items():
            if isinstance(value, Image):
                value = copy.copy(value)
            else:
                value = copy.deepcopy(value)
            result_dict[key] = value
        new = Subject(result_dict)
        new.applied_transforms = self.applied_transforms[:]
        new.transforms_metrics = self.transforms_metrics[:]
        return new

    def __len__(self):
        return len(self.get_images(intensity_only=False))

    @staticmethod
    def _parse_images(images: List[Tuple[str, Image]]) -> None:
        # Check that it's not empty
        if not images:
            raise ValueError('A subject without images cannot be created')

    @property
    def shape(self):
        """Return shape of first image in subject.

        Consistency of shapes across images in the subject is checked first.
        """
        self.check_consistent_attribute('shape')
        return self.get_first_image().shape

    @property
    def spatial_shape(self):
        """Return spatial shape of first image in subject.

        Consistency of spatial shapes across images in the subject is checked
        first.
        """
        self.check_consistent_spatial_shape()
        return self.get_first_image().spatial_shape

    @property
    def spacing(self):
        """Return spacing of first image in subject.

        Consistency of spacings across images in the subject is checked first.
        """
        self.check_consistent_attribute('spacing')
        return self.get_first_image().spacing

    @property
    def history(self):
        from ..transforms.transform import Transform
        transform_classes = {cls.__name__: cls for cls in get_subclasses(Transform)}
        transforms_list = []
        for transform_name, arguments in self.applied_transforms:
            transform = transform_classes[transform_name](**arguments)
            transforms_list.append(transform)
        return transforms_list

    @property
    def _transforms_metrics(self):
        return self.transforms_metrics

    def get_composed_history(self) -> 'Transform':
        from ..transforms.augmentation.composition import Compose
        return Compose(self.history)

    def get_inverse_transform(self, warn=True) -> 'Transform':
        return self.get_composed_history().inverse(warn=warn)

    def apply_inverse_transform(self, warn=True) -> 'Subject':
        transformed = self.get_inverse_transform(warn=warn)(self)
        transformed.clear_history()
        return transformed

    def clear_history(self) -> None:
        self.applied_transforms = []
        self.transforms_metrics = []

    def check_consistent_attribute(self, attribute: str) -> None:
        values_dict = {}
        iterable = self.get_images_dict(intensity_only=False).items()
        for image_name, image in iterable:
            values_dict[image_name] = getattr(image, attribute)
        num_unique_values = len(set(values_dict.values()))
        if num_unique_values > 1:
            message = (
                f'More than one {attribute} found in subject images:'
                f'\n{pprint.pformat(values_dict)}'
            )
            raise RuntimeError(message)

    def check_consistent_spatial_shape(self) -> None:
        self.check_consistent_attribute('spatial_shape')

    def check_consistent_orientation(self) -> None:
        self.check_consistent_attribute('orientation')

    def check_consistent_affine(self):
        # https://github.com/fepegar/torchio/issues/354
        affine = None
        first_image = None
        iterable = self.get_images_dict(intensity_only=False).items()
        for image_name, image in iterable:
            if affine is None:
                affine = image.affine
                first_image = image_name
            elif not np.allclose(affine, image.affine, rtol=1e-6, atol=1e-6):
                message = (
                    f'Images "{first_image}" and "{image_name}" do not occupy'
                    ' the same physical space.'
                    f'\nAffine of "{first_image}":'
                    f'\n{pprint.pformat(affine)}'
                    f'\nAffine of "{image_name}":'
                    f'\n{pprint.pformat(image.affine)}'
                )
                raise RuntimeError(message)

    def check_consistent_space(self):
        self.check_consistent_spatial_shape()
        self.check_consistent_affine()

    def get_images_dict(
            self,
            intensity_only=True,
            include: Optional[Sequence[str]] = None,
            exclude: Optional[Sequence[str]] = None,
            ) -> Dict[str, Image]:
        images = {}
        for image_name, image in self.items():
            if not isinstance(image, Image):
                continue
            if intensity_only and not image[TYPE] == INTENSITY:
                continue
            if include is not None and image_name not in include:
                continue
            if exclude is not None and image_name in exclude:
                continue
            images[image_name] = image
        return images

    def get_images(
            self,
            intensity_only=True,
            include: Optional[Sequence[str]] = None,
            exclude: Optional[Sequence[str]] = None,
            ) -> List[Image]:
        images_dict = self.get_images_dict(intensity_only=intensity_only, include=include, exclude=exclude)
        return list(images_dict.values())

    def get_first_image(self) -> Image:
        return self.get_images(intensity_only=False)[0]

    def add_metrics(self,
                    transform: 'Transform',
                    metrics: dict) -> None:
        from torchio.transforms.augmentation import RandomTransform
        if isinstance(transform, RandomTransform):
            self.transforms_metrics.append((transform.name, metrics))

    # flake8: noqa: F821
    def add_transform(
            self,
            transform: 'Transform',
            parameters_dict: dict,
            ) -> None:
        self.applied_transforms.append((transform.name, parameters_dict))

    def add_transform_all(
            self,
            transform: 'Transform',
            ) -> None:

        parameters_dict = transform.__dict__.copy()
        self.applied_transforms.append((transform.name, parameters_dict))

    def load(self) -> None:
        """Load images in subject."""
        for image in self.get_images(intensity_only=False):
            image.load()

    def update_attributes(self) -> None:
        # This allows to get images using attribute notation, e.g. subject.t1
        self.__dict__.update(self)

    def add_image(self, image: Image, image_name: str) -> None:
        """Add an image."""
        self[image_name] = image
        self.update_attributes()

    def remove_image(self, image_name: str) -> None:
        """Remove an image."""
        del self[image_name]

    def plot(self, **kwargs) -> None:
        """Plot images."""
        from ..visualization import plot_subject  # avoid circular import
        plot_subject(self, **kwargs)
