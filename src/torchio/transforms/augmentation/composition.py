from __future__ import annotations

import warnings
from typing import Dict
from typing import Sequence
from typing import Union

import numpy as np
import torch

from . import RandomTransform
from .. import Transform
from ...data.subject import Subject
from typing import List


TypeTransformsDict = Union[Dict[Transform, float], Sequence[Transform]]


class Compose(Transform):
    """Compose several transforms together.

    Args:
        transforms: Sequence of instances of
            :class:`~torchio.transforms.Transform`.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.
    """

    def __init__(self, transforms: Sequence[Transform], **kwargs):
        super().__init__(parse_input=False, **kwargs)
        for transform in transforms:
            if not callable(transform):
                message = (
                    'One or more of the objects passed to the Compose'
                    f' transform are not callable: "{transform}"'
                )
                raise TypeError(message)
        self.transforms = list(transforms)

    def __len__(self):
        return len(self.transforms)

    def __getitem__(self, index) -> Transform:
        return self.transforms[index]

    def __repr__(self) -> str:
        return f'{self.name}({self.transforms})'

    def apply_transform(self, subject: Subject) -> Subject:
        for transform in self.transforms:
            subject = transform(subject)  # type: ignore[assignment]
        return subject

    def is_invertible(self) -> bool:
        return all(t.is_invertible() for t in self.transforms)

    def inverse(self, warn: bool = True) -> Compose:
        """Return a composed transform with inverted order and transforms.

        Args:
            warn: Issue a warning if some transforms are not invertible.
        """
        transforms = []
        for transform in self.transforms:
            if transform.is_invertible():
                transforms.append(transform.inverse())
            elif warn:
                message = f'Skipping {transform.name} as it is not invertible'
                warnings.warn(message, RuntimeWarning, stacklevel=2)
        transforms.reverse()
        result = Compose(transforms)
        if not transforms and warn:
            warnings.warn(
                'No invertible transforms found',
                RuntimeWarning,
                stacklevel=2,
            )
        return result


class OneOf(RandomTransform):
    """Apply only one of the given transforms.

    Args:
        transforms: Dictionary with instances of
            :class:`~torchio.transforms.Transform` as keys and
            probabilities as values. Probabilities are normalized so they sum
            to one. If a sequence is given, the same probability will be
            assigned to each transform.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    Example:
        >>> import torchio as tio
        >>> colin = tio.datasets.Colin27()
        >>> transforms_dict = {
        ...     tio.RandomAffine(): 0.75,
        ...     tio.RandomElasticDeformation(): 0.25,
        ... }  # Using 3 and 1 as probabilities would have the same effect
        >>> transform = tio.OneOf(transforms_dict)
        >>> transformed = transform(colin)
    """

    def __init__(self, transforms: TypeTransformsDict, **kwargs):
        super().__init__(parse_input=False, **kwargs)
        self.transforms_dict = self._get_transforms_dict(transforms)

    def apply_transform(self, subject: Subject) -> Subject:
        weights = torch.Tensor(list(self.transforms_dict.values()))
        index = torch.multinomial(weights, 1)
        transforms = list(self.transforms_dict.keys())
        transform = transforms[index]
        transformed = transform(subject)
        return transformed  # type: ignore[return-value]

    def _get_transforms_dict(
        self,
        transforms: TypeTransformsDict,
    ) -> Dict[Transform, float]:
        if isinstance(transforms, dict):
            transforms_dict = dict(transforms)
            self._normalize_probabilities(transforms_dict)
        else:
            try:
                p = 1 / len(transforms)
            except TypeError as e:
                message = (
                    'Transforms argument must be a dictionary or a sequence,'
                    f' not {type(transforms)}'
                )
                raise ValueError(message) from e
            transforms_dict = {transform: p for transform in transforms}
        for transform in transforms_dict:
            if not isinstance(transform, Transform):
                message = (
                    'All keys in transform_dict must be instances of'
                    f'torchio.Transform, not "{type(transform)}"'
                )
                raise ValueError(message)
        return transforms_dict

    @staticmethod
    def _normalize_probabilities(
        transforms_dict: Dict[Transform, float],
    ) -> None:
        probabilities = np.array(list(transforms_dict.values()), dtype=float)
        if np.any(probabilities < 0):
            message = (
                f'Probabilities must be greater or equal to zero, not "{probabilities}"'
            )
            raise ValueError(message)
        if np.all(probabilities == 0):
            message = (
                'At least one probability must be greater than zero,'
                f' but they are "{probabilities}"'
            )
            raise ValueError(message)
        for transform, probability in transforms_dict.items():
            transforms_dict[transform] = probability / probabilities.sum()


class ListOf(RandomTransform):
    """Apply sequencly all of the given transforms.

    Args:
        transforms: A list  of
            :py:class:`~torchio.transforms.transform.Transform`
        p : probabilities

    Example:
        >>> import torchio
        >>> ixi = torchio.datasets.ixi.IXITiny('ixi', download=True)
        >>> sample = ixi[0]
        >>> transforms_dict = []
        ...     torchio.transforms.RandomAffine(),
        ...     torchio.transforms.RandomElasticDeformation(),
        ... ]
        >>> transform = torchio.transforms.OneOf(transforms_dict)

    """
    def __init__(
            self,
            transforms: Sequence[Transform],
            p: float = 1,
            **kwargs
            ):
        super().__init__(p=p, **kwargs)
        self.transforms_list = transforms

    def apply_transform(self, sample: Subject):

        transformed_list=[]
        for tt in self.transforms_list:
            transformed_list.append(tt(sample))

        return transformed_list


def compose_from_history(history: List):
    """Builds a list of transformations and seeds to reproduce a given subject's transformations from its history

    Args:
        history: subject history given as a list of tuples containing (transformation_name, transformation_parameters)
    Returns:
        Tuple (List of transforms, list of seeds to reproduce the transforms from the history)
    """
    trsfm_list = []
    seed_list = []
    for trsfm_name, trsfm_params in history:
        # No need to add the RandomDownsample since its Resampling operation is taken into account in the history
        if trsfm_name == 'RandomDownsample':
            continue
        # Add the seed if there is one (if the transform is random)
        if 'seed' in trsfm_params.keys():
            seed_list.append(trsfm_params['seed'])
        else:
            seed_list.append(None)
        # Gather all available attributes from the transformations' history
        # Ugly fix for RandomSwap's patch_size...
        trsfm_no_seed = {key: json.loads(value) if type(value) == str and value.startswith('[') else value
                         for key, value in trsfm_params.items() if key != 'seed'}
        # Special case for the interpolation as it is stored as a string in the history, a conversion is needed
        if 'interpolation' in trsfm_no_seed.keys():
            trsfm_no_seed['interpolation'] = getattr(Interpolation, trsfm_no_seed['interpolation'].split('.')[1])
        # Special cases when an argument is needed in the __init__
        if trsfm_name == 'RandomLabelsToImage':
            trsfm_func = getattr(torchio, trsfm_name)(label_key=trsfm_no_seed['label_key'])

        elif trsfm_name == 'Resample':
            if 'target' in trsfm_no_seed.keys():
                trsfm_func = getattr(torchio, trsfm_name)(target=trsfm_no_seed['target'])
            elif 'target_spacing' in trsfm_no_seed.keys():
                trsfm_func = getattr(torchio, trsfm_name)(target=trsfm_no_seed['target_spacing'])

        else:
            trsfm_func = getattr(torchio, trsfm_name)()
        trsfm_func.__dict__ = trsfm_no_seed
        trsfm_list.append(trsfm_func)
    return trsfm_list, seed_list
