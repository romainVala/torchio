from __future__ import annotations

from typing import Tuple

import torch

from .. import Transform
from ...typing import TypeRangeFloat
from ...transforms.data_parser import TypeTransformInput


class RandomTransform(Transform):
    """Base class for stochastic augmentation transforms.

    Args:
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.
    """
    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)

    def __call__(
        self,
        data: TypeTransformInput,
        seed: int = None,
    ) -> TypeTransformInput:
        """Transform data and return a result of the same type.

        Args:
            data: Instance of :py:class:`~torchio.Subject`, 4D
                :py:class:`torch.Tensor` or 4D NumPy array with dimensions
                :math:`(C, W, H, D)`, where :math:`C` is the number of channels
                and :math:`W, H, D` are the spatial dimensions. If the input is
                a tensor, the affine matrix is an identity and a tensor will be
                also returned.
            seed: Seed for :py:mod:`torch` random number generator.
        """
        if not seed:
            seed = self._get_random_seed()

        # Store the current rng_state to reset it after the execution
        torch_rng_state = torch.random.get_rng_state()
        torch.manual_seed(seed=seed)
        self.seed = seed

        transformed = super().__call__(data=data)

        torch.random.set_rng_state(torch_rng_state)
        return transformed

    def add_include_exclude(self, kwargs):
        kwargs['include'] = self.include
        kwargs['exclude'] = self.exclude
        return kwargs

    def parse_degrees(
            self,
            degrees: TypeRangeFloat,
    ) -> Tuple[float, float]:
        return self._parse_range(degrees, 'degrees')

    def parse_translation(
            self,
            translation: TypeRangeFloat,
    ) -> Tuple[float, float]:
        return self._parse_range(translation, 'translation')

    @staticmethod
    def sample_uniform(a: float, b: float) -> float:
        return torch.FloatTensor(1).uniform_(a, b).item()

    @staticmethod
    def _get_random_seed() -> int:
        """Generate a random seed.

        Returns:
            A random seed as an int.
        """
        return int(torch.randint(0, 2**31, (1,)).item())

    def sample_uniform_sextet(self, params):
        results = []
        for (a, b) in zip(params[::2], params[1::2]):
            results.append(self.sample_uniform(a, b))
        return torch.Tensor(results)
