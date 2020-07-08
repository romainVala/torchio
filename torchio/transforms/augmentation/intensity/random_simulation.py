
from typing import Union, Tuple, Optional, Dict, List
import torch
from ....torchio import DATA, TypeData, TypeRangeFloat, TypeNumber, AFFINE, INTENSITY, TYPE, LABEL
from ....data.subject import Subject
from ....data.image import Image
from .. import RandomTransform

MEAN_RANGE = (0.1, 0.9)
STD_RANGE = (0.01, 0.1)


class RandomSimulation(RandomTransform):
    r"""Generate a random image from label maps.

    Args:
        label_keys: Keys designating the label images in the sample that
            will be used to generate the new image.
            If a single string is given, the corresponding label map is expected
            to be a label encoding map.
            Default will use all images labeled as label in the sample.
        image_key: Key to which the new volume will be saved.
            If this key corresponds to an already existing volume, zero elements from
            the new volume will be filled with elements of the original volume.
        coefficients: Dictionary containing the mean and standard deviation for
            each label. For each value :math:`v`, if a tuple
            :math:`(a, b)` is provided then
            :math:`v \sim \mathcal{U}(a, b)`.
            Default values are :math:`(0.1, 0.9)` for the means and
            :math:`(0.01, 0.1)` for the standard deviations.
        binary: Boolean to tell if label maps should be binarized.
        p: Probability that this transform will be applied.
        seed: See :py:class:`~torchio.transforms.augmentation.RandomTransform`.

    .. note:: The simulated values are in :math: `[0, 1]`. This means that if an
        already existing image is given, it should have been normalized beforehand.

    Example:
        >>> import torchio
        >>> from torchio import RandomSimulation, DATA, RescaleIntensity, Compose
        >>> from torchio.datasets import Colin27
        >>> colin = Colin27(2008)
        >>> # Using the default coefficients
        >>> transform = RandomSimulation(label_keys='cls')
        >>> # Using custom coefficients
        >>> label_values = colin['cls'][DATA].unique()
        >>> coefficients = {
        >>>     i: {
        >>>         'mean': i / len(label_values), 'std': 0.01
        >>>     } for i in range(1, len(label_values))
        >>> }
        >>> transform = RandomSimulation(label_keys='cls', coefficients=coefficients)
        >>> # Inpainting the simulated image on the original T1 image
        >>> rescale_transform = RescaleIntensity((0, 1), (1, 99))   # Rescale intensity before inpainting
        >>> simulation_transform = RandomSimulation(label_keys='cls', image_key='t1')
        >>> transform = Compose([rescale_transform, simulation_transform])
        >>> transformed = transform(colin)  # colin has a new key 'image' with the simulated image
    """
    def __init__(
            self,
            label_keys: Optional[Union[str, List[str], Tuple[str, ...]]] = None,
            coefficients: Optional[Dict[Union[str, TypeNumber], Dict[str, TypeRangeFloat]]] = None,
            image_key: str = 'image',
            binary: bool = True,
            p: float = 1,
            seed: Optional[int] = None,
            **kwargs
            ):
        super().__init__(p=p, seed=seed, **kwargs)
        self.label_keys = label_keys
        self.coefficients = self.parse_coefficients(coefficients)
        self.image_key = image_key
        self.binary = binary

    def apply_transform(self, sample: Subject) -> dict:
        random_parameters_images_dict = {}
        original_image = sample.get(self.image_key)
        final_image = None

        if isinstance(self.label_keys, str):
            image_dict = sample[self.label_keys]
            data = image_dict[DATA]
            labels = data.unique()[1:]
            for i, label in enumerate(labels, 1):
                mean, std = self.get_params(i)
                random_parameters_images_dict[i] = {'mean': mean, 'std': std}
                tissue = self.generate_tissue(data == label, mean, std)
                if final_image is None:
                    final_image = Image(type=INTENSITY, affine=image_dict[AFFINE], tensor=tissue[0])
                else:
                    final_image[DATA] += tissue

        else:
            for image_name, image_dict in sample.get_images_dict(intensity_only=False).items():
                if image_dict[TYPE] == LABEL:
                    if self.label_keys is None or image_name in self.label_keys:
                        mean, std = self.get_params(image_name)
                        random_parameters_images_dict[image_name] = {'mean': mean, 'std': std}
                        if self.binary:
                            data = image_dict[DATA] >= 0.5          # May fail if PV between more than 2 structures
                        else:
                            data = image_dict[DATA]
                        tissue = self.generate_tissue(data, mean, std)
                        if final_image is None:
                            final_image = Image(type=INTENSITY, affine=image_dict[AFFINE], tensor=tissue[0])
                        else:
                            final_image[DATA] += tissue

        if original_image is not None:
            final_image[DATA][final_image[DATA] == 0] = original_image[DATA][final_image[DATA] == 0]

        sample[self.image_key] = final_image
        sample.add_transform(self, random_parameters_images_dict)
        return sample

    def parse_coefficients(self, coefficients):
        parsed_coefficients = {'mean': MEAN_RANGE, 'std': STD_RANGE}

        if coefficients is None:
            coefficients = {}

        for label_key, dictionary in coefficients.items():
            if list(dictionary.keys()) != ['mean', 'std']:
                raise KeyError(f'Given coefficients {dictionary.keys()} do not match {["mean", "std"]}')
            mean = self.parse_coefficient(dictionary['mean'], 'mean')
            std = self.parse_coefficient(dictionary['std'], 'std')
            parsed_coefficients.update({label_key: {'mean': mean, 'std': std}})

        return parsed_coefficients

    @staticmethod
    def parse_coefficient(
            nums_range: TypeRangeFloat,
            name: str,
            ) -> Tuple[float, float]:
        if isinstance(nums_range, (int, float)):
            return nums_range, nums_range

        if len(nums_range) != 2:
            raise ValueError(
                f'If {name} is a sequence,'
                f' it must be of len 2, not {nums_range}')
        min_value, max_value = nums_range
        if min_value > max_value:
            raise ValueError(
                f'If {name} is a sequence, the second value must be'
                f' equal or greater than the first, not {nums_range}')
        return min_value, max_value

    def get_params(
        self,
        label: Union[str, TypeNumber]
    ) -> Tuple[float, float]:
        if label in self.coefficients:
            mean_range, std_range = self.coefficients[label]['mean'], self.coefficients[label]['std']
        else:
            mean_range, std_range = self.coefficients['mean'], self.coefficients['std']

        mean = torch.FloatTensor(1).uniform_(*mean_range).item()
        std = torch.FloatTensor(1).uniform_(*std_range).item()
        return mean, std

    @staticmethod
    def generate_tissue(
            data: TypeData,
            mean: TypeNumber,
            std: TypeNumber,
            ) -> TypeData:
        # Create the simulated tissue using a gaussian random variable
        data_shape = data.shape
        gaussian = torch.randn(data_shape) * std + mean
        return gaussian.clamp(1e-06, 1) * data
