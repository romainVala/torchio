import copy
import collections
from typing import Dict, Sequence, Optional, Callable

from deprecated import deprecated
from torch.utils.data import Dataset

from ..utils import get_stem
from ..torchio import DATA, AFFINE, TYPE, PATH, STEM, TypePath, LABEL, INTENSITY
from .image import Image
from .io import write_image
from .subject import Subject
import torch

from utils_file import gfile, get_parent_path

class ImagesDataset(Dataset):
    """Base TorchIO dataset.

    :py:class:`~torchio.data.dataset.ImagesDataset`
    is a reader of 3D medical images that directly
    inherits from :class:`torch.utils.data.Dataset`.
    It can be used with a :class:`torch.utils.data.DataLoader`
    for efficient loading and augmentation.
    It receives a list of subjects, where each subject is an instance of
    :py:class:`torchio.data.subject.Subject` containing instances of
    :py:class:`torchio.data.image.Image`.
    The file format must be compatible with `NiBabel`_ or `SimpleITK`_ readers.
    It can also be a directory containing
    `DICOM`_ files.

    Indexing an :py:class:`~torchio.data.dataset.ImagesDataset` returns an
    instance of :py:class:`~torchio.data.subject.Subject`. Check out the
    documentation for both classes for usage examples.

    Example:

        >>> sample = images_dataset[0]
        >>> sample
        Subject(Keys: ('image', 'label'); images: 2)
        >>> image = sample['image']  # or sample.image
        >>> image.shape
        torch.Size([1, 176, 256, 256])
        >>> image.affine
        array([[   0.03,    1.13,   -0.08,  -88.54],
               [   0.06,    0.08,    0.95, -129.66],
               [   1.18,   -0.06,   -0.11,  -67.15],
               [   0.  ,    0.  ,    0.  ,    1.  ]])

    Args:
        subjects: Sequence of instances of
            :class:`~torchio.data.subject.Subject`.
        transform: An instance of :py:class:`torchio.transforms.Transform`
            that will be applied to each sample.

    Example:
        >>> import torchio
        >>> from torchio import ImagesDataset, Image, Subject
        >>> from torchio.transforms import RescaleIntensity, RandomAffine, Compose
        >>> subject_a = Subject([
        ...     t1=Image('t1.nrrd', type=torchio.INTENSITY),
        ...     t2=Image('t2.mha', type=torchio.INTENSITY),
        ...     label=Image('t1_seg.nii.gz', type=torchio.LABEL),
        ...     age=31,
        ...     name='Fernando Perez',
        >>> ])
        >>> subject_b = Subject(
        ...     t1=Image('colin27_t1_tal_lin.minc', type=torchio.INTENSITY),
        ...     t2=Image('colin27_t2_tal_lin_dicom', type=torchio.INTENSITY),
        ...     label=Image('colin27_seg1.nii.gz', type=torchio.LABEL),
        ...     age=56,
        ...     name='Colin Holmes',
        ... )
        >>> subjects_list = [subject_a, subject_b]
        >>> transforms = [
        ...     RescaleIntensity((0, 1)),
        ...     RandomAffine(),
        ... ]
        >>> transform = Compose(transforms)
        >>> subjects_dataset = ImagesDataset(subjects_list, transform=transform)
        >>> subject_sample = subjects_dataset[0]

    .. _NiBabel: https://nipy.org/nibabel/#nibabel
    .. _SimpleITK: https://itk.org/Wiki/ITK/FAQ#What_3D_file_formats_can_ITK_import_and_export.3F
    .. _DICOM: https://www.dicomstandard.org/
    .. _affine matrix: https://nipy.org/nibabel/coordinate_systems.html
    """

    def __init__(
            self,
            subjects: Sequence[Subject],
            transform: Optional[Callable] = None,
            save_to_dir = None,
            load_from_dir = None,
            add_to_load = None,
            add_to_load_regexp = None,
            ):
        self.load_from_dir = load_from_dir
        self.add_to_load = add_to_load
        self.add_to_load_regexp = add_to_load_regexp
        if not load_from_dir:
            self._parse_subjects_list(subjects)
        self.subjects = subjects
        self._transform: Optional[Callable]
        self.set_transform(transform)
        self.save_to_dir = save_to_dir

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, index: int) -> dict:
        if not isinstance(index, int):
            raise ValueError(f'Index "{index}" must be int, not {type(index)}')

        if self.load_from_dir:
            sample = torch.load(self.subjects[index])
            if self.add_to_load is not None:
                #print('adding sample with {}'.format(self.add_to_load))
                ii = sample.get_images()
                image_path = ii[0]['path']
                if 'original' in self.add_to_load:
                    #print('adding original sample')
                    ss = Subject(image = Image(image_path, INTENSITY))
                    # sss = self._get_sample_dict_from_subject(ss)
                    #sss = copy.deepcopy(ss)
                    sss = ss

                    sample['original'] = sss['image']

                    if self.add_to_load=='original': #trick to use both orig and mask :hmmm....
                        add_to_load = None
                    else:
                        add_to_load = self.add_to_load[8:]
                else:
                    add_to_load = self.add_to_load

                if add_to_load is not None:
                    image_add = gfile(get_parent_path(image_path), self.add_to_load_regexp)[0]
                    #print('adding image {} to {}'.format(image_add,self.add_to_load))
                    ss = Subject(image = Image(image_add, LABEL))
                    #sss = self._get_sample_dict_from_subject(ss)
                    #sss = copy.deepcopy(ss)
                    sss = ss
                    hh = sample.history
                    for hhh in hh:
                        if 'RandomElasticDeformation' in hhh[0]:
                            from torchio.transforms import RandomElasticDeformation
                            num_cp =  hhh[1]['coarse_grid'].shape[1]
                            rr = RandomElasticDeformation(num_control_points=num_cp)
                            sss = rr.apply_given_transform(sss, hhh[1]['coarse_grid'])

                    sample[add_to_load] = sss['image']
            #print('sample with keys {}'.format(sample.keys()))
        else:
            subject = self.subjects[index]
            sample = copy.deepcopy(subject) # cheap since images not loaded yet
            # sample = self._get_sample_dict_from_subject(subject)
            sample.load()

        # Apply transform (this is usually the bottleneck)
        if self._transform is not None:
            sample = self._transform(sample)

        if self.save_to_dir is not None:
            res_dir = self.save_to_dir
            fname = res_dir + '/sample{:05d}'.format(index)
            if 'image_orig' in sample: sample.pop('image_orig')
            torch.save(sample, fname + '_sample.pt')

        return sample

    def set_transform(self, transform: Optional[Callable]) -> None:
        """Set the :attr:`transform` attribute.

        Args:
            transform: An instance of :py:class:`torchio.transforms.Transform`.
        """
        if transform is not None and not callable(transform):
            raise ValueError(
                f'The transform must be a callable object, not {transform}')
        self._transform = transform

    def get_transform(self) :
        return self._transform

    @staticmethod
    def _parse_subjects_list(subjects_list: Sequence[Subject]) -> None:
        # Check that it's list or tuple
        if not isinstance(subjects_list, collections.abc.Sequence):
            raise TypeError(
                f'Subject list must be a sequence, not {type(subjects_list)}')

        # Check that it's not empty
        if not subjects_list:
            raise ValueError('Subjects list is empty')

        # Check each element
        for subject in subjects_list:
            if not isinstance(subject, Subject):
                message = (
                    'Subjects list must contain instances of torchio.Subject,'
                    f' not "{type(subject)}"'
                )
                raise TypeError(message)

    @classmethod
    @deprecated(
        'ImagesDataset.save_sample is deprecated. Use Image.save instead'
    )
    def save_sample(
            cls,
            sample: Subject,
            output_paths_dict: Dict[str, TypePath],
            ) -> None:
        for key, output_path in output_paths_dict.items():
            tensor = sample[key][DATA].squeeze()  # assume 2D if (1, 1, H, W)
            affine = sample[key][AFFINE]
            write_image(tensor, affine, output_path)
