import copy
import collections
from typing import Dict, Sequence, Optional, Callable

from deprecated import deprecated
from torch.utils.data import Dataset

from ..torchio import DATA, AFFINE, TypePath, LABEL, INTENSITY
from .image import Image
from .io import write_image
from .subject import Subject
import torch

from utils_file import gfile, get_parent_path

class SubjectsDataset(Dataset):
    """Base TorchIO dataset.

    :py:class:`~torchio.data.dataset.SubjectsDataset`
    is a reader of 3D medical images that directly
    inherits from :class:`torch.utils.data.Dataset`.
    It can be used with a :class:`torch.utils.data.DataLoader`
    for efficient loading and augmentation.
    It receives a list of instances of
    :py:class:`torchio.data.subject.Subject`.

    Args:
        subjects: Sequence of instances of
            :class:`~torchio.data.subject.Subject`.
        transform: An instance of :py:class:`torchio.transforms.Transform`
            that will be applied to each subject.

    Example:
        >>> from torchio import SubjectsDataset, ScalarImage, LabelMap, Subject
        >>> from torchio.transforms import RescaleIntensity, RandomAffine, Compose
        >>> subject_a = Subject(
        ...     t1=ScalarImage('t1.nrrd',),
        ...     t2=ScalarImage('t2.mha',),
        ...     label=LabelMap('t1_seg.nii.gz'),
        ...     age=31,
        ...     name='Fernando Perez',
        ... )
        >>> subject_b = Subject(
        ...     t1=ScalarImage('colin27_t1_tal_lin.minc',),
        ...     t2=ScalarImage('colin27_t2_tal_lin_dicom',),
        ...     label=LabelMap('colin27_seg1.nii.gz'),
        ...     age=56,
        ...     name='Colin Holmes',
        ... )
        >>> subjects_list = [subject_a, subject_b]
        >>> transforms = [
        ...     RescaleIntensity((0, 1)),
        ...     RandomAffine(),
        ... ]
        >>> transform = Compose(transforms)
        >>> subjects_dataset = SubjectsDataset(subjects_list, transform=transform)
        >>> subject = subjects_dataset[0]

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

    def __getitem__(self, index: int) -> Subject:
        if not isinstance(index, int):
            raise ValueError(f'Index "{index}" must be int, not {type(index)}')

        if self.load_from_dir:
            subject = torch.load(self.subjects[index])
            if self.add_to_load is not None:
                #print('adding subject with {}'.format(self.add_to_load))
                ii = subject.get_images()
                image_path = ii[0]['path']
                if 'original' in self.add_to_load:
                    #print('adding original subject')
                    ss = Subject(image = Image(image_path, INTENSITY))
                    # sss = self._get_sample_dict_from_subject(ss)
                    #sss = copy.deepcopy(ss)
                    sss = ss

                    subject['original'] = sss['image']

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
                    hh = subject.history
                    for hhh in hh:
                        if 'RandomElasticDeformation' in hhh[0]:
                            from torchio.transforms import RandomElasticDeformation
                            num_cp =  hhh[1]['coarse_grid'].shape[1]
                            rr = RandomElasticDeformation(num_control_points=num_cp)
                            sss = rr.apply_given_transform(sss, hhh[1]['coarse_grid'])

                    subject[add_to_load] = sss['image']
            #print('subject with keys {}'.format(subject.keys()))
        else:
            subject = self.subjects[index]
            subject = copy.deepcopy(subject) # cheap since images not loaded yet
            # subject = self._get_sample_dict_from_subject(subject)
            subject.load()

        # Apply transform (this is usually the bottleneck)
        if self._transform is not None:
            subject = self._transform(subject)

        if self.save_to_dir is not None:
            res_dir = self.save_to_dir
            fname = res_dir + '/subject{:05d}'.format(index)
            if 'image_orig' in subject: subject.pop('image_orig')
            torch.save(subject, fname + '_subject.pt')

        return subject

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
        'SubjectsDataset.save_sample is deprecated. Use Image.save instead'
    )
    def save_sample(
            cls,
            subject: Subject,
            output_paths_dict: Dict[str, TypePath],
            ) -> None:
        for key, output_path in output_paths_dict.items():
            tensor = subject[key][DATA]
            affine = subject[key][AFFINE]
            write_image(tensor, affine, output_path)


@deprecated(
    'ImagesDataset is deprecated in v0.18.0. Use SubjectsDataset instead.')
class ImagesDataset(SubjectsDataset):
    pass
