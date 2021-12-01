import copy
from typing import Sequence, Optional, Callable, Iterable, Dict

from torch.utils.data import Dataset

from .subject import Subject
import torch
from ..utils import get_subjects_from_batch

class SubjectsDataset(Dataset):
    """Base TorchIO dataset.

    Reader of 3D medical images that directly inherits from the PyTorch
    :class:`~torch.utils.data.Dataset`. It can be used with a PyTorch
    :class:`~torch.utils.data.DataLoader` for efficient loading and
    augmentation. It receives a list of instances of :class:`~torchio.Subject`
    and an optional transform applied to the volumes after loading.

    Args:
        subjects: List of instances of :class:`~torchio.Subject`.
        transform: An instance of :class:`~torchio.transforms.Transform`
            that will be applied to each subject.
        load_getitem: Load all subject images before returning it in
            :meth:`__getitem__`. Set it to ``False`` if some of the images will
            not be needed during training.

    Example:
        >>> import torchio as tio
        >>> subject_a = tio.Subject(
        ...     t1=tio.ScalarImage('t1.nrrd',),
        ...     t2=tio.ScalarImage('t2.mha',),
        ...     label=tio.LabelMap('t1_seg.nii.gz'),
        ...     age=31,
        ...     name='Fernando Perez',
        ... )
        >>> subject_b = tio.Subject(
        ...     t1=tio.ScalarImage('colin27_t1_tal_lin.minc',),
        ...     t2=tio.ScalarImage('colin27_t2_tal_lin_dicom',),
        ...     label=tio.LabelMap('colin27_seg1.nii.gz'),
        ...     age=56,
        ...     name='Colin Holmes',
        ... )
        >>> subjects_list = [subject_a, subject_b]
        >>> transforms = [
        ...     tio.RescaleIntensity(out_min_max=(0, 1)),
        ...     tio.RandomAffine(),
        ... ]
        >>> transform = tio.Compose(transforms)
        >>> subjects_dataset = tio.SubjectsDataset(subjects_list, transform=transform)
        >>> subject = subjects_dataset[0]

    .. _NiBabel: https://nipy.org/nibabel/#nibabel
    .. _SimpleITK: https://itk.org/Wiki/ITK/FAQ#What_3D_file_formats_can_ITK_import_and_export.3F
    .. _DICOM: https://www.dicomstandard.org/
    .. _affine matrix: https://nipy.org/nibabel/coordinate_systems.html

    .. tip:: To quickly iterate over the subjects without loading the images,
        use :meth:`dry_iter()`.
    """  # noqa: E501

    def __init__(
            self,
            subjects: Sequence[Subject],
            transform: Optional[Callable] = None,
            save_to_dir = None,
            load_from_dir = None,
            add_to_load = None,
            add_to_load_regexp = None,
            load_getitem: bool = True,
            ):
        self.load_from_dir = load_from_dir
        self.add_to_load = add_to_load
        self.add_to_load_regexp = add_to_load_regexp
        if not load_from_dir:
            self._parse_subjects_list(subjects)
        self._subjects = subjects
        self._transform: Optional[Callable]
        self.set_transform(transform)
        self.save_to_dir = save_to_dir
        self.load_getitem = load_getitem

    def __len__(self):
        return len(self._subjects)

    def __getitem__(self, index: int) -> Subject:
        if not isinstance(index, int):
            raise ValueError(f'Index "{index}" must be int, not {type(index)}')

        if self.load_from_dir:
            subject = torch.load(self._subjects[index])
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
                    from utils_file import gfile, get_parent_path

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
            try:
                index = int(index)
            except (RuntimeError, TypeError):
                message = (
                    f'Index "{index}" must be int or compatible dtype,'
                    f' but an object of type "{type(index)}" was passed'
                )
                raise ValueError(message)

            subject = self._subjects[index]
            subject = copy.deepcopy(subject)  # cheap since images not loaded yet
            if self.load_getitem:
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

    @classmethod
    def from_batch(cls: 'SubjectsDataset', batch: Dict) -> 'SubjectsDataset':
        """Instantiate a dataset from a batch generated by a data loader.

        Args:
            batch: Dictionary generated by a data loader, containing data that
                can be converted to instances of :class:`~.torchio.Subject`.
        """
        subjects = get_subjects_from_batch(batch)
        return cls(subjects)

    def dry_iter(self):
        """Return the internal list of subjects.

        This can be used to iterate over the subjects without loading the data
        and applying any transforms::

        >>> names = [subject.name for subject in dataset.dry_iter()]
        """
        return self._subjects

    def set_transform(self, transform: Optional[Callable]) -> None:
        """Set the :attr:`transform` attribute.

        Args:
            transform: Callable object, typically an subclass of
                :class:`torchio.transforms.Transform`.
        """
        if transform is not None and not callable(transform):
            message = (
                'The transform must be a callable object,'
                f' but it has type {type(transform)}'
            )
            raise ValueError(message)
        self._transform = transform

    def get_transform(self) :
        return self._transform

    @staticmethod
    def _parse_subjects_list(subjects_list: Iterable[Subject]) -> None:
        # Check that it's an iterable
        try:
            iter(subjects_list)
        except TypeError as e:
            message = (
                f'Subject list must be an iterable, not {type(subjects_list)}'
            )
            raise TypeError(message) from e

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
