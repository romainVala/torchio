import os
import torch
import warnings
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from typing import Union, Dict, List
from ....data.subject import Subject, Image
from ....data.io import nib_to_sitk, read_image
from ... import SpatialTransform
from .... import DATA, AFFINE


class Coregister(SpatialTransform):
    r"""Apply a coregistration transform to the data
    to a specified target using Elastix.
        Args:
            target: target represents the fixed image to coregister to.
            Can be one of:
                    - string: the transform will look for the corresponding
                      key in the subject.
                    - path: the transform will read the data specified in
                      the path.
                    - instance of :class:`~torchio.Image`:
                    the transform will directly use it as a reference.
            estimation_mapping: mapping of which coregistration parameters
                                to apply to which image. Can be one of:
                                - string:  the estimated coregistration
                                  parameters of the corresponding key
                                  of the subject will be used for all
                                  data.
                                - dict: the transform will use the keys for
                                  the parameter estimation and the list of
                                  values for application. For example,
                                  ``{'t1': ['t1', 't2', 'mask']}`` will apply
                                  the transform estimated on ``'t1'->'target'``
                                  for the three keys ``'t1'``, ``'t2'`` and
                                  ``'mask'``.
                                - ``None``: the coregistrations are
                                  estimated and applied separately
                                  for each data.
            default_parameter_map: parameter map to use in Elastix's
                                   'ElastixImageFilter'_ coregistration.
                                   Either one of ``['translation',
                                   'affine', 'rigid', 'non-rigid']``
                                   or a :class:`SimpleITK.ParameterMap` object.
                                   See `Elastix github`_ for the
                                   ``ParameterMap`` keys.
            kwargs: See :class:`~torchio.transforms.Transform` for
                    additional keyword arguments.
        .. _Elastix github: https://github.com/SuperElastix/elastix/blob/522843d90ff586be051c480514cd14a88db45dbf/src/Core/Main/elxParameterObject.cxx#L260-L362
        .. _ElastixImageFilter: https://simpleelastix.readthedocs.io/
        """  # noqa: E501

    def __init__(
            self,
            target: Union[str, Path, Image],
            estimation_mapping: Union[str, Dict, None] = None,
            default_parameter_map: Union[str, sitk.ParameterMap] = 'rigid',
            **kwargs):
        super().__init__(**kwargs)
        self.target = target
        self.estimation_mapping = estimation_mapping
        self.default_parameter_map = default_parameter_map
        self.estimated_parameter_maps = {}
        if isinstance(self.target, str):
            if self.exclude:
                self.exclude.append(self.target)
            else:
                self.exclude = [self.target]
        self.args_names = (
            'target',
            'estimation_mapping',
            'default_parameter_map',
        )

    def apply_transform(self, subject: Subject) -> Subject:
        # Parse self.target
        if isinstance(self.target, (Path, str)):
            if self.target in subject.keys():

                image_reference = subject[self.target].as_sitk(force_3d=True)
                affine = subject[self.target][AFFINE]

            else:
                try:
                    data, affine = read_image(self.target)
                    image_reference = nib_to_sitk(data=data,
                                                  affine=affine,
                                                  force_3d=True)
                except FileNotFoundError:
                    warnings.warn(
                        f'Coregistration target: {self.target} not found for '
                        f'subject: {subject}.\nSkipping coregistration.')
                    return subject

        elif isinstance(self.target, Image):
            image_reference = self.target.as_sitk(force_3d=True)
            affine = self.target[AFFINE]
        elif isinstance(self.target, (np.ndarray, torch.Tensor)):
            affine = np.eye(4)
            image_reference = nib_to_sitk(self.target,
                                          affine=affine,
                                          force_3d=True)
        elif not isinstance(self.target, Dict):
            warnings.warn(f'Unrecognized target argument: {self.target}.'
                          f'\nSkipping coregistration.')
            return subject

        imgs = subject.get_images_dict(intensity_only=False,
                                       exclude=self.exclude)
        transform_map = {}
        # if estimation_mapping is None :
        estimation_mapping = {img_key: img_key for img_key in imgs.keys()}
        if isinstance(self.estimation_mapping, str):
            if self.estimation_mapping in subject.keys():
                estimation_mapping = {
                    self.estimation_mapping:
                    list(imgs.keys())}
            else:
                warnings.warn(f'Key: {self.estimation_mapping} '
                              f'not found in subject: {subject}.'
                              f'\nResuming coregistration without '
                              f'transform map')

        elif isinstance(self.estimation_mapping, Dict):
            estimation_mapping = {
                img_key: estimated_img_key
                for img_key, estimated_img_key
                in self.estimation_mapping.items()
                if img_key in imgs.keys()
            }

        for estimation_img_name, img_names in estimation_mapping.items():

            estimation_img = imgs[estimation_img_name]

            if estimation_img_name not in transform_map.keys():
                parameter_map = self._estimate_coregisteration(
                    estimation_img,
                    image_reference,
                    default_parameter_map=self.default_parameter_map)
                transform_map[estimation_img_name] = parameter_map
            if isinstance(img_names, str):
                img_name = img_names
                moving_img = imgs[img_name]
                coregistered = self._apply_coregistration(
                    moving_img,
                    transform_map[estimation_img_name]
                )
                moving_img.set_data(coregistered)
                moving_img.affine = affine
            else:
                for img_name in img_names:
                    moving_img = imgs[img_name]
                    coregistered = self._apply_coregistration(
                        moving_img,
                        transform_map[estimation_img_name]
                    )
                    moving_img.set_data(coregistered)
                    moving_img.affine = affine
        self.estimated_parameter_maps = transform_map
        #os.remove('TransformParameters.0.txt')
        return subject

    @staticmethod
    def _estimate_coregisteration(
            img_src: Image,
            img_ref: sitk.SimpleITK.Image,
            default_parameter_map: Union[str, sitk.ParameterMap] = 'rigid'
    ):

        transform_parameter_maps = []

        for data_img in img_src[DATA]:
            data_sitk = nib_to_sitk(data_img[np.newaxis, ...],
                                    affine=img_src.affine,
                                    force_3d=True)
            if isinstance(default_parameter_map, str):
                p_map = sitk.GetDefaultParameterMap(default_parameter_map)
                p_map['WriteResultImage'] = ['false']
            else:
                p_map = default_parameter_map

            elastix_filter = sitk.ElastixImageFilter()
            elastix_filter.SetFixedImage(img_ref)
            elastix_filter.SetMovingImage(data_sitk)
            elastix_filter.SetParameterMap(p_map)
            elastix_filter.LogToConsoleOff()
            elastix_filter.LogToFileOff()
            elastix_filter.Execute()

            transform_parameter_maps.append(
                elastix_filter.GetTransformParameterMap()[0])

        return transform_parameter_maps

    @staticmethod
    def _apply_coregistration(
            img_src: Image,
            transform_parameter_maps: List[sitk.ParameterMap]
    ):

        coregistered_data = []

        #for data_img, trsfm_map in zip(img_src[DATA], transform_parameter_maps):  #Rrr list of different size
        trsfm_map = transform_parameter_maps[0] #rrr tocheck can it be greater than 1 ?
        for data_img in img_src[DATA]:

            data_sitk = nib_to_sitk(data_img[np.newaxis, ...],
                                    affine=img_src.affine,
                                    force_3d=True)
            transformed_img = sitk.Transformix(data_sitk, trsfm_map)
            coregistered_img = torch.as_tensor(
                np.transpose(sitk.GetArrayFromImage(transformed_img)))
            coregistered_data.append(coregistered_img.unsqueeze(0))

        return torch.cat(coregistered_data)
