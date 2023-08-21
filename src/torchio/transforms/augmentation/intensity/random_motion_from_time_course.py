import math
import torch
import warnings
import numpy as np
from collections import defaultdict
from typing import Dict, Tuple, List, Union, Optional
from scipy.interpolate import pchip_interpolate
try:
    import finufft
    _finufft = True
except ImportError:
    _finufft = False
import SimpleITK as sitk

from .. import RandomTransform
from ... import IntensityTransform, FourierTransform
from ....data.io import nib_to_sitk
from ....data.subject import Subject
from ....metrics.fitpar_metrics import compute_motion_metrics #todo remove from PR
from ...preprocessing.spatial.crop import Crop
from ...preprocessing.spatial.pad import Pad


class RandomMotionFromTimeCourse(RandomTransform, IntensityTransform):
    r"""Simulate motion artefact

    Simulate motion artefact from a random time course of object positions.

    parameters relatated to simulate 3 types of displacement random noise swllow or sudden mouvement

    Args:
        nT (int): number of points of the simulated time course (not very important, as it will be interpolated afer
            to the select phase axis dimension
        maxDisp (float, float): (min, max) value of displacement in the perlin noise (useless if noiseBasePars is 0)
        maxRot (float, float): (min, max) value of rotation in the perlin noise (useless if noiseBasePars is 0)
        noiseBasePars (float, float): (min, max) base value of the perlin noise to generate for the time course
            optional (float, float, float) where the third is the probability to perform this type of noise
        swallowFrequency (int, int): (min, max) number of swallowing movements to generate in the time course
            optional (float, float, float) where the third is the probability to perform this type of noise
        swallowMagnitude (float, float): (min, max) magnitude of the swallowing movements to generate
        suddenFrequency (int, int): (min, max) number of sudden movements to generate in the time course
            optional (float, float, float) where the third is the probability to perform this type of noise
        suddenMagnitude (float, float): (min, max) magnitude of the sudden movements to generate
            if euler_motion_params is not None previous parameter are not used
        maxGlobalDisp (float, float): (min, max) of the global translations. A random number is taken from
            this interval to scale each translations if they are bigger. If None, it won't be used. Note that the
            previous maxDisp maxRot swallowMagnitude suddenMagnitude are controling the amplitude of the perturbation
            but we then add all of then so that at the end we do not control the global amplitude. In order to find
            the global amplitude (and scaled it), we take the max of all 2 by 2 positions distance
        maxGlobalRot same as  maxGlobalDisp but for Rotations
        preserve_center_frequency_pct : percentage of the total time to be set to zero for simulated euler motion
            parameters. (Usefull to avoid big global displacement)
        euler_motion_params : movement parameters to use (if specified, no movement is simulated). Note that in this
            case this is no more a random transform, and one could use directly MotionFromTimeCourse.
            if you modify this attribute after the class instance has been created, you need to also set the
            attribute simulate_displacement to false
        displacement_shift (string): How to correct for Global displacement, (ie which choice of reference position)
            possible choice are None (default)/ "center_zero" / "1D_wFT" / "1D_wFT2"
            "center_zero": kspace center is the reference. "1D_wFT": reference is a weighted mean of position,
            where the weights are the fft coeficient (summed over the kspace plane). "1D_wFT2" same as 1D_wFT
            but the weights are squared (very similar to center_zero)
        coregistration_to_orig (bool): Default false. If true it will use Elastix, to co-register the corrupted
            motion volume to the original one. This is the prefer solution to avoid any global displacement
            as none of the displacement_shift strategies defined above will be always correct.
            (but this comes with an extra computational cost)
        phase_encoding_choice (tuple of ints): potential phase encoding dims (slowest) to use
            (randomly chosen from the input tuple)
        oversampling_pct (float): percentage with which the data will be oversampled in the image domain prior
            to applying the motion
        nufft_type (string): "1D_type2" (default) or "1D_type1" for testing purpose, should not be changed

    Note for suddenFrequency and swallowFrequency min max must differ and the max is never achieved,
        so to have 0 put (0,1)

    todo define parameter as degrees or translation in RandomMotion (specify chosen random distribution)
    todo short description of time course simu + reference gallichan retromoco

    .. note:: quite long execution times compare to other transforms
    .. warning:: extra dependency on finufft, if rotation is modeled, and SimpleElastix if coregistration
        is chosen

    """

    def __init__(
            self,
            nT: int = 200,
            maxDisp: Tuple[float, float] = (2,5),
            maxRot: Tuple[float, float] = (2,5),
            noiseBasePars: Tuple[float, float] = (5,15),
            swallowFrequency: Tuple[float, float] = (0,5),
            swallowMagnitude: Tuple[float, float] = (2,6),
            suddenFrequency: Tuple[int, int] = (0,5),
            suddenMagnitude: Tuple[float, float] = (2,6),
            maxGlobalDisp: Tuple[float, float] = (1,2),
            maxGlobalRot: Tuple[float, float] = (1,2),
            preserve_center_frequency_pct: float = 0,
            euler_motion_params: Union[List, np.ndarray, str] = None,
            displacement_shift_strategy: str = None,
            coregistration_to_orig=False,
            phase_encoding_choice: List = [1],
            oversampling_pct: float = 0.3,
            nufft_type: str ='1D_type2',
            **kwargs
            ):
        super().__init__(**kwargs)
        self.nT = nT
        self.maxDisp = maxDisp
        self.maxRot = maxRot
        self.noiseBasePars = noiseBasePars
        self.swallowFrequency = swallowFrequency
        self.swallowMagnitude = swallowMagnitude
        self.suddenFrequency = suddenFrequency
        self.suddenMagnitude = suddenMagnitude
        self.maxGlobalDisp = maxGlobalDisp
        self.maxGlobalRot = maxGlobalRot
        self.displacement_shift_strategy = displacement_shift_strategy
        self.preserve_center_frequency_pct = preserve_center_frequency_pct
        self.kspace_order = self._rand_kspace_order(phase_encoding_choice)
        self.nufft_type = nufft_type
        self.coregistration_to_orig = coregistration_to_orig
        if euler_motion_params is None:
            self.euler_motion_params = None
            self.simulate_displacement = True
        else:
            self.euler_motion_params = read_euler_motion_params(euler_motion_params)
            self.simulate_displacement = False
        self.oversampling_pct = oversampling_pct
        self.to_subtract = None


    def apply_transform(self, subject: Subject) -> Subject:

        arguments = defaultdict(dict)
        if self.simulate_displacement:
            self._simulate_random_trajectory()
        for image_name, image_dict in self.get_images_dict(subject).items():
            arguments["euler_motion_params"][image_name] = self.euler_motion_params
            arguments["displacement_shift_strategy"][image_name] = self.displacement_shift_strategy
            arguments["kspace_order"][image_name] = self.kspace_order
            arguments["oversampling_pct"][image_name] = self.oversampling_pct
            arguments["nufft_type"][image_name] = self.nufft_type
            arguments["coregistration_to_orig"][image_name] = self.coregistration_to_orig

        transform = MotionFromTimeCourse(**self.add_include_exclude(arguments))
        transformed = transform(subject)
        self._metrics = transform._metrics
        return transformed

    def random_params(self,
                      maxDisp: Tuple[float, float] = (2,5),
                      maxRot: Tuple[float, float] = (2,5),
                      noiseBasePars: Tuple[float, float] = (5,15),
                      swallowFrequency: Tuple[float, float] = (0,5),
                      swallowMagnitude: Tuple[float, float] = (2,6),
                      suddenFrequency: Tuple[int, int] = (0,5),
                      suddenMagnitude: Tuple[float, float] = (2,6),
                      maxGlobalDisp: Tuple[float, float] = None,
                      maxGlobalRot: Tuple[float, float] = None
                      ):
        maxDisp_r = self._rand_uniform(min=maxDisp[0], max=maxDisp[1])
        maxRot_r = self._rand_uniform(min=maxRot[0], max=maxRot[1])
        noiseBasePars_r = self._rand_uniform(min=noiseBasePars[0], max=noiseBasePars[1])
        swallowMagnitude_r = [self._rand_uniform(min=swallowMagnitude[0], max=swallowMagnitude[1]),
                            self._rand_uniform(min=swallowMagnitude[0], max=swallowMagnitude[1])]

        suddenMagnitude_r = [self._rand_uniform(min=suddenMagnitude[0], max=suddenMagnitude[1]),
                           self._rand_uniform(min=suddenMagnitude[0], max=suddenMagnitude[1])]

        swallowFrequency_r = torch.randint(swallowFrequency[0], swallowFrequency[1], (1,)).item()
        suddenFrequency_r = torch.randint(suddenFrequency[0], suddenFrequency[1], (1,)).item()

        maxGlobalDisp_r = self._rand_uniform(min=maxGlobalDisp[0],
                                           max=maxGlobalDisp[1]) if maxGlobalDisp else float('inf')
        maxGlobalRot_r = self._rand_uniform(min=maxGlobalRot[0],
                                          max=maxGlobalRot[1]) if maxGlobalRot else float('inf')

        # prba to include the different type of noise
        proba_noiseBase = noiseBasePars[2] if len(noiseBasePars) == 3 else 1
        proba_swallow = swallowFrequency[2] if len(swallowFrequency) == 3 else 1
        proba_sudden = suddenFrequency[2] if len(suddenFrequency) == 3 else 1

        do_noise, do_swallow, do_sudden = False, False, False

        while (do_noise or do_swallow or do_sudden) is False:  # at least one is not false
            do_noise = self._rand_uniform() <= proba_noiseBase
            do_swallow = self._rand_uniform() <= proba_swallow
            do_sudden = self._rand_uniform() <= proba_sudden
        if do_noise is False: noiseBasePars = 0
        if do_swallow is False: swallowFrequency = 0
        if do_sudden is False: suddenFrequency = 0

        return (maxDisp_r, maxRot_r, noiseBasePars_r, swallowMagnitude_r, suddenMagnitude_r, swallowFrequency_r,
                suddenFrequency_r, maxGlobalDisp_r, maxGlobalRot_r)


    def perlinNoise1D(self, npts, weights):
        if not isinstance(weights, list):
            weights = range(int(round(weights)))
            weights = np.power([2] * len(weights), weights)

        n = len(weights)
        xvals = np.linspace(0, 1, npts)
        total = np.zeros((npts, 1))

        for i in range(n):
            frequency = 2 ** i
            this_npts = round(npts / frequency)

            if this_npts > 1:
                total += weights[i] * pchip_interpolate(np.linspace(0, 1, this_npts),
                                                        self._rand_uniform(shape=this_npts)[..., np.newaxis],
                                                        xvals)
        #            else:
        # TODO does it matter print("Maxed out at octave {}".format(i))

        total = total - np.min(total)
        total = total / np.max(total)
        return total.reshape(-1)

    def _simulate_random_trajectory(self):
        """
        Simulates the parameters of the transformation through the vector euler_motion_params using 6 dimensions
        (3 translations and 3 rotations).
        """

        maxDisp, maxRot, noiseBasePars, swallowMagnitude, suddenMagnitude, swallowFrequency, suddenFrequency, \
        maxGlobalDisp, maxGlobalRot = self.random_params(maxDisp=self.maxDisp, maxRot=self.maxRot,
                 noiseBasePars=self.noiseBasePars, swallowFrequency=self.swallowFrequency,
                 swallowMagnitude=self.swallowMagnitude, suddenFrequency=self.suddenFrequency,
                 suddenMagnitude=self.suddenMagnitude, maxGlobalDisp=self.maxGlobalDisp,
                 maxGlobalRot=self.maxGlobalRot)

        if noiseBasePars > 0:
            euler_motion_params = np.asarray([self.perlinNoise1D(self.nT, noiseBasePars) - 0.5 for _ in range(6)])
            euler_motion_params[:3] *= maxDisp
            euler_motion_params[3:] *= maxRot
        else:
            euler_motion_params = np.zeros((6, self.nT))
        # add in swallowing-like movements - just to z direction and pitch
        if swallowFrequency > 0:
            swallowTraceBase = np.exp(-np.linspace(0, 100, self.nT))
            swallowTrace = np.zeros(self.nT)

            for i in range(swallowFrequency):
                rand_shifts = int(round(self._rand_uniform() * self.nT))
                rolled = np.roll(swallowTraceBase, rand_shifts, axis=0)
                swallowTrace += rolled

            euler_motion_params[2, :] += swallowMagnitude[0] * swallowTrace
            euler_motion_params[3, :] += swallowMagnitude[1] * swallowTrace

        # add in random sudden movements in any direction
        if suddenFrequency > 0:
            suddenTrace = np.zeros(euler_motion_params.shape)

            for i in range(suddenFrequency):
                iT_sudden = int(np.ceil(self._rand_uniform() * self.nT))
                to_add = np.asarray([suddenMagnitude[0] * (2 * self._rand_uniform(shape=3) - 1),
                                     suddenMagnitude[1] * (2 * self._rand_uniform(shape=3) - 1)]).reshape((-1, 1))
                suddenTrace[:, iT_sudden:] = np.add(suddenTrace[:, iT_sudden:], to_add)

            euler_motion_params += suddenTrace

        if self.preserve_center_frequency_pct:
            center = np.int32(np.floor(euler_motion_params.shape[1] / 2))
            nbpts = np.int32(np.floor(euler_motion_params.shape[1] * self.preserve_center_frequency_pct / 2))
            euler_motion_params[:, center - nbpts:center + nbpts] = 0

        # rescale to global max if asked
        # max is compute for all trans (or rot) diff
        if self.maxGlobalRot is not None :
            trans_diff = euler_motion_params.T[:,None,:3] - euler_motion_params.T[None,:,:3]  #numpy broadcating rule !
            ddtrans = np.linalg.norm(trans_diff, axis=2)
            ddrot = np.linalg.norm(euler_motion_params.T[:,None,3:] - euler_motion_params.T[None,:,3:] , axis=-1)

            euler_motion_params[:3, :] = euler_motion_params[:3, :] * maxGlobalDisp / ddtrans.max()
            euler_motion_params[3:, :] = euler_motion_params[3:, :] * maxGlobalRot / ddrot.max()

        self.euler_motion_params = euler_motion_params


    def read_euler_motion_params(euler_motion_params):
        '''
        :param euler_motion_params:
        '''
        fpars = None
        if isinstance(euler_motion_params, np.ndarray):
            fpars = euler_motion_params
        elif isinstance(euler_motion_params, list):
            fpars = np.asarray(euler_motion_params)
        if fpars.shape[0] != 6:
            warnings.warn("Given motion parameters has {} on the first dimension. "
                          "Expected 6 (3 translations and 3 rotations). Setting motion to None".format(fpars.shape[0]))
            fpars = None
        elif len(fpars.shape) != 2:
            warnings.warn("Expected motion parameters to be of shape (6, N), found {}. Setting motion to None".format(
                fpars.shape))
            fpars = None

        if np.any(np.isnan(fpars)):
            # assume it is the last column, as can happen if the the csv line ends with ,
            fpars = fpars[:, :-1]
            if np.any(np.isnan(fpars)):
                warnings.warn('There is still NaN in the euler_motion_params, it will crash the nufft')


    def _rand_uniform(self, min=0.0, max=1.0, shape=1):
        rand = torch.FloatTensor(shape).uniform_(min, max)
        if shape == 1:
            return rand.item()
        return rand.numpy()


    def _rand_kspace_order(self, phase_encoding_choise):
        if not isinstance(phase_encoding_choise, list):
            phase_encoding_choise = [phase_encoding_choise]
        #choose phase axis (among user defined choices, ie list of possibilitie)
        random_phase_axis = self._rand_choice(phase_encoding_choise)
        #choose random order
        kspace_order_axis = torch.randperm(3)
        # put the phase at the end
        result = torch.cat([kspace_order_axis[kspace_order_axis!=random_phase_axis],torch.tensor([random_phase_axis])])

        return result.numpy()

    def _rand_choice(self, array):
        chosen_idx = torch.randint(0, len(array), (1, ))
        return array[chosen_idx]


class MotionFromTimeCourse(IntensityTransform, FourierTransform):
    r"""Add MRI motion artifact (computed in kspace).

        euler_motion_params : movement parameters to use (
        displacement_shift (string): How to correct for Global displacement, (ie which choice of reference position)
            possible choice are None (default)/ "center_zero" / "1D_wFT" / "1D_wFT2"
            "center_zero": kspace center is the reference. "1D_wFT": reference is a weighted mean of position,
            where the weights are the fft coeficient (summed over the kspace plane). "1D_wFT2" same as 1D_wFT
            but the weights are squared (very similar to center_zero)
        coregistration_to_orig (bool): Default false. If true it will use Elastix, to co-register the corrupted
            motion volume to the original one. This is the prefer solution to avoid any global displacement
            as none of the displacement_shift strategies defined above will be always correct.
            (but this comes with an extra computational cost)
        kspace order (3 tuple) dimension ordering of the kspace axis, the last one will then define the slowest varying
            axis, and this is where the motion takes place (2 other dimension are assumed, step wise stationary)
        phase_encoding_choice (tuple of ints): potential phase encoding dims (slowest) to use
            (randomly chosen from the input tuple)
        oversampling_pct (float): percentage with which the data will be oversampled in the image domain prior
            to applying the motion
        nufft_type (string): "1D_type2" (default) or "1D_type1" for testing purpose, should not be changed

        """

    def __init__(
            self,
            euler_motion_params: Union[List, np.ndarray, str],
            displacement_shift_strategy: str = None,
            coregistration_to_orig: bool = False,
            kspace_order: int = (0,2,1),
            oversampling_pct: float = 0,
            nufft_type: str = '1D_type1',
            **kwargs
            ):
        super().__init__(**kwargs)
        self.displacement_shift_strategy = displacement_shift_strategy
        self.kspace_order = kspace_order
        self.euler_motion_params = euler_motion_params
        self.oversampling_pct = oversampling_pct
        self.to_subtract = None
        self.nufft_type = nufft_type
        self.coregistration_to_orig = coregistration_to_orig
        self.args_names = ("euler_motion_params", "displacement_shift_strategy", "kspace_order",
                           "oversampling_pct", "nufft_type", "coregistration_to_orig")

    def apply_transform(self, subject: Subject) -> Subject:
        euler_motion_params = self.euler_motion_params
        displacement_shift_strategy = self.displacement_shift_strategy
        kspace_order = tuple(self.kspace_order)
        oversampling_pct = self.oversampling_pct
        nufft_type = self.nufft_type
        coregistration_to_orig = self.coregistration_to_orig

        for image_name, image_dict in self.get_images_dict(subject).items():
            if self.arguments_are_dict():
                euler_motion_params = self.euler_motion_params[image_name]
                displacement_shift_strategy = self.displacement_shift_strategy[image_name]
                kspace_order = tuple(self.kspace_order[image_name])
                oversampling_pct = self.oversampling_pct[image_name]
                nufft_type = self.nufft_type[image_name]
                coregistration_to_orig = self.coregistration_to_orig[image_name]

            original_image = np.squeeze(image_dict['data'])

            if oversampling_pct > 0.0:
                original_image, voxel_to_pad = oversample_volume_array(original_image, oversampling_pct)

            phase_encoding_shape = original_image.shape[kspace_order[-1]]
            #TODO test if oversampling_pct we should add zero in euler_motion_params and not interp to oversampled shape
            if euler_motion_params.shape[1] != phase_encoding_shape:
                euler_motion_params = self._interpolate_euler_motion_params(euler_motion_params,
                                                                            len_output=phase_encoding_shape)

            if displacement_shift_strategy is not None:
                euler_motion_params, self.to_subtract = self.demean_euler_motion_params(euler_motion_params,
                                                                 self.fourier_transform_for_nufft(original_image),
                                                                 displacement_shift_strategy,
                                                                 fast_dim = kspace_order[:2])
                # important to save to get the corrected euler_motion_params in history
                if self.arguments_are_dict():
                    self.euler_motion_params[image_name] = euler_motion_params
                else:
                    self.euler_motion_params = euler_motion_params
            corrupted_im = self.apply_motion_in_kspace(original_image, euler_motion_params, kspace_order, nufft_type)

            # magnitude
            corrupted_im = abs(corrupted_im)
            if coregistration_to_orig:
                corrupted_im = self.ElastixRegisterAndReslice(corrupted_im, original_image)

            if oversampling_pct > 0.0:
                corrupted_im = crop_volume_array(corrupted_im, voxel_to_pad)

            image_dict["data"] = corrupted_im[np.newaxis, ...]
            image_dict['data'] = torch.from_numpy(image_dict['data']).float()

        # todo remove from PR
        # second time we compute the fourier transform (in case we ask for demean too) ...
        self._metrics = compute_motion_metrics(euler_motion_params, self.fourier_transform_for_nufft(original_image),
                                               fast_dim=np.array(kspace_order)[:2])

        return subject


    def ElastixRegisterAndReslice(self, img_src, img_ref):
        img1 = img_src #.data
        img2 = img_ref #.data
        #I think affine does not matter here ... to check
        affine = np.eye(4)
        i1, i2 = nib_to_sitk(np.expand_dims(img1,0), affine), nib_to_sitk(np.expand_dims(img2,0), affine)
        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetFixedImage(i2);
        elastixImageFilter.SetMovingImage(i1)
        elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("rigid"))
        elastixImageFilter.LogToConsoleOff()
        elastixImageFilter.Execute()

        reslice_img = np.transpose(sitk.GetArrayFromImage(elastixImageFilter.GetResultImage()))

        return reslice_img



    def _interpolate_euler_motion_params(
            self,
            euler_motion_params,
            tr_fpars = None,
            tr_to_interpolate = 2.4,
            len_output = 250
            ):

        fpars_length = euler_motion_params.shape[1]
        if tr_fpars is None:
            xp = np.linspace(0,1,fpars_length)
            x  = np.linspace(0,1,len_output)
        else:
            xp = np.asarray(range(fpars_length))*tr_fpars
            x = np.asarray(range(len_output))*tr_to_interpolate
        interpolated_fpars = np.asarray([np.interp(x, xp, fp) for fp in euler_motion_params])
        if xp[-1]<x[-1]:
            diff = x[-1] - xp[-1]
            npt_added = diff/tr_to_interpolate
            print(f'adding {npt_added:.1f}')
        return interpolated_fpars

    def scale_image(self,image,  scale_factor):
        def fourier_transform_for_nufft(image):
            output = (np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(image)))).astype(np.complex128)
            return output
        i = tio.ScalarImage('sub-CC00284BN13_ses-90801_desc-drawem9_dseg.nii.gz')
        image = i.data[0]

        image_shape=image.shape

        lin_spaces = [np.linspace(-0.5, 0.5-1/x, x)*2*math.pi for x in image_shape]  # todo it suposes 1 vox = 1mm
        #remove 1/x to avoid small scaling
        lin_spaces = [np.linspace(-0.5+ 1 / x, 0.5 , x) * 2 * math.pi for x in  image_shape]  # todo it suposes 1 vox = 1mm
        meshgrids = np.meshgrid(*lin_spaces, indexing='ij')
        # pour une affine on ajoute de 1, dans les coordone du point, mais pour le augmented kspace
        # on ajoute la phase initial, donc 0 ici
        meshgrids.append(np.zeros(image_shape))
        grid_out = np.array([mg * scale_factor for mg in meshgrids])

        eps = 1E-7
        #grid_out = grid_out * scale_factor

        f = np.zeros(grid_out[0].shape, dtype=np.complex128, order='F').flatten() #(order='F')
        ip = np.asfortranarray(image.numpy().astype(complex) )

        finufft.nufft3d2(grid_out[0].flatten(order='F'), grid_out[1].flatten(order='F'),
                         grid_out[2].flatten(order='F'), ip,
                         eps=eps, out=f, debug=0, spread_debug=0, spread_sort=2, fftw=0, modeord=0,
                         chkbnds=0, upsampfac=1.25, isign=-1)  # upsampling at 1.25 saves time at low precisions

        #f = f * np.exp(-1j * grid_out[3].flatten(order='F'))
        f = f.reshape(ip.shape,order='F')

        iout = abs( inv_fourier_transform_for_nufft(f) )
        return iout


    def _rotate_coordinates_1D_motion(self, euler_motion_params, image_shape, kspace_order, Apply_inv_affine=True):
        # Apply_inv_affinne is True for the nufft_type1 and false
        # for the nuft2 we also add a 1 voxel translation to euler_motion_params todo check with different resolution
        euler_motion_params_local = euler_motion_params.copy()
        if Apply_inv_affine is False: #case for nufft_type2 add a one voxel shift
            # one voxel shift if odd ! todo resolution ?
            off_center = np.array([(x / 2 - x // 2) * 2 for x in image_shape])
            euler_motion_params_local[:3, :] = euler_motion_params_local[:3, :] - \
                                         np.repeat(np.expand_dims(off_center, 1), euler_motion_params_local.shape[1], axis=1)

        lin_spaces = [np.linspace(-0.5, 0.5-1/x, x)*2*math.pi for x in image_shape]  # todo it suposes 1 vox = 1mm
        #remove 1/x to avoid small scaling

        meshgrids = np.meshgrid(*lin_spaces, indexing='ij')
        # pour une affine on ajoute de 1, dans les coordone du point, mais pour le augmented kspace
        # on ajoute la phase initial, donc 0 ici
        meshgrids.append(np.zeros(image_shape))

        grid_out = np.array([mg for mg in meshgrids])
        dim_order = np.hstack([np.array(0), np.array(kspace_order)+1]) #first dim is 4 (homogeneous coordinates)
        grid_out = grid_out.transpose(dim_order) #so that wanted phase dim is at the end

        if euler_motion_params_local.shape[1] != grid_out.shape[3]:
            raise('dimenssion issue euler_motion_params and phase dim')
        #applied motion at each phase step (on the kspace grid plan)

        for nnp in range(euler_motion_params_local.shape[1]):
            aff = get_matrix_from_euler_and_trans(euler_motion_params_local[:,nnp])
            if Apply_inv_affine:
                aff = np.linalg.inv(aff)
            grid_plane = grid_out[:, :, :, nnp]
            shape_mem = grid_plane.shape
            grid_plane_moved = np.matmul(aff.T, grid_plane.reshape(4,shape_mem[1]*shape_mem[2])) #equ15 A.T * k0
            #grid_plane_moved = np.matmul( grid_plane.reshape(4,shape_mem[1]*shape_mem[2]).T, aff.T).T # r0.T * A.T
            grid_out[:, :, :, nnp] = grid_plane_moved.reshape(shape_mem)

        #tricky to insverse a given transpose
        dim_order_back = np.array([np.argwhere(dim_order==k)[0][0] for k in range(4)])
        grid_out = grid_out.transpose(dim_order_back) #go back to original order

        return grid_out


    def _trans_and_nufft_type1(self, image, euler_motion_params, kspace_order):
        warnings.warn('nufft_type1 is implemented for testing purpose, '
                'you should prefer the nufft_type2 as it is more correct')
        freq_domain = self.fourier_transform_for_nufft(image)

        eps = 1E-7
        f = np.zeros(freq_domain.shape, dtype=np.complex128, order='F')

        grid_out = self._rotate_coordinates_1D_motion(euler_motion_params, freq_domain.shape,
                                                      kspace_order, Apply_inv_affine=True)

        phase_shift = grid_out[3].flatten(order='F')
        exp_phase_shift = np.exp( 1j * phase_shift)  #+1j -> x z == tio, y inverse

        # same F order as phase_shift if not inversion x z
        freq_domain_data_flat = freq_domain.flatten(order='F')* exp_phase_shift

        finufft.nufft3d1(grid_out[0].flatten(order='F'), grid_out[1].flatten(order='F'),
                         grid_out[2].flatten(order='F'), freq_domain_data_flat,
                         eps=eps, out=f, debug=0, spread_debug=0, spread_sort=2, fftw=0, modeord=0,
                         chkbnds=0, upsampfac=1.25, isign= 1)  # upsampling at 1.25 saves time at low precisions
        #im_out = f.reshape(image.shape, order='F')
        #im_out = f.flatten().reshape(image.shape)
        im_out = f / f.size

        return im_out


    def _trans_and_nufft_type2(self, image, euler_motion_params, kspace_order ):
        eps = 1E-7
        grid_out = self._rotate_coordinates_1D_motion(euler_motion_params, image.shape,
                                                      kspace_order, Apply_inv_affine=False)

        f = np.zeros(grid_out[0].shape, dtype=np.complex128, order='F').flatten() #(order='F')
        ip = np.asfortranarray(image.numpy().astype(complex) )

        finufft.nufft3d2(grid_out[0].flatten(order='F'), grid_out[1].flatten(order='F'),
                         grid_out[2].flatten(order='F'), ip,
                         eps=eps, out=f, debug=0, spread_debug=0, spread_sort=2, fftw=0, modeord=0,
                         chkbnds=0, upsampfac=1.25, isign=-1)  # upsampling at 1.25 saves time at low precisions

        f = f * np.exp(-1j * grid_out[3].flatten(order='F'))
        f = f.reshape(ip.shape,order='F')

        iout = abs( self.inv_fourier_transform_for_nufft(f) )
        return iout

    def apply_motion_in_kspace(self, image, euler_motion_params, kspace_order, nufft_type):
        if not _finufft:
            raise ImportError('finufft not available')

        if 'type1' in nufft_type:
            # fft
            corrupted_im = self._trans_and_nufft_type1(image, euler_motion_params, kspace_order)
        else:  # nufft_type2
            corrupted_im = self._trans_and_nufft_type2(image, euler_motion_params, kspace_order)

        return corrupted_im

    def demean_euler_motion_params(self, euler_motion_params, original_image_fft, displacement_shift_strategy,
                       fast_dim=(0,2)):
        #compute a weighted average of motion time course, (separatly on each Euler parameters)
        #return a new time course, shifted .
        # we assume motion only in the slowest phase dimension (1D motion)

        if displacement_shift_strategy == "center_zero":
            center = np.int32(np.floor(euler_motion_params.shape[1] / 2))
            to_subtract = euler_motion_params[:, center]
            to_subtract_tile = np.tile(to_subtract[..., np.newaxis], (1, euler_motion_params.shape[1]))
            euler_motion_params = euler_motion_params - to_subtract_tile
            return euler_motion_params, to_subtract

        #original_image_fft = self.fourier_transform_for_nufft(original_image) #fucking bug, coef_shaw was computed on image

        # coef_shaw = np.sqrt( np.sum(abs(original_image_fft**2), axis=(0,2)) ) ;
        # should be equivalent if fft is done from real image, but not if the phase is acquired,
        # CF Todd 2015 "Prospective motion correction of 3D echo-planar imaging data for functional MRI
        # using optical tracking"
        if displacement_shift_strategy == '1D_wFT':
            coef_shaw = np.abs( np.sqrt(np.sum( original_image_fft * np.conjugate(original_image_fft),
                                                axis=fast_dim )));
        elif displacement_shift_strategy == '1D_wFT2':
            #coef_shaw = np.abs( np.sum( original_image_fft * np.conjugate(original_image_fft), axis=fast_dim ));
            coef_shaw = np.abs( np.sum( original_image_fft **2, axis=fast_dim ))
        else:
            warnings.warn(f'No motion time course demean defined, for parameter displacement_shift_strategy '
                          f'{displacement_shift_strategy}')

        coef_shaw = coef_shaw / np.sum(coef_shaw)

        to_subtract = np.zeros(6)
        for i in range(0,6):
            to_subtract[i] = np.sum(euler_motion_params[i,:] * coef_shaw) / np.sum(coef_shaw)
            euler_motion_params[i,:] = euler_motion_params[i,:] - to_subtract[i]

        return euler_motion_params, to_subtract


def oversample_volume_array(volume, oversampling_pct):
    data_shape = list(volume.shape)
    voxel_to_pad = (np.ceil(np.asarray(data_shape) * oversampling_pct / 2) * 2).astype(int)
    tpad = Pad(voxel_to_pad)
    if isinstance(volume, torch.Tensor):
        volume = volume.unsqueeze(0)
    else : #numpy
        volume = np.expand_dims(volume,0)
    vol_pad = tpad(volume)
    return vol_pad.squeeze(), voxel_to_pad

def crop_volume_array(volume, voxel_to_crop):
    if isinstance(volume, torch.Tensor):
        volume = volume.unsqueeze(0)
    else : #numpy
        volume = np.expand_dims(volume,0)
    tcrop = Crop(voxel_to_crop)
    vol_crop = tcrop(volume)
    return vol_crop.squeeze()


def get_matrix_from_euler_and_trans(P, rot_order='yxz', rotation_center=None):
    # default choosen as the same default as simpleitk
    rot = np.deg2rad(P[3:6])
    aff = np.eye(4)
    aff[:3,3] = P[:3]  #translation
    if rot_order=='xyz':
        aff[:3,:3]  = euler2mat(rot[0], rot[1], rot[2], axes='sxyz')
    elif rot_order=='yxz':
        aff[:3,:3] = euler2mat(rot[1], rot[0], rot[2], axes='syxz') #strange simpleitk convention of euler ... ?
    else:
        raise(f'rotation order {rot_order} not implemented')

    if rotation_center is not None:
        aff = change_affine_rotation_center(aff, rotation_center)
    return aff


# to avoid, an extra dependency, athough this one is nice work from mathew Brett!
# I copy paste the function from transforms3d.euler import euler2mat
# map axes strings to/from tuples of inner axis, parity, repetition, frame

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

def euler2mat(ai, aj, ak, axes='sxyz'):
    """Return rotation matrix from Euler angles and axis sequence.

    Parameters
    ----------
    ai : float
        First rotation angle (according to `axes`).
    aj : float
        Second rotation angle (according to `axes`).
    ak : float
        Third rotation angle (according to `axes`).
    axes : str, optional
        Axis specification; one of 24 axis sequences as string or encoded
        tuple - e.g. ``sxyz`` (the default).

    Returns
    -------
    mat : array (3, 3)
        Rotation matrix or affine.

    Examples
    --------
    >>> R = euler2mat(1, 2, 3, 'syxz')
    >>> np.allclose(np.sum(R[0]), -1.34786452)
    True
    >>> R = euler2mat(1, 2, 3, (0, 1, 0, 1))
    >>> np.allclose(np.sum(R[0]), -0.383436184)
    True
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]
    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
    ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
    cc, cs = ci*ck, ci*sk
    sc, ss = si*ck, si*sk

    M = np.eye(3)
    if repetition:
        M[i, i] = cj
        M[i, j] = sj*si
        M[i, k] = sj*ci
        M[j, i] = sj*sk
        M[j, j] = -cj*ss+cc
        M[j, k] = -cj*cs-sc
        M[k, i] = -sj*ck
        M[k, j] = cj*sc+cs
        M[k, k] = cj*cc-ss
    else:
        M[i, i] = cj*ck
        M[i, j] = sj*sc-cs
        M[i, k] = sj*cc+ss
        M[j, i] = cj*sk
        M[j, j] = sj*ss+cc
        M[j, k] = sj*cs-sc
        M[k, i] = -sj
        M[k, j] = cj*si
        M[k, k] = cj*ci
    return M
