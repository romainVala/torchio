import math
import torch
import warnings
import numpy as np
from collections import defaultdict
from typing import Dict, Tuple, List, Union, Optional
from scipy.interpolate import pchip_interpolate
from transforms3d.euler import euler2mat
try:
    import finufft
    _finufft = True
except ImportError:
    _finufft = False
import SimpleITK as sitk

from .. import RandomTransform
from ... import IntensityTransform
from ....data.io import nib_to_sitk
from ....data.subject import Subject
from ....metrics.fitpar_metrics import compute_motion_metrics #todo remove from PR

class RandomMotionFromTimeCourse(RandomTransform, IntensityTransform):
    r"""Simulate motion artefact

    Simulate motion artefact from a random time course of object positions.

    parameters relatated to simulate 3 types of displacement random noise swllow or sudden mouvement

    Args:
        nT (int): number of points of the time course
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
            if fitpars is not None previous parameter are not used
        maxGlobalDisp (float, float): (min, max) of the global translations. A random number is taken from this interval
            to scale each translations if they are bigger. If None, it won't be used
        maxGlobalRot same as  maxGlobalDisp but for Rotations
        fitpars : movement parameters to use (if specified, will be applied as such, no movement is simulated)
        displacement_shift (bool): whether or not to subtract the time course by the values of the center of the kspace
        freq_encoding_dim (tuple of ints): potential frequency encoding dims to use (one of them is randomly chosen)
        nufft (bool): whether or not to apply nufft (if false, no rotation is applied ! )
        oversampling_pct (float): percentage with which the data will be oversampled in the image domain prior to applying the motion
        verbose (bool): verbose

    Note currently on freq_encoding_dim=0 give the same ringing direction for rotation and translation, dim 1 and 2 are not coherent
    Note for suddenFrequency and swallowFrequency min max must differ and the max is never achieved, so to have 0 put (0,1)

    todo define parameter as degrees or translation in RandomMotion (specify chosen random distribution)
    todo short description of time course simu + reference gallichan retromoco

    .. note:: quite long execution times compare to other transforms
    .. warning:: extra dependency on finufft, if rotation is modeled, and SimpleElastix if coregistration
        is chosen

    """

    def __init__(self, nT: int = 200, maxDisp: Tuple[float, float] = (2,5), maxRot: Tuple[float, float] = (2,5),
                 noiseBasePars: Tuple[float, float] = (5,15), swallowFrequency: Tuple[float, float] = (0,5),
                 swallowMagnitude: Tuple[float, float] = (2,6), suddenFrequency: Tuple[int, int] = (0,5),
                 suddenMagnitude: Tuple[float, float] = (2,6), maxGlobalDisp: Tuple[float, float] = None,
                 maxGlobalRot: Tuple[float, float] = None, fitpars: Union[List, np.ndarray, str] = None,
                 seed: int = None, displacement_shift_strategy: str = None, freq_encoding_dim: List = [0],
                 oversampling_pct: float = 0.3, preserve_center_frequency_pct: float = 0,
                 nufft_type: str ='1D_type1', coregistration_to_orig=False, **kwargs):
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
        self.freq_encoding_choice = freq_encoding_dim
        self.frequency_encoding_dim = self._rand_choice(self.freq_encoding_choice)
        self.nufft_type = nufft_type
        self.coregistration_to_orig = coregistration_to_orig
        self.seed = seed
        if fitpars is None:
            self.fitpars = None
            self.simulate_displacement = True
        else:
            self.fitpars = read_fitpars(fitpars)
            self.simulate_displacement = False
        self.oversampling_pct = oversampling_pct
        if not _finufft:
            raise ImportError('finufft cannot be imported')
        self.to_substract = None


    def apply_transform(self, subject: Subject) -> Subject:

        arguments = defaultdict(dict)
        if self.simulate_displacement:
            self._simulate_random_trajectory()
        for image_name, image_dict in self.get_images_dict(subject).items():
            arguments["fitpars"][image_name] = self.fitpars
            arguments["displacement_shift_strategy"][image_name] = self.displacement_shift_strategy
            arguments["frequency_encoding_dim"][image_name] = self.frequency_encoding_dim
            arguments["oversampling_pct"][image_name] = self.oversampling_pct
            arguments["nufft_type"][image_name] = self.nufft_type
            arguments["coregistration_to_orig"][image_name] = self.coregistration_to_orig

        transform = MotionFromTimeCourse(**self.add_include_exclude(arguments))
        transformed = transform(subject)
        self._metrics = transform._metrics
        return transformed

    def random_params(self, maxDisp: Tuple[float, float] = (2,5), maxRot: Tuple[float, float] = (2,5),
                 noiseBasePars: Tuple[float, float] = (5,15), swallowFrequency: Tuple[float, float] = (0,5),
                 swallowMagnitude: Tuple[float, float] = (2,6), suddenFrequency: Tuple[int, int] = (0,5),
                 suddenMagnitude: Tuple[float, float] = (2,6), maxGlobalDisp: Tuple[float, float] = None,
                 maxGlobalRot: Tuple[float, float] = None):
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
        Simulates the parameters of the transformation through the vector fitpars using 6 dimensions (3 translations and
        3 rotations).
        """

        maxDisp, maxRot, noiseBasePars, swallowMagnitude, suddenMagnitude, swallowFrequency, suddenFrequency, \
        maxGlobalDisp, maxGlobalRot = self.random_params(maxDisp=self.maxDisp, maxRot=self.maxRot,
                 noiseBasePars=self.noiseBasePars, swallowFrequency=self.swallowFrequency,
                 swallowMagnitude=self.swallowMagnitude, suddenFrequency=self.suddenFrequency,
                 suddenMagnitude=self.suddenMagnitude, maxGlobalDisp=self.maxGlobalDisp,
                 maxGlobalRot=self.maxGlobalRot)

        if noiseBasePars > 0:
            fitpars = np.asarray([self.perlinNoise1D(self.nT, noiseBasePars) - 0.5 for _ in range(6)])
            fitpars[:3] *= maxDisp
            fitpars[3:] *= maxRot
        else:
            fitpars = np.zeros((6, self.nT))
        # add in swallowing-like movements - just to z direction and pitch
        if swallowFrequency > 0:
            swallowTraceBase = np.exp(-np.linspace(0, 100, self.nT))
            swallowTrace = np.zeros(self.nT)

            for i in range(swallowFrequency):
                rand_shifts = int(round(self._rand_uniform() * self.nT))
                rolled = np.roll(swallowTraceBase, rand_shifts, axis=0)
                swallowTrace += rolled

            fitpars[2, :] += swallowMagnitude[0] * swallowTrace
            fitpars[3, :] += swallowMagnitude[1] * swallowTrace

        # add in random sudden movements in any direction
        if suddenFrequency > 0:
            suddenTrace = np.zeros(fitpars.shape)

            for i in range(suddenFrequency):
                iT_sudden = int(np.ceil(self._rand_uniform() * self.nT))
                to_add = np.asarray([suddenMagnitude[0] * (2 * self._rand_uniform(shape=3) - 1),
                                     suddenMagnitude[1] * (2 * self._rand_uniform(shape=3) - 1)]).reshape((-1, 1))
                suddenTrace[:, iT_sudden:] = np.add(suddenTrace[:, iT_sudden:], to_add)

            fitpars += suddenTrace

        if self.preserve_center_frequency_pct:
            center = np.int32(np.floor(fitpars.shape[1] / 2))
            if self.displacement_shift_strategy == "center_zero":  # added here to remove global motion outside center
                to_substract = fitpars[:, center]
                to_substract_tile = np.tile(to_substract[..., np.newaxis], (1, fitpars.shape[1]))
                fitpars = fitpars - to_substract_tile

            nbpts = np.int32(np.floor(fitpars.shape[1] * self.preserve_center_frequency_pct / 2))
            fitpars[:, center - nbpts:center + nbpts] = 0

        # rescale to global max if asked
        # max is compute for all trans (or rot) diff
        if self.maxGlobalRot is not None :
            trans_diff = fitpars.T[:,None,:3] - fitpars.T[None,:,:3]  #numpy broadcating rule !
            ddtrans = np.linalg.norm(trans_diff, axis=2)
            ddrot = np.linalg.norm(fitpars.T[:,None,3:] - fitpars.T[None,:,3:] , axis=-1)

            fitpars[:3, :] = fitpars[:3, :] * maxGlobalDisp / ddtrans.max()
            fitpars[3:, :] = fitpars[3:, :] * maxGlobalRot / ddrot.max()

        self.fitpars = fitpars


    def read_fitpars(fitpars):
        '''
        :param fitpars:
        '''
        fpars = None
        if isinstance(fitpars, np.ndarray):
            fpars = fitpars
        elif isinstance(fitpars, list):
            fpars = np.asarray(fitpars)
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
                warnings.warn('There is still NaN in the fitpar, it will crash the nufft')
        return fpars


    def _rand_uniform(self, min=0.0, max=1.0, shape=1):
        rand = torch.FloatTensor(shape).uniform_(min, max)
        if shape == 1:
            return rand.item()
        return rand.numpy()


    def _rand_choice(self, array):
        chosen_idx = torch.randint(0, len(array), (1, ))
        return array[chosen_idx]


class MotionFromTimeCourse(IntensityTransform):
    def __init__(self, fitpars: Union[List, np.ndarray, str], displacement_shift_strategy: str,
                 frequency_encoding_dim: int, oversampling_pct: float,
                 nufft_type: str = '1D_type1', coregistration_to_orig: bool = False,
                 **kwargs):
        """
        parameters to simulate 3 types of displacement random noise swllow or sudden mouvement
        :param nT (int): number of points of the time course
        :param fitpars : movement parameters to use (if specified, will be applied as such, no movement is simulated)
        :param displacement_shift (bool): whether or not to substract the time course by the values of the center of the kspace
        :param freq_encoding_dim (tuple of ints): potential frequency encoding dims to use (one of them is randomly chosen)
        :param oversampling_pct (float): percentage with which the data will be oversampled in the image domain prior to applying the motion
        :param verbose (bool): verbose
        Note currently on freq_encoding_dim=0 give the same ringing direction for rotation and translation, dim 1 and 2 are not coherent
        Note fot suddenFrequency and swallowFrequency min max must differ and the max is never achieved, so to have 0 put (0,1)
        """
        super().__init__(**kwargs)
        self.displacement_shift_strategy = displacement_shift_strategy
        self.frequency_encoding_dim = frequency_encoding_dim
        self.fitpars = fitpars
        self.oversampling_pct = oversampling_pct
        if not _finufft:
            raise ImportError('finufft cannot be imported')
        self.to_substract = None
        self.nufft_type = nufft_type
        self.coregistration_to_orig = coregistration_to_orig
        self.args_names = ("fitpars", "displacement_shift_strategy", "frequency_encoding_dim",
                           "oversampling_pct", "nufft_type", "coregistration_to_orig")

    def apply_transform(self, subject: Subject) -> Subject:
        fitpars = self.fitpars
        displacement_shift_strategy = self.displacement_shift_strategy
        frequency_encoding_dim = self.frequency_encoding_dim
        oversampling_pct = self.oversampling_pct
        nufft_type = self.nufft_type
        coregistration_to_orig = self.coregistration_to_orig

        for image_name, image_dict in self.get_images_dict(subject).items():
            if self.arguments_are_dict():
                fitpars = self.fitpars[image_name]
                displacement_shift_strategy = self.displacement_shift_strategy[image_name]
                frequency_encoding_dim = self.frequency_encoding_dim[image_name]
                oversampling_pct = self.oversampling_pct[image_name]
                nufft_type = self.nufft_type[image_name]
                coregistration_to_orig = self.coregistration_to_orig[image_name]

            #image_data = np.squeeze(image_dict['data'])[..., np.newaxis, np.newaxis]
            #original_image = np.squeeze(image_data[:, :, :, 0, 0])
            original_image = np.squeeze(image_dict['data'])

            if oversampling_pct > 0.0:
                original_image_shape = original_image.shape
                original_image = self._oversubject(original_image, oversampling_pct)

            # fft
            im_freq_domain = self._fft_im(original_image)

            self._calc_dimensions(original_image.shape, frequency_encoding_dim=frequency_encoding_dim)

            if displacement_shift_strategy is not None: #demean before interpolation
                if '1D' in displacement_shift_strategy: #new strategy to demean, just 1D fft
                    fitpars, self.to_substract = self.demean_fitpars(fitpars, im_freq_domain, displacement_shift_strategy,
                                                                fast_dim = (frequency_encoding_dim, self.phase_encoding_dims[1]))
                    if self.arguments_are_dict():
                        self.fitpars[image_name] = fitpars
                    else:
                        self.fitpars = fitpars #important to save to get the correct fitpar in history

            if fitpars.shape[1] != self.phase_encoding_shape[0]:
                fitpars = self._interpolate_fitpars(fitpars, len_output=self.phase_encoding_shape[0])

            if 'type1' in nufft_type:
                corrupted_im = self._trans_and_nufft_type1(im_freq_domain, fitpars)
            else: #nufft_type2
                corrupted_im = self._trans_and_nufft_type2(original_image, fitpars)
            fitpars_interp = None #just to skip in _comput_motion_metrics

            # magnitude
            corrupted_im = abs(corrupted_im)
            if coregistration_to_orig:
                corrupted_im = self.ElastixRegisterAndReslice(corrupted_im, original_image)

            if oversampling_pct > 0.0:
                corrupted_im = self.crop_volume(corrupted_im, original_image_shape)

            image_dict["data"] = corrupted_im[np.newaxis, ...]
            image_dict['data'] = torch.from_numpy(image_dict['data']).float()

        #todo remove from PR
        self._metrics = compute_motion_metrics(fitpars, fitpars_interp, im_freq_domain,
                                               self.frequency_encoding_dim, self.phase_encoding_dims)

        return subject

    def _calc_dimensions(self, im_shape, frequency_encoding_dim):
        """
        calculate dimensions based on im_shape
        :param im_shape (list/tuple) : image shape
        - sets self.phase_encoding_dims, self.phase_encoding_shape, self.num_phase_encoding_steps, self.frequency_encoding_dim
        """
        pe_dims = [0, 1, 2]
        pe_dims.pop(frequency_encoding_dim)
        self.phase_encoding_dims = pe_dims
        im_shape = list(im_shape)
        self.im_shape = im_shape.copy()
        im_shape.pop(frequency_encoding_dim)
        self.phase_encoding_shape = im_shape
        #frequency_encoding_dim = len(im_shape) - 1 if frequency_encoding_dim == -1 else frequency_encoding_dim

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



    def _interpolate_fitpars(self, fpars, tr_fpars=None, tr_to_interpolate=2.4, len_output=250):
        fpars_length = fpars.shape[1]
        if tr_fpars is None: #case where fitpart where give as it in random motion (the all timecourse is fitted to kspace
            xp = np.linspace(0,1,fpars_length)
            x  = np.linspace(0,1,len_output)
        else:
            xp = np.asarray(range(fpars_length))*tr_fpars
            x = np.asarray(range(len_output))*tr_to_interpolate
        interpolated_fpars = np.asarray([np.interp(x, xp, fp) for fp in fpars])
        if xp[-1]<x[-1]:
            diff = x[-1] - xp[-1]
            npt_added = diff/tr_to_interpolate
            print(f'adding {npt_added:.1f}')
        return interpolated_fpars


    def _rotate_coordinates_1D_motion(self, fitpar, image_shape, Apply_inv_affine=True):
        # Apply_inv_affinne is True for the nufft_type1 and false
        # for the nuft2 we also add a 1 voxel translation to fitpar todo check with different resolution

        if Apply_inv_affine is False: #case for nufft_type2 add a one voxel shift
            off_center = np.array([(x / 2 - x // 2) * 2 for x in image_shape])  # one voxel shift if odd ! todo resolution ?
            # aff_offenter = np.eye(4); aff_offenter[:3,3] = -off_center
            fitpar[:3, :] = fitpar[:3, :] - np.repeat(np.expand_dims(off_center, 1), fitpar.shape[1], axis=1)

        lin_spaces = [np.linspace(-0.5, 0.5-1/x, x)*2*math.pi for x in image_shape]  # todo it suposes 1 vox = 1mm
        #remove 1/x to avoid small scaling

        meshgrids = np.meshgrid(*lin_spaces, indexing='ij')
        # pour une affine on ajoute de 1, dans les coordone du point, mais pour le augmented kspace on ajoute la phase initial, donc 0 ici
        meshgrids.append(np.zeros(image_shape))

        grid_coords = np.array([mg for mg in meshgrids]) #grid_coords = np.array([mg.flatten(order='F') for mg in meshgrids])
        grid_out = grid_coords
        #applied motion at each phase step (on the kspace grid plan)
        for nnp in range(fitpar.shape[1]):
            aff = get_matrix_from_euler_and_trans(fitpar[:,nnp])
            if Apply_inv_affine:
                aff = np.linalg.inv(aff)
            grid_plane = grid_out[:,:,nnp,:]
            shape_mem = grid_plane.shape
            grid_plane_moved = np.matmul(aff.T, grid_plane.reshape(4,shape_mem[1]*shape_mem[2])) #equ15 A.T * k0
            #grid_plane_moved = np.matmul( grid_plane.reshape(4,shape_mem[1]*shape_mem[2]).T, aff.T).T # r0.T * A.T
            grid_out[:, :, nnp, :] = grid_plane_moved.reshape(shape_mem)

        return grid_out


    def _trans_and_nufft_type1(self, freq_domain, fitpar):
        if not _finufft:
            raise ImportError('finufft not available')
        eps = 1E-7
        f = np.zeros(freq_domain.shape, dtype=np.complex128, order='F')

        grid_out = self._rotate_coordinates_1D_motion(fitpar, freq_domain.shape, Apply_inv_affine=True)

        phase_shift = grid_out[3].flatten(order='F')
        exp_phase_shift = np.exp( 1j * phase_shift)  #+1j -> x z == tio, y inverse

        freq_domain_data_flat = freq_domain.flatten(order='F')* exp_phase_shift # same F order as phase_shift if not inversion x z

        finufft.nufft3d1(grid_out[0].flatten(order='F'), grid_out[1].flatten(order='F'),
                         grid_out[2].flatten(order='F'), freq_domain_data_flat,
                         eps=eps, out=f, debug=0, spread_debug=0, spread_sort=2, fftw=0, modeord=0,
                         chkbnds=0, upsampfac=1.25, isign= 1)  # upsampling at 1.25 saves time at low precisions
        #im_out = f.reshape(image.shape, order='F')
        #im_out = f.flatten().reshape(image.shape)
        im_out = f / f.size

        return im_out


    def _trans_and_nufft_type2(self, image, fitpar, trans_last=False):
        if not _finufft:
            raise ImportError('finufft not available')
        eps = 1E-7
        grid_out = self._rotate_coordinates_1D_motion(fitpar, image.shape, Apply_inv_affine=False)

        f = np.zeros(grid_out[0].shape, dtype=np.complex128, order='F').flatten() #(order='F')
        ip = np.asfortranarray(image.numpy().astype(complex) )

        finufft.nufft3d2(grid_out[0].flatten(order='F'), grid_out[1].flatten(order='F'), grid_out[2].flatten(order='F'), ip,
                         eps=eps, out=f, debug=0, spread_debug=0, spread_sort=2, fftw=0, modeord=0,
                         chkbnds=0, upsampfac=1.25, isign=-1)  # upsampling at 1.25 saves time at low precisions

        f = f * np.exp(-1j * grid_out[3].flatten(order='F'))
        f = f.reshape(ip.shape,order='F')
        #f = np.ascontiguousarray(f)  #pas l'aire de changer grand chose
        iout = abs( np.fft.ifftshift(np.fft.ifftn(f)))
        #iout = abs( np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(f))))
        return iout


    def demean_fitpars(self, fitpars, original_image_fft, displacement_shift_strategy,
                       fast_dim=(0,2)):

        to_substract = np.array([0, 0, 0, 0, 0, 0])

        #ok only 1D fftp
        #coef_shaw = np.sqrt( np.sum(abs(original_image_fft**2), axis=(0,2)) ) ;
        #should be equivalent if fft is done from real image, but not if the phase is acquired, CF Todd 2015 "Prospective motion correction of 3D echo-planar imaging data for functional MRI using optical tracking"
        if displacement_shift_strategy == '1D_wTF':
            coef_shaw = np.abs( np.sqrt(np.sum( original_image_fft * np.conjugate(original_image_fft), axis=fast_dim )));
        if displacement_shift_strategy == '1D_wTF2':
            #coef_shaw = np.abs( np.sum( original_image_fft * np.conjugate(original_image_fft), axis=fast_dim ));
            coef_shaw = np.abs( np.sum( original_image_fft **2, axis=fast_dim ))
            coef_shaw = coef_shaw / np.sum(coef_shaw)


        if fitpars.shape[1] != coef_shaw.shape[0] :
            #just interpolate end to end. at image slowest dimention size
            fitpars = self._interpolate_fitpars(fitpars, len_output=coef_shaw.shape[0])

        to_substract = np.zeros(6)
        for i in range(0,6):
            to_substract[i] = np.sum(fitpars[i,:] * coef_shaw) / np.sum(coef_shaw)
            fitpars[i,:] = fitpars[i,:] - to_substract[i]

        return fitpars, to_substract  #note the 1D fitpar, may have been interpolated to phase dim but should not matter for the rest


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

