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

from .. import RandomTransform
from ... import IntensityTransform


class RandomMotionFromTimeCourse(RandomTransform, IntensityTransform):

    def __init__(self, nT: int = 200, maxDisp: Tuple[float, float] = (2,5), maxRot: Tuple[float, float] = (2,5),
                 noiseBasePars: Tuple[float, float] = (5,15), swallowFrequency: Tuple[float, float] = (0,5),
                 swallowMagnitude: Tuple[float, float] = (2,6), suddenFrequency: Tuple[int, int] = (0,5),
                 suddenMagnitude: Tuple[float, float] = (2,6), maxGlobalDisp: Tuple[float, float] = None,
                 maxGlobalRot: Tuple[float, float] = None, fitpars: Union[List, np.ndarray, str] = None,
                 seed: int = None, displacement_shift_strategy: str = None, freq_encoding_dim: List = [0],
                 tr: float = 2.3, es: float = 4E-3, oversampling_pct: float = 0.3,
                 preserve_center_frequency_pct: float = 0, correct_motion: bool = False, res_dir: str = None,
                 **kwargs):
        """
        parameters to simulate 3 types of displacement random noise swllow or sudden mouvement
        :param nT (int): number of points of the time course
        :param maxDisp (float, float): (min, max) value of displacement in the perlin noise (useless if noiseBasePars is 0)
        :param maxRot (float, float): (min, max) value of rotation in the perlin noise (useless if noiseBasePars is 0)
        :param noiseBasePars (float, float): (min, max) base value of the perlin noise to generate for the time course
        optional (float, float, float) where the third is the probability to perform this type of noise
        :param swallowFrequency (int, int): (min, max) number of swallowing movements to generate in the time course
        optional (float, float, float) where the third is the probability to perform this type of noise
        :param swallowMagnitude (float, float): (min, max) magnitude of the swallowing movements to generate
        :param suddenFrequency (int, int): (min, max) number of sudden movements to generate in the time course
        optional (float, float, float) where the third is the probability to perform this type of noise
        :param suddenMagnitude (float, float): (min, max) magnitude of the sudden movements to generate
        if fitpars is not None previous parameter are not used
        :param maxGlobalDisp (float, float): (min, max) of the global translations. A random number is taken from this interval
        to scale each translations if they are bigger. If None, it won't be used
        :param maxGlobalRot same as  maxGlobalDisp but for Rotations
        :param fitpars : movement parameters to use (if specified, will be applied as such, no movement is simulated)
        :param displacement_shift (bool): whether or not to subtract the time course by the values of the center of the kspace
        :param freq_encoding_dim (tuple of ints): potential frequency encoding dims to use (one of them is randomly chosen)
        :param tr (float): repetition time of the data acquisition (used for interpolating the time course movement)
        :param es (float): echo spacing time of the data acquisition (used for interpolating the time course movement)
        :param nufft (bool): whether or not to apply nufft (if false, no rotation is applied ! )
        :param oversampling_pct (float): percentage with which the data will be oversampled in the image domain prior to applying the motion
        :param verbose (bool): verbose
        Note currently on freq_encoding_dim=0 give the same ringing direction for rotation and translation, dim 1 and 2 are not coherent
        Note for suddenFrequency and swallowFrequency min max must differ and the max is never achieved, so to have 0 put (0,1)
        """
        super().__init__(**kwargs)
        self.tr = tr
        self.es = es
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
        self.correct_motion = correct_motion
        self.to_substract = None
        self.res_dir = res_dir
        self.nb_saved = 0
        """
        self.args_names = ("fitpars", "oversampling_pct", "displacement_shift_strategy", "frequency_encoding_dim",
                           "tr", "es")
        """

    def apply_transform(self, sample):
        arguments = defaultdict(dict)
        if self.simulate_displacement:
            self._simulate_random_trajectory()
        for image_name, image_dict in self.get_images_dict(sample).items():
            """
            image_data = np.squeeze(image_dict['data'])[..., np.newaxis, np.newaxis]
            original_image = np.squeeze(image_data[:, :, :, 0, 0])

            if self.oversampling_pct > 0.0:
                original_image_shape = original_image.shape
                original_image = self._oversample(original_image, self.oversampling_pct)

            self._calc_dimensions(original_image.shape)
            """
            arguments["fitpars"][image_name] = self.fitpars
            arguments["displacement_shift_strategy"][image_name] = self.displacement_shift_strategy
            arguments["frequency_encoding_dim"][image_name] = self.frequency_encoding_dim
            arguments["oversampling_pct"][image_name] = self.oversampling_pct
            arguments["tr"][image_name] = self.tr
            arguments["es"][image_name] = self.es
            arguments["correct_motion"][image_name] = self.correct_motion

        transform = MotionFromTimeCourse(**self.add_include_exclude(arguments))
        transformed = transform(sample)
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

    def _calc_dimensions(self, im_shape):
        """
        calculate dimensions based on im_shape
        :param im_shape (list/tuple) : image shape
        - sets self.phase_encoding_dims, self.phase_encoding_shape, self.num_phase_encoding_steps, self.frequency_encoding_dim
        """
        pe_dims = [0, 1, 2]
        pe_dims.pop(self.frequency_encoding_dim)
        self.phase_encoding_dims = pe_dims
        im_shape = list(im_shape)
        self.im_shape = im_shape.copy()
        im_shape.pop(self.frequency_encoding_dim)
        self.phase_encoding_shape = im_shape
        self.frequency_encoding_dim = len(self.im_shape) - 1 if self.frequency_encoding_dim == -1 \
            else self.frequency_encoding_dim

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
            print(f'max trans {ddtrans.max()} rot {ddrot.max()} setting to {maxGlobalDisp} {maxGlobalRot }')
            fitpars[:3, :] = fitpars[:3, :] * maxGlobalDisp / ddtrans.max()
            fitpars[3:, :] = fitpars[3:, :] * maxGlobalRot / ddrot.max()

        self.fitpars = fitpars

    def save_to_dir(self, image_dict):

        volume_path = image_dict['path']
        dd = volume_path.split('/')
        volume_name = dd[len(dd)-2] + '_' + image_dict['stem']
        nb_saved = image_dict['index']
        import os
        resdir = self.res_dir + '/mvt_param/'
        if not os.path.isdir(resdir): os.mkdir(resdir)

        fname = resdir + 'ssim_{}_N{:05d}_suj_{}'.format(image_dict['metrics']['ssim'],
                                                    nb_saved, volume_name)
        np.savetxt(fname + '_mvt.csv', self.fitpars, delimiter=',')

    def _rand_uniform(self, min=0.0, max=1.0, shape=1):
        rand = torch.FloatTensor(shape).uniform_(min, max)
        if shape == 1:
            return rand.item()
        return rand.numpy()

    def _rand_choice(self, array):
        chosen_idx = torch.randint(0, len(array), (1, ))
        return array[chosen_idx]


def create_rotation_matrix_3d(angles):
    """
    given a list of 3 angles, create a 3x3 rotation matrix that describes rotation about the origin
    :param angles (list or numpy array) : rotation angles in 3 dimensions
    :return (numpy array) : rotation matrix 3x3
    """

    mat1 = np.array([[1., 0., 0.],
                     [0., math.cos(angles[0]), math.sin(angles[0])],
                     [0., -math.sin(angles[0]), math.cos(angles[0])]],
                    dtype='float')

    mat2 = np.array([[math.cos(angles[1]), 0., math.sin(angles[1])],
                     [0., 1., 0.],
                     [-math.sin(angles[1]), 0., math.cos(angles[1])]],
                    dtype='float')

    mat3 = np.array([[math.cos(angles[2]), math.sin(angles[2]), 0.],
                     [-math.sin(angles[2]), math.cos(angles[2]), 0.],
                     [0., 0., 1.]],
                    dtype='float')

    mat = (mat1 @ mat2) @ mat3
    return mat


def calculate_mean_FD_P(motion_params):
    """
    Method to calculate Framewise Displacement (FD)  as per Power et al., 2012
    """
    translations = np.transpose(np.abs(np.diff(motion_params[0:3, :])))
    rotations = np.transpose(np.abs(np.diff(motion_params[3:6, :])))

    fd = np.sum(translations, axis=1) + (50 * np.pi / 180) * np.sum(rotations, axis=1)
    #fd = np.insert(fd, 0, 0)

    return np.mean(fd)


def calculate_mean_Disp_P_old(motion_params, weights=None):
    """
    Same as previous, but without taking the diff between frame
    """
    if weights is None:
        #print('motion pa shape {}'.format(motion_params.shape))
        translations = np.transpose(np.abs(motion_params[0:3, :]))
        rotations = np.transpose(np.abs(motion_params[3:6, :]))
        fd = np.mean(translations, axis=1) + (50 * np.pi / 180) * np.mean(rotations, axis=1)
        #print('df shape {}'.format(fd.shape))

        return np.mean(fd)
    else:
        #print('motion pa shape {}'.format(motion_params.shape))

        motion_params = motion_params.reshape([6, np.prod(weights.shape)])
        mp = np.stack( [mm*weights.flatten() for mm in motion_params] )

        translations = np.transpose(np.abs(mp[0:3, :]))
        rotations = np.transpose(np.abs(mp[3:6, :]))

        fd = np.mean(translations, axis=1) + (50 * np.pi / 180) * np.mean(rotations, axis=1)
        #print('df shape {}'.format(fd.shape))

        return np.sum(fd) / np.sum(weights)

def calculate_mean_Disp_P(motion_params, weights=None, rmax=80):
    """
    :param motion_params: supose of size 6 * nbt
    :param weights: size nbt
    :param rmax:
    :return:
    """
    nbt = motion_params.shape[1]
    if weights is None:
        weights = np.ones(nbt)
    motion_params = motion_params - np.sum( motion_params * weights, axis=1, keepdims=True) / np.sum(weights)
    fd = np.zeros(nbt)
    #compute Frame displacement of each frame
    for tt in range(nbt):
        fp = motion_params[:,tt]
        fd[tt] = np.sum(np.abs(fp[:3])) + (rmax * np.pi/180) * np.sum(np.abs(fp[3:6]))
    return np.sum(fd * weights) / np.sum(weights)

def calculate_mean_Disp_J(motion_params, rmax=80, center_of_mass=np.array([0,0,0]), weights=None ):
    """
    Method to calculate mean displacement as per Jenkinson et al. 2002
    motion_params, 6 euler params, of shape [6 nbt]
    rmax radius of the sphere
    center_of_mass of the sphere
    we remove the weighted mean to the motion_param (by analogie with RMSE computation, which is minimum when the weigthed mean is substracted=
    """
    #transform euler fitpar to affine
    nbt = motion_params.shape[1]
    if weights is None:
        weights = np.ones(nbt)

    motion_params = motion_params - np.sum( motion_params * weights, axis=1, keepdims=True) / np.sum(weights)

    # T_rb_prev = pm[0].reshape(4, 4)   # for Frame displacement
    fd = np.zeros(nbt)
    for i in range(1, nbt):
        P = np.hstack((motion_params[:, i], np.array([1, 1, 1, 0, 0, 0])))
        T_rb =  spm_matrix(P, order=0)
        #M = np.dot(T_rb, np.linalg.inv(T_rb_prev)) - np.eye(4)  #for Frame displacmeent
        Ma = T_rb - np.eye(4)  #for Frame displacmeent
        A = Ma[0:3, 0:3]
        bt = Ma[0:3, 3]
        bt = bt + np.dot(A,center_of_mass)
        fd[i] = np.sqrt( (rmax * rmax / 5) * np.trace(np.dot(A.T, A)) + np.dot(bt.T, bt) )
        #T_rb_prev = T_rb

    return np.sum(fd * weights) / np.sum(weights)

def calculate_mean_RMSE_trans_rot(fit_pars, weights=None):
    #Minimum RMSE when fit_pars have a weighted mean to zero
    nbt = fit_pars.shape[1]
    if weights is None:
        weights = np.ones(nbt)
    #remove weighted mean
    fit_pars = fit_pars - np.sum( fit_pars * weights, axis=1, keepdims=True) / np.sum(weights)

    r1 = np.sum(fit_pars[0:3] * fit_pars[0:3], axis=0)
    r2 = np.sum(fit_pars[3:6] * fit_pars[3:6], axis=0)

    resT = np.sqrt( np.sum(r1*weights) / np.sum(weights) )
    resR = np.sqrt( np.sum(r2*weights) / np.sum(weights) )
    return  resT, resR

def calculate_mean_RMSE_displacment(fit_pars, coef=None):
    """
    very crude approximation where rotation in degree and translation are average ...
    """
    if coef is None:
        r1 = np.sqrt(np.sum(fit_pars[0:3] * fit_pars[0:3], axis=0))
        rms1 = np.sqrt(np.mean(r1 * r1))
        r2 = np.sqrt(np.sum(fit_pars[3:6] * fit_pars[3:6], axis=0))
        rms2 = np.sqrt(np.mean(r2 * r2))
        res = (rms1 + rms2) / 2
    else:
        r1 = np.sqrt(np.sum(fit_pars[0:3] * fit_pars[0:3], axis=0))
        #rms1 = np.sqrt(np.mean(r1 * r1))
        rms1  = np.sqrt(np.sum(coef*r1*r1)/np.sum(coef))
        r2 = np.sqrt(np.sum(fit_pars[3:6] * fit_pars[3:6], axis=0))
        #rms2 = np.sqrt(np.mean(r2 * r2))
        rms2 = np.sqrt(np.sum(coef * r2 * r2) / np.sum(coef))

        res = (rms1 + rms2) / 2

    return res


def spm_matrix(P, order=0):
    """
    FORMAT [A] = spm_matrix(P )
    P(0)  - x translation
    P(1)  - y translation
    P(2)  - z translation
    P(3)  - x rotation around x in degree
    P(4)  - y rotation around y in degree
    P(5)  - z rotation around z in degree
    P(6)  - x scaling
    P(7)  - y scaling
    P(8)  - z scaling
    P(9) - x affine
    P(10) - y affine
    P(11) - z affine

    order - application order of transformations. if order (the Default): T*R*Z*S if order==0 S*Z*R*T
    """
    convert_to_torch=False
    if torch.is_tensor(P):
        P = P.numpy()
        convert_to_torch=True

    #[P[3], P[4], P[5]] = [P[3] * 180 / np.pi, P[4] * 180 / np.pi, P[5] * 180 / np.pi]  # degre to radian
    P[3], P[4], P[5] = P[3]*np.pi/180, P[4]*np.pi/180, P[5]*np.pi/180 #degre to radian

    T = np.array([[1,0,0,P[0]],[0,1,0,P[1]],[0,0,1,P[2]],[0,0,0,1]])
    R1 =  np.array([[1,0,0,0],
                    [0,np.cos(P[3]),np.sin(P[3]),0],#sing change compare to spm because neuro versus radio ?
                    [0,-np.sin(P[3]),np.cos(P[3]),0],
                    [0,0,0,1]])
    R2 =  np.array([[np.cos(P[4]),0,np.sin(P[4]),0],
                    [0,1,0,0],
                    [-np.sin(P[4]),0,np.cos(P[4]),0],
                    [0,0,0,1]])
    R3 =  np.array([[np.cos(P[5]),np.sin(P[5]),0,0],  #sing change compare to spm because neuro versus radio ?
                    [-np.sin(P[5]),np.cos(P[5]),0,0],
                    [0,0,1,0],
                    [0,0,0,1]])

    #R = R3.dot(R2.dot(R1)) #fsl convention (with a sign difference)
    R = (R1.dot(R2)).dot(R3)

    Z = np.array([[P[6],0,0,0],[0,P[7],0,0],[0,0,P[8],0],[0,0,0,1]])
    S = np.array([[1,P[9],P[10],0],[0,1,P[11],0],[0,0,1,0],[0,0,0,1]])
    if order==0:
        A = S.dot(Z.dot(R.dot(T)))
    else:
        A = T.dot(R.dot(Z.dot(S)))

    if convert_to_torch:
        A = torch.from_numpy(A).float()

    return A


class MotionFromTimeCourse(IntensityTransform):
    def __init__(self, fitpars: Union[List, np.ndarray, str], displacement_shift_strategy: str,
                 frequency_encoding_dim: int, tr: float, es: float, oversampling_pct: float,
                 correct_motion: bool = False, **kwargs):
        """
        parameters to simulate 3 types of displacement random noise swllow or sudden mouvement
        :param nT (int): number of points of the time course
        :param fitpars : movement parameters to use (if specified, will be applied as such, no movement is simulated)
        :param displacement_shift (bool): whether or not to substract the time course by the values of the center of the kspace
        :param freq_encoding_dim (tuple of ints): potential frequency encoding dims to use (one of them is randomly chosen)
        :param tr (float): repetition time of the data acquisition (used for interpolating the time course movement)
        :param es (float): echo spacing time of the data acquisition (used for interpolating the time course movement)
        :param oversampling_pct (float): percentage with which the data will be oversampled in the image domain prior to applying the motion
        :param verbose (bool): verbose
        Note currently on freq_encoding_dim=0 give the same ringing direction for rotation and translation, dim 1 and 2 are not coherent
        Note fot suddenFrequency and swallowFrequency min max must differ and the max is never achieved, so to have 0 put (0,1)
        """
        super().__init__(**kwargs)
        self.tr = tr
        self.es = es
        self.displacement_shift_strategy = displacement_shift_strategy
        self.frequency_encoding_dim = frequency_encoding_dim
        self.fitpars = fitpars
        self.oversampling_pct = oversampling_pct
        if not _finufft:
            raise ImportError('finufft cannot be imported')
        self.correct_motion = correct_motion
        self.to_substract = None
        self.nb_saved = 0
        self.args_names = ("fitpars", "displacement_shift_strategy", "frequency_encoding_dim", "tr", "es",
                           "oversampling_pct", "correct_motion")

    def apply_transform(self, sample):
        fitpars = self.fitpars
        displacement_shift_strategy = self.displacement_shift_strategy
        frequency_encoding_dim = self.frequency_encoding_dim
        oversampling_pct = self.oversampling_pct
        tr = self.tr
        es = self.es
        correct_motion = self.correct_motion
        for image_name, image_dict in self.get_images_dict(sample).items():
            if self.arguments_are_dict():
                fitpars = self.fitpars[image_name]
                displacement_shift_strategy = self.displacement_shift_strategy[image_name]
                frequency_encoding_dim = self.frequency_encoding_dim[image_name]
                oversampling_pct = self.oversampling_pct[image_name]
                tr = self.tr[image_name]
                es = self.es[image_name]
                correct_motion = self.correct_motion[image_name]

            image_data = np.squeeze(image_dict['data'])[..., np.newaxis, np.newaxis]
            original_image = np.squeeze(image_data[:, :, :, 0, 0])
            if oversampling_pct > 0.0:
                original_image_shape = original_image.shape
                original_image = self._oversample(original_image, oversampling_pct)

            # fft
            im_freq_domain = self._fft_im(original_image)
            print(f'displacement_shift_strategy is {displacement_shift_strategy}')
            self._calc_dimensions(original_image.shape, frequency_encoding_dim=frequency_encoding_dim)

            if displacement_shift_strategy is not None:
                if '1D' in displacement_shift_strategy: #new strategy to demean, just 1D fft
                    fitpars, self.to_substract = demean_fitpars(fitpars, im_freq_domain, displacement_shift_strategy,
                                                                fast_dim = (frequency_encoding_dim, self.phase_encoding_dims[1]))
                #not sure why is the second phase_encoding_dims

            if fitpars.ndim == 4:  # we assume the interpolation has been done on the input
                fitpars_interp = fitpars
            else:
                fitpars_interp = _interpolate_space_timing(fitpars=fitpars, es=es, tr=tr,
                                                           phase_encoding_shape=self.phase_encoding_shape,
                                                           frequency_encoding_dim=frequency_encoding_dim)
                fitpars_interp = _tile_params_to_volume_dims(params_to_reshape=fitpars_interp,
                                                             im_shape=self.im_shape)
            if displacement_shift_strategy is not None:
                if not '1D' in displacement_shift_strategy:
                    fitpars_interp, self.to_substract = demean_fitpars(fitpars_interp=fitpars_interp, original_image_fft=im_freq_domain,
                                                    displacement_shift_strategy=displacement_shift_strategy)

            fitpars_vox = fitpars_interp.reshape((6, -1))
            translations, rotations = fitpars_vox[:3], np.radians(fitpars_vox[3:])
            translated_im_freq_domain = _translate_freq_domain(freq_domain=im_freq_domain,
                                                               translations=translations)
            apply_rotation = np.sum(np.abs(rotations.flatten())) > 0
            # iNufft for rotations
            if _finufft and apply_rotation:
                corrupted_im = _nufft(freq_domain_data=translated_im_freq_domain, rotations=rotations,
                                      im_shape=original_image.shape, frequency_encoding_dim=frequency_encoding_dim,
                                      phase_encoding_shape=self.phase_encoding_shape)
                corrupted_im = corrupted_im / corrupted_im.size  # normalize

            else:
                corrupted_im = self._ifft_im(translated_im_freq_domain)

            if correct_motion:
                corrected_im = self.do_correct_motion(corrupted_im.copy(), fitpars_interp, frequency_encoding_dim, self.phase_encoding_shape)
                image_dict["data_cor"] = corrected_im[np.newaxis, ...]
                image_dict['data_cor'] = torch.from_numpy(image_dict['data_cor']).float()

            # magnitude
            corrupted_im = abs(corrupted_im)

            if oversampling_pct > 0.0:
                corrupted_im = self.crop_volume(corrupted_im, original_image_shape)

            image_dict["data"] = corrupted_im[np.newaxis, ...]
            image_dict['data'] = torch.from_numpy(image_dict['data']).float()
        """
        if self.res_dir is not None:
            self.save_to_dir(image_dict)
        """
        self._compute_motion_metrics(fitpars=fitpars, fitpars_interp=fitpars_interp, img_fft=im_freq_domain)

        return sample

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

    def _compute_motion_metrics(self, fitpars, fitpars_interp, img_fft):
        self._metrics = dict()
        self._metrics["mean_DispP"] = calculate_mean_Disp_P(fitpars)
        self._metrics["mean_DispJ"] = calculate_mean_Disp_J(fitpars)
        #self._metrics["rmse_Disp"] = calculate_mean_RMSE_displacment(fitpars)
        self._metrics["rmse_Trans"], self._metrics["rmse_Rot"] = calculate_mean_RMSE_trans_rot(fitpars)

        frequency_encoding_dim = self.frequency_encoding_dim
        if isinstance(frequency_encoding_dim,dict):
            for k in frequency_encoding_dim.keys(): #kind of weird, but we use with only one dict ... (if not probabely same value)
                val = frequency_encoding_dim[k]
            frequency_encoding_dim=val
        dim_to_average = (frequency_encoding_dim, self.phase_encoding_dims[1] ) # warning, why slowest dim the phase_encoding_dims[0]

        coef_TF = np.sum(abs(img_fft), axis=(0,2)) ;
        coef_shaw = np.sqrt( np.sum(abs(img_fft**2), axis=(0,2)) ) ;
        # I do not see diff, but may be better to write with complex conjugate, here the fft is done on abs image, so I guess the
        # phase does not matter (Cf todd 2015)
        #print(f'averagin TF coef on dim {dim_to_average} shape coef {coef_TF.shape}')
        if fitpars.shape[1] != coef_TF.shape[0] :
            #just interpolate end to end. at image slowest dimention size
            fitpars = _interpolate_fitpars(fitpars, len_output=coef_TF.shape[0])
            #print(f'interp fitpar for wcoef new shape {fitpars.shape}')

        self._metrics["meanDispJ_wTF"]  = calculate_mean_Disp_J(fitpars,  weights=coef_TF)
        self._metrics["meanDispJ_wSH"]  = calculate_mean_Disp_J(fitpars,  weights=coef_shaw)
        self._metrics["meanDispJ_wTF2"] = calculate_mean_Disp_J(fitpars,  weights=coef_TF**2)
        self._metrics["meanDispJ_wSH2"] = calculate_mean_Disp_J(fitpars,  weights=coef_shaw**2)

        self._metrics["meanDispP_wSH"] = calculate_mean_Disp_P(fitpars,  weights=coef_shaw)
        self._metrics["rmse_Trans_wSH"], self._metrics["rmse_Rot_wSH"] = calculate_mean_RMSE_trans_rot(fitpars, weights=coef_shaw)
        self._metrics["rmse_Trans_wTF2"], self._metrics["rmse_Rot_wTF2"] = calculate_mean_RMSE_trans_rot(fitpars, weights=coef_TF**2)

        #compute meand disp as weighted mean (weigths beeing TF coef)
        w_coef = np.abs(img_fft)
        ff = fitpars_interp
        disp_mean=[]
        for i in range(0, 6):
            ffi = ff[i].reshape(-1)
            w_coef_flat = w_coef.reshape(-1)
            self._metrics[f'wTF_Disp_{i}'] = np.sum(ffi * w_coef_flat) / np.sum(w_coef_flat)
            self._metrics[f'wTF2_Disp_{i}'] = np.sum(ffi * w_coef_flat**2) / np.sum(w_coef_flat**2)
            disp_mean.append(  np.sum(np.abs(ffi) * w_coef_flat) / np.sum(w_coef_flat) )
        #self._metrics['wTF_absDisp_t'] = np.mean(disp_mean[:3])
        #self._metrics['wTF_absDisp_r'] = np.mean(disp_mean[3:])
        #self._metrics['wTF_absDisp_a'] = np.mean(disp_mean)
        ff = fitpars
        for i in range(0, 6):
            ffi = ff[i].reshape(-1)
            self._metrics[f'wTFshort_Disp_{i}'] = np.sum(ffi * coef_TF) / np.sum(coef_TF)
            self._metrics[f'wTFshort2_Disp_{i}'] = np.sum(ffi * coef_TF**2) / np.sum(coef_TF**2)
            self._metrics[f'wSH_Disp_{i}'] = np.sum(ffi * coef_shaw) / np.sum(coef_shaw)
            self._metrics[f'wSH2_Disp_{i}'] = np.sum(ffi * coef_shaw**2) / np.sum(coef_shaw**2)
            self._metrics[f'mean_Disp_{i}'] = np.mean(ffi)
            self._metrics[f'center_Disp_{i}'] = ffi[ffi.shape[0]//2]
        #at the end only SH and SH2 seems ok
        # TF2 == SH2  but TFshort==TF and TFshort2 < TF2 !
    def do_correct_motion(self, image, fitpars_interp, frequency_encoding_dim, phase_encoding_shape ):
        #works only if pure rotation or pure translation
        im_freq_domain = self._fft_im(image)
        # print('translation')
        fitpars_vox = fitpars_interp.reshape((6, -1))
        translations, rotations = fitpars_vox[:3], np.radians(fitpars_vox[3:])

        #arg fftshift is needed, for translation only
        im_freq_domain = np.fft.fftshift(im_freq_domain)
        translated_im_freq_domain = _translate_freq_domain(im_freq_domain, translations, inv_transfo=True)
        translated_im_freq_domain = np.fft.fftshift(translated_im_freq_domain)

        # iNufft for rotations
        apply_rotation = np.sum(np.abs(fitpars_interp[3:,:].flatten())) > 0
        # iNufft for rotations
        if _finufft and apply_rotation:

            corrected_im = _nufft(freq_domain_data=translated_im_freq_domain, rotations=rotations,
                                  im_shape=image.shape, frequency_encoding_dim=frequency_encoding_dim,
                                  phase_encoding_shape=phase_encoding_shape, inv_transfo=True)
            corrected_im = corrected_im / corrected_im.size  # normalize
        else:
            corrected_im = self._ifft_im(translated_im_freq_domain)
        # magnitude
        corrected_im = abs(corrected_im)

        return corrected_im

    def do_correct_motion_inv(self, image, fitpars_interp, frequency_encoding_dim, phase_encoding_shape ):
        #does not work either
        im_freq_domain = self._fft_im(image)

        # print('translation')
        fitpars_vox = fitpars_interp.reshape((6, -1))
        translations, rotations = fitpars_vox[:3], np.radians(fitpars_vox[3:])
        apply_rotation = np.sum(np.abs(fitpars_interp[3:,:].flatten())) > 0
        # iNufft for rotations
        if _finufft and apply_rotation:
            corrected_im = _nufft(freq_domain_data=im_freq_domain, rotations=rotations,
                                  im_shape=image.shape, frequency_encoding_dim=frequency_encoding_dim,
                                  phase_encoding_shape=phase_encoding_shape, inv_transfo=True)
            corrected_im = corrected_im / corrected_im.size  # normalize

            im_freq_domain = self._fft_im(corrected_im)

        im_freq_domain = np.fft.fftshift(im_freq_domain)
        translated_im_freq_domain = _translate_freq_domain(im_freq_domain, translations, inv_transfo=True)

        corrected_im = self._ifft_im(translated_im_freq_domain)

        corrected_im = abs(corrected_im)

        return corrected_im


def _interpolate_space_timing_1D(fitpars, nT, phase_encoding_shape, frequency_encoding_dim, phase_encoding_dims):
    n_phase = phase_encoding_shape[0]
    nT = nT
    # Time steps
    mg_total = np.linspace(0, 1, n_phase)
    # Equidistant time spacing
    teq = np.linspace(0, 1, nT)
    # Actual interpolation
    fitpars_interp = np.asarray([np.interp(mg_total, teq, params) for params in fitpars])
    # Reshaping to phase encoding dimensions
    # Add missing dimension
    fitpars_interp = np.expand_dims(fitpars_interp,
                                    axis=[frequency_encoding_dim + 1, phase_encoding_dims[1] + 1])
    return fitpars_interp


def _translate_freq_domain(freq_domain, translations, inv_transfo=False):
    """
    image domain translation by adding phase shifts in frequency domain
    :param freq_domain - frequency domain data 3d numpy array:
    :return frequency domain array with phase shifts added according to self.translations:
    """
    translations = -translations if inv_transfo else translations

    lin_spaces = [np.linspace(-0.5, 0.5, x) for x in freq_domain.shape]  # todo it suposes 1 vox = 1mm
    meshgrids = np.meshgrid(*lin_spaces, indexing='ij')
    grid_coords = np.array([mg.flatten() for mg in meshgrids])

    phase_shift = np.multiply(grid_coords, translations).sum(axis=0)  # phase shift is added
    exp_phase_shift = np.exp(-2j * math.pi * phase_shift)
    freq_domain_translated = exp_phase_shift * freq_domain.reshape(-1)

    return freq_domain_translated.reshape(freq_domain.shape)


def _tile_params_to_volume_dims(params_to_reshape, im_shape):
    target_shape = [6] + im_shape
    data_shape = params_to_reshape.shape
    tiles = np.floor_divide(target_shape, data_shape, dtype=int)
    return np.tile(params_to_reshape, reps=tiles)


def _interpolate_space_timing(fitpars, es, tr, phase_encoding_shape, frequency_encoding_dim):
    n_phase, n_slice = phase_encoding_shape[0], phase_encoding_shape[1]
    # Time steps
    t_steps = n_phase * tr
    # Echo spacing dimension
    dim_es = np.cumsum(es * np.ones(n_slice)) - es
    dim_tr = np.cumsum(tr * np.ones(n_phase)) - tr
    # Build grid
    mg_es, mg_tr = np.meshgrid(*[dim_es, dim_tr])
    mg_total = mg_es + mg_tr  # MP-rage timing
    # Flatten grid and sort values
    mg_total = np.sort(mg_total.reshape(-1))
    # Equidistant time spacing
    teq = np.linspace(0, t_steps, fitpars.shape[1])
    # Actual interpolation
    fitpars_interp = np.asarray([np.interp(mg_total, teq, params) for params in fitpars])
    # Reshaping to phase encoding dimensions
    fitpars_interp = fitpars_interp.reshape([6] + phase_encoding_shape)
    # Add missing dimension
    fitpars_interp = np.expand_dims(fitpars_interp, axis=frequency_encoding_dim + 1)
    return fitpars_interp

def _interpolate_fitpars(fpars, tr_fpars=None, tr_to_interpolate=2.4, len_output=250):
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


def _rotate_coordinates(rotations, im_shape, frequency_encoding_dim, phase_encoding_shape, inv_transfo=False):
    """
    :return: grid_coordinates after applying self.rotations
    """

    rotations = -rotations if inv_transfo else rotations

    center = [math.ceil((x - 1) / 2) for x in im_shape]

    [i1, i2, i3] = np.meshgrid(2 * (np.arange(im_shape[0]) - center[0]) / im_shape[0],
                               2 * (np.arange(im_shape[1]) - center[1]) / im_shape[1],
                               2 * (np.arange(im_shape[2]) - center[2]) / im_shape[2], indexing='ij')

    # to rotate coordinate between -1 and 1 is not equivalent to compute it betawe -100 and 100 and divide by 100
    # special thanks to the matlab code from Gallichan  https://github.com/dgallichan/retroMoCoBox

    grid_coordinates = np.array([i1.flatten('F'), i2.flatten('F'), i3.flatten('F')])

    rotations = rotations.reshape([3] + list(im_shape))
    ix = (len(im_shape) + 1) * [slice(None)]
    ix[frequency_encoding_dim + 1] = 0  # dont need to rotate along freq encoding

    rotations = rotations[tuple(ix)].reshape(3, -1)
    rotation_matrices = np.apply_along_axis(create_rotation_matrix_3d, axis=0, arr=rotations).transpose(
        [-1, 0, 1])
    rotation_matrices = rotation_matrices.reshape(phase_encoding_shape + [3, 3])
    rotation_matrices = np.expand_dims(rotation_matrices, frequency_encoding_dim)

    rotation_matrices = np.tile(rotation_matrices,
                                reps=([im_shape[frequency_encoding_dim] if i == frequency_encoding_dim else 1
                                       for i in range(5)]))  # tile in freq encoding dimension

    # bug fix same order F as for grid_coordinates where it will be multiply to
    rotation_matrices = rotation_matrices.reshape([-1, 3, 3], order='F')

    # tile grid coordinates for vectorizing computation
    grid_coordinates_tiled = np.tile(grid_coordinates, [3, 1])
    grid_coordinates_tiled = grid_coordinates_tiled.reshape([3, -1], order='F').T
    rotation_matrices = rotation_matrices.reshape([-1, 3])  # reshape for matrix multiplication, so no order F

    new_grid_coords = (rotation_matrices * grid_coordinates_tiled).sum(axis=1)

    # reshape new grid coords back to 3 x nvoxels
    new_grid_coords = new_grid_coords.reshape([3, -1], order='F')

    # scale data between -pi and pi
    max_vals = [1, 1, 1]
    new_grid_coordinates_scaled = [(new_grid_coords[i, :] / max_vals[i]) * math.pi for i in [0, 1, 2]]
    #                                       range(new_grid_coords.shape[0])]
    # new_grid_coordinates_scaled = [np.asfortranarray(i) for i in new_grid_coordinates_scaled]
    # rrr why already flat ... ?

    # self.new_grid_coordinates_scaled = new_grid_coordinates_scaled
    # self.grid_coordinates = grid_coordinates
    # self.new_grid_coords = new_grid_coords
    return new_grid_coordinates_scaled, [grid_coordinates, new_grid_coords]


def _nufft(freq_domain_data, rotations, frequency_encoding_dim, phase_encoding_shape, im_shape, eps=1E-7, inv_transfo=False):
    """
    rotate coordinates and perform nufft
    :param freq_domain_data:
    :param eps: see finufft doc
    :param eps: precision of nufft
    :return: nufft of freq_domain_data after applying self.rotations
    """
    if not _finufft:
        raise ImportError('finufft not available')

    new_grid_coords = _rotate_coordinates(inv_transfo=inv_transfo, rotations=rotations, im_shape=im_shape,
                                          frequency_encoding_dim=frequency_encoding_dim, phase_encoding_shape=phase_encoding_shape)[0]
    # initialize array for nufft output
    f = np.zeros(im_shape, dtype=np.complex128, order='F')

    freq_domain_data_flat = np.asfortranarray(freq_domain_data.flatten(order='F'))

    finufft.nufft3d1(new_grid_coords[0], new_grid_coords[1], new_grid_coords[2], freq_domain_data_flat,
                     eps=eps, out=f, debug=0, spread_debug=0, spread_sort=2, fftw=0, modeord=0,
                     chkbnds=0, upsampfac=1.25)  # upsampling at 1.25 saves time at low precisions
    im_out = f.reshape(im_shape, order='F')

    return im_out


def demean_fitpars(fitpars_interp, original_image_fft, displacement_shift_strategy,
                   fast_dim=(0,2)):

    to_substract = np.array([0, 0, 0, 0, 0, 0])

    if '1D' in displacement_shift_strategy:
        fitpars = fitpars_interp
        #ok only 1D fftp
        print(f'averaging TF coeficient on dim {fast_dim} for disp corection {displacement_shift_strategy} ')
        #coef_shaw = np.sqrt( np.sum(abs(original_image_fft**2), axis=(0,2)) ) ;
        #should be equivalent if fft is done from real image, but not if the phase is acquired, CF Todd 2015 "Prospective motion correction of 3D echo-planar imaging data for functional MRI using optical tracking"
        if displacement_shift_strategy == '1D_wTF':
            coef_shaw = np.abs( np.sqrt(np.sum( original_image_fft * np.conjugate(original_image_fft), axis=fast_dim )));
        if displacement_shift_strategy == '1D_wTF2':
            coef_shaw = np.abs( np.sum( original_image_fft * np.conjugate(original_image_fft), axis=fast_dim ));


        if fitpars.shape[1] != coef_shaw.shape[0] :
            #just interpolate end to end. at image slowest dimention size
            fitpars = _interpolate_fitpars(fitpars, len_output=coef_shaw.shape[0])
            #print(f'interp fitpar for wcoef new shape {fitpars.shape}')

        to_substract = np.zeros(6)
        for i in range(0,6):
            to_substract[i] = np.sum(fitpars[i,:] * coef_shaw) / np.sum(coef_shaw)
            fitpars[i,:] = fitpars[i,:] - to_substract[i]

        return fitpars, to_substract  #note the 1D fitpar, may have been interpolated to phase dim but should not matter for the rest

    if displacement_shift_strategy == "demean":
        o_shape = original_image_fft.shape
        tfi = np.abs(original_image_fft) #np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(original_image))).astype(np.complex128))
        ss = tfi
        # Almost identical   ss = np.sqrt(tfi * np.conjugate(tfi))
        ff = fitpars_interp
        to_substract = np.zeros(6)
        for i in range(0, 6):
            ffi = ff[i].reshape(-1)
            ssi = ss.reshape(-1)
            to_substract[i] = np.sum(ffi * ssi) / np.sum(ssi)

        to_substract_tile = np.tile(to_substract[..., np.newaxis, np.newaxis, np.newaxis],
                                    (1, o_shape[0], o_shape[1], o_shape[2]))
        fitpars_interp = np.subtract(fitpars_interp, to_substract_tile)

    elif displacement_shift_strategy == "demean_half":
        nb_pts_around = 31
        print('RR demean_center around {}'.format(nb_pts_around))
        # let's take the weight from the tf, but only in the center (+- 11 pts)
        o_shape = original_image_fft.shape
        tfi = np.abs(original_image_fft) #np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(original_image))).astype(np.complex128))
        ss = tfi
        center = [int(round(dd / 2)) for dd in o_shape]
        center_half = [int(round(dd / 2)) for dd in center]
        #fov_inf = center_half
        #fov_sup = center + center_half
        fov_inf = [dd - nb_pts_around for dd in center];
        fov_sup = [dd + nb_pts_around for dd in center];

        ss[0:fov_inf[0], :, :] = 0
        ss[:, 0:fov_inf[1], :] = 0
        ss[:, :, 0:fov_inf[2]] = 0
        ss[fov_sup[0]:, :, :] = 0
        ss[:, fov_sup[1]:, :] = 0
        ss[:, :, fov_sup[2]:] = 0

        ff = fitpars_interp

        to_substract = np.zeros(6)
        for i in range(0, 6):
            ffi = ff[i].reshape(-1)
            ssi = ss.reshape(-1)
            to_substract[i] = np.sum(ffi * ssi) / np.sum(ssi)

        to_substract_tile = np.tile(to_substract[..., np.newaxis, np.newaxis, np.newaxis],
                                    (1, o_shape[0], o_shape[1], o_shape[2]))
        fitpars_interp = np.subtract(fitpars_interp, to_substract_tile)

    elif displacement_shift_strategy == "center_zero":
        dim = fitpars_interp.shape
        center = [int(round(dd / 2)) for dd in dim]
        to_substract = fitpars_interp[:, center[1], center[2], center[3]]
        to_substract_tile = np.tile(to_substract[..., np.newaxis, np.newaxis, np.newaxis], (1, dim[1], dim[2], dim[3]))
        fitpars_interp = fitpars_interp - to_substract_tile

    return fitpars_interp, to_substract

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
        warnings.warn("Expected motion parameters to be of shape (6, N), found {}. Setting motion to None".format(fpars.shape))
        fpars = None

    if np.any(np.isnan(fpars)) :
        #assume it is the last column, as can happen if the the csv line ends with ,
        fpars = fpars[:, :-1]
        if np.any(np.isnan(fpars)):
            warnings.warn('There is still NaN in the fitpar, it will crash the nufft')
    return fpars
