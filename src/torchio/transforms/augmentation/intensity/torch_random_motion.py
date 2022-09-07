import math
import torch
import warnings
import numpy as np
import pandas as pd
from scipy.interpolate import pchip_interpolate
import itertools
from typing import Dict
from torchkbnufft import AdjKbNufft
from torchkbnufft.math import complex_mult, imag_exp, absolute
from torchio.transforms.augmentation.random_transform import RandomTransform
import torch.nn.functional as F


class TorchRandomMotionFromTimeCourse(RandomTransform):

    def __init__(self, nT=200, maxDisp=(2,5), maxRot=(2,5), noiseBasePars=(5,15),
                 swallowFrequency=(0,5), swallowMagnitude=(2,6),
                 suddenFrequency=(0,5), suddenMagnitude=(2,6),
                 fitpars=None, read_func=lambda x: pd.read_csv(x, header=None).values,
                 displacement_shift=1, freq_encoding_dim=[0], tr=2.3, es=4E-3,
                 nufft=True,  oversampling_pct=0.3, proba_to_augment: float = 1,
                 verbose=False, preserve_center_pct=0, correct_motion=False, res_dir=None, cuda=False, metrics: Dict = None):
        """
        parameters to simulate 3 types of displacement random noise swllow or sudden mouvement
        :param nT (int): number of points of the time course
        :param maxDisp (float, float): (min, max) value of displacement in the perlin noise (useless if noiseBasePars is 0)
        :param maxRot (float, float): (min, max) value of rotation in the perlin noise (useless if noiseBasePars is 0)
        :param noiseBasePars (float, float): (min, max) base value of the perlin noise to generate for the time course
        optional (float, float, float) where the third is the probability to performe this type of noise
        :param swallowFrequency (int, int): (min, max) number of swallowing movements to generate in the time course
        optional (float, float, float) where the third is the probability to performe this type of noise
        :param swallowMagnitude (float, float): (min, max) magnitude of the swallowing movements to generate
        :param suddenFrequency (int, int): (min, max) number of sudden movements to generate in the time course
        optional (float, float, float) where the third is the probability to performe this type of noise
        :param suddenMagnitude (float, float): (min, max) magnitude of the sudden movements to generate
        if fitpars is not None previous parameter are not used
        :param fitpars : movement parameters to use (if specified, will be applied as such, no movement is simulated)
        :param read_func (function): if fitpars is a string, function to use to read the data. Must return an array of shape (6, nT)        :param displacement_shift (bool): whether or not to substract the time course by the values of the center of the kspace
        :param freq_encoding_dim (tuple of ints): potential frequency encoding dims to use (one of them is randomly chosen)
        :param tr (float): repetition time of the data acquisition (used for interpolating the time course movement)
        :param es (float): echo spacing time of the data acquisition (used for interpolating the time course movement)
        :param nufft (bool): whether or not to apply nufft (if false, no rotation is aaplyed ! )
        :param oversampling_pct (float): percentage with which the data will be oversampled in the image domain prior to applying the motion
        :param verbose (bool): verbose
        Note currently on freq_encoding_dim=0 give the same ringing direction for rotation and translation, dim 1 and 2 are not coherent
        Note fot suddenFrequency and swallowFrequency min max must differ and the max is never achieved, so to have 0 put (0,1)
        """
        super(TorchRandomMotionFromTimeCourse, self).__init__(verbose=verbose, p=proba_to_augment, metrics=metrics)
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
        self.displacement_shift = displacement_shift
        self.preserve_center_frequency_pct = preserve_center_pct
        self.freq_encoding_choice = freq_encoding_dim
        self.frequency_encoding_dim = np.random.choice(self.freq_encoding_choice)
        self.read_func = read_func
        self.displacement_substract = np.zeros(6)
        if fitpars is None:
            self.fitpars = None
            self.simulate_displacement = True
        else:
            self.fitpars = self.read_fitpars(fitpars)
            self.simulate_displacement = False
        self.nufft = nufft
        self.cuda = cuda
        self.oversampling_pct = oversampling_pct
        self.proba_to_augment = proba_to_augment
        self.preserve_center_pct = preserve_center_pct
        self.correct_motion = correct_motion
        self.res_dir = res_dir
        self.nb_saved = 0

    def apply_transform(self, sample):
        parameters_motion = {}
        for image_name, image_dict in sample.get_images_dict().items():

            do_it = np.random.uniform() <= self.proba_to_augment

            parameters_motion['simu_param'] = dict(noisPar=0.0, maxDisp=0.0, maxRot=0.0, swallowFrequency=0.0,
            swallowMagnitude=[0.0,0.0], suddenFrequency=0.0, suddenMagnitude=[0.0,0.0])

            if not do_it:
                sample[image_name]['motion'] = False
                return sample
            else:
                sample[image_name]['motion'] = True

            #image_data = np.squeeze(image_dict['data'])[..., np.newaxis, np.newaxis]
            image_data = image_dict['data'].unsqueeze(1)
            original_image = torch.squeeze(image_data)
            if self.oversampling_pct > 0.0:
                original_image_shape = original_image.shape
                original_image = self._oversample(original_image, self.oversampling_pct)

            self._calc_dimensions(original_image.shape)

            if self.simulate_displacement:
                fitpars_interp = self._simulate_random_trajectory()
                parameters_motion['simu_param'] = self.simu_param
            else:
                if self.fitpars.ndim==4:
                    fitpars_interp = self.fitpars
                else:
                    fitpars_interp = self._interpolate_space_timing(self.fitpars)
                    fitpars_interp = self._tile_params_to_volume_dims(fitpars_interp)

            if self.displacement_shift > 1:
                fitpars_interp = self.demean_fitpar(fitpars_interp, original_image)[0]

            fitpars_vox = fitpars_interp.reshape((6, -1))
            self.translations, self.rotations = fitpars_vox[:3], np.radians(fitpars_vox[3:])
            self.translations, self.rotations = torch.from_numpy(self.translations), torch.from_numpy(self.rotations)
            if self.cuda:
                original_image = original_image.cuda()
            # fft
            im_freq_domain = self._fft_im(original_image)
            translated_im_freq_domain = self._translate_freq_domain(freq_domain=im_freq_domain)
            # iNufft for rotations
            if self.nufft:
                corrupted_im = self._nufft(translated_im_freq_domain)
                corrupted_im = corrupted_im/(corrupted_im.numel()/2)  # normalize
            else:
                corrupted_im = self._ifft_im(translated_im_freq_domain)

            if self.correct_motion:
                corrected_im = self.do_correct_motion(corrupted_im)
                image_dict["data_cor"] = corrected_im[np.newaxis, ...]
                image_dict['data_cor'] = torch.from_numpy(image_dict['data_cor']).float()

            # magnitude
            corrupted_im = absolute(corrupted_im.squeeze(), -1)

            if self.oversampling_pct > 0.0:
                corrupted_im = self.crop_volume(corrupted_im, original_image_shape)

            image_dict["data"] = corrupted_im.T.float()
            #image_dict['data'] = torch.from_numpy(image_dict['data']).float()

            #add extra field to follow what have been done
            #sample[image_name]['fit_pars'] = self.fitpars
            #sample[image_name]['fit_pars_interp'] = self.fitpars_interp
            if self.res_dir is not None:
                self.save_to_dir(image_dict)

        metrics = dict()
        if self.fitpars.ndim == 2:
            metrics['mean_DispP'] = calculate_mean_Disp_P(self.fitpars)
            metrics['rmse_Disp'] = calculate_mean_RMSE_displacment(self.fitpars)
        metrics['mean_DispP_iterp'] = calculate_mean_Disp_P(fitpars_interp)
        metrics['rmse_Disp_iterp'] = calculate_mean_RMSE_displacment(fitpars_interp)

        ff_interp, to_substract = self.demean_fitpar(fitpars_interp, original_image)
        metrics['TFsubstract'] = to_substract
        metrics['rmse_DispTF'] = calculate_mean_RMSE_displacment(ff_interp, original_image)

        parameters_motion['metrics_motion'] = metrics

        #sample.add_transform(self, parameters_motion)
        return sample
        #output type is double, TODO where to cast in Float ?

    def _fft_im(self, image, signal_ndim=3):
        output = self._fftshift2d(torch.rfft(image, signal_ndim=signal_ndim, onesided=False, normalized=True))
        return output.squeeze()

    def _ifft_im(self, freq_domain, signal_ndim=3):
        output = torch.ifft(self._ifftshift2d(freq_domain), signal_ndim=signal_ndim)
        return output.squeeze()

    def roll_n(self, X, axis, n):
        '''
        from https://github.com/tomrunia/PyTorchSteerablePyramid
        '''
        f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
        b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
        front = X[f_idx]
        back = X[b_idx]
        return torch.cat([back, front], axis)

    def _ifftshift2d(self, x):
        '''
        from https://github.com/tomrunia/PyTorchSteerablePyramid
        '''
        real, imag = torch.unbind(x, -1)
        for dim in range(len(real.size()) - 1, -1, -1):
            real = self.roll_n(real, axis=dim, n=real.size(dim) // 2)
            imag = self.roll_n(imag, axis=dim, n=imag.size(dim) // 2)
        return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

    def _fftshift2d(self, x):
        '''
         from https://github.com/tomrunia/PyTorchSteerablePyramid
        '''
        real, imag = torch.unbind(x, -1)
        for dim in range(0, len(real.size())):
            n_shift = real.size(dim) // 2
            if real.size(dim) % 2 != 0:
                n_shift += 1  # for odd-sized images
            real = self.roll_n(real, axis=dim, n=n_shift)
            imag = self.roll_n(imag, axis=dim, n=n_shift)
        return torch.stack((real, imag), -1)

    @staticmethod
    def _oversample(data, perc_oversampling=.10):
        """
        Oversamples data with a zero padding. Adds perc_oversampling percentage values
        :param data (ndarray): array to pad
        :param perc_oversampling (float): percentage of oversampling to add to data (based on its current shape)
        :return oversampled version of the data:
        """
        data_shape = list(data.shape)
        to_pad = np.ceil(np.asarray(data_shape) * perc_oversampling / 2) * 2
        # to force an even number if odd, this will shift the volume when croping
        # print("Pading at {}".format(to_pad))
        left_pad = np.floor(to_pad / 2).astype(int)
        right_pad = np.ceil(to_pad / 2).astype(int)
        to_pad = [x for x in itertools.chain.from_iterable(itertools.zip_longest(left_pad, right_pad)) ]
        return F.pad(data, to_pad)

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


    def do_correct_motion(self, image):
        im_freq_domain = self._fft_im(image)
        # print('translation')
        translated_im_freq_domain = self._translate_freq_domain(freq_domain=im_freq_domain, inv_transfo=True)
        # print('rotaion')
        # iNufft for rotations
        if self.nufft:
            corrected_im = self._nufft(translated_im_freq_domain, inv_transfo=True)
            corrected_im = corrected_im / corrected_im.size  # normalize
        else:
            corrected_im = self._ifft_im(translated_im_freq_domain)
        # magnitude
        corrected_im = abs(corrected_im)

        return corrected_im


    @staticmethod
    def get_params():
        pass

    def read_fitpars(self, fitpars):
        '''
        :param fitpars:
        '''
        fpars = None
        if isinstance(fitpars, np.ndarray):
            fpars = fitpars
        elif isinstance(fitpars, list):
            fpars = np.asarray(fitpars)
        elif isinstance(fitpars, str):
            try:
                fpars = self.read_func(fitpars)
            except:
                warnings.warn("Could not read {} with given function. Motion parameters are set to None".format(fpars))
                fpars = None
        if fpars.shape[0] != 6:
            warnings.warn("Given motion parameters has {} on the first dimension. "
                          "Expected 6 (3 translations and 3 rotations). Setting motions to None".format(fpars.shape[0]))
            fpars = None
        elif len(fpars.shape) != 2:
            warnings.warn("Expected motion parameters to be of shape (6, N), found {}. Setting motions to None".format(fpars.shape))
            fpars = None

        if self.displacement_shift > 0:
            to_substract = fpars[:, int(round(self.nT / 2))]
            fpars = np.subtract(fpars, to_substract[..., np.newaxis])
            #print('removing to fit_pars {}'.format(to_substract))
            self.displacement_substract = to_substract

        #print(fpars.shape)
        if np.any(np.isnan(fpars)) :
            #assume it is the last column, as can happen if the the csv line ends with ,
            fpars = fpars[:, :-1]
            if np.any(np.isnan(fpars)):
                warnings.warn('There is still NaN in the fitpar, it will crash the nufft')
        self.nT = fpars.shape[1]

        return fpars

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
        self.phase_encoding_shape = im_shape #[ im_shape[pp-1] for pp in pe_dims]
        self.num_phase_encoding_steps = self.phase_encoding_shape[0] * self.phase_encoding_shape[1]
        self.frequency_encoding_dim = len(self.im_shape) - 1 if self.frequency_encoding_dim == -1 \
            else self.frequency_encoding_dim

    # no more used
    def _center_k_indices_to_preserve(self):
        """get center k indices of freq domain"""
        mid_pts = [int(math.ceil(x / 2)) for x in self.phase_encoding_shape]
        num_pts_preserve = [math.ceil(self.preserve_center_frequency_pct * x) for x in self.phase_encoding_shape]
        ind_to_remove = {val + 1: slice(mid_pts[i] - num_pts_preserve[i], mid_pts[i] + num_pts_preserve[i])
                         for i, val in enumerate(self.phase_encoding_dims)}
        ix_to_remove = [ind_to_remove.get(dim, slice(None)) for dim in range(4)]
        return ix_to_remove

    @staticmethod
    def perlinNoise1D(npts, weights):
        if not isinstance(weights, list):
            weights = range(int(round(weights)))
            weights = np.power([2] * len(weights), weights)

        n = len(weights)
        xvals = np.linspace(0, 1, npts)
        total = np.zeros((npts, 1))

        for i in range(n):
            frequency = 2**i
            this_npts = round(npts / frequency)

            if this_npts > 1:
                total += weights[i] * pchip_interpolate(np.linspace(0, 1, this_npts), np.random.random((this_npts, 1)),
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
        maxDisp = np.random.uniform(low=self.maxDisp[0], high=self.maxDisp[1])
        maxRot = np.random.uniform(low=self.maxRot[0], high=self.maxRot[1])
        noiseBasePars = np.random.uniform(low=self.noiseBasePars[0], high=self.noiseBasePars[1])
        swallowFrequency = np.random.randint(low=self.swallowFrequency[0], high=self.swallowFrequency[1])
        swallowMagnitude = [np.random.uniform(low=self.swallowMagnitude[0], high=self.swallowMagnitude[1]),
                            np.random.uniform(low=self.swallowMagnitude[0], high=self.swallowMagnitude[1])]
        suddenFrequency = np.random.randint(low=self.suddenFrequency[0], high=self.suddenFrequency[1])
        suddenMagnitude = [np.random.uniform(low=self.suddenMagnitude[0], high=self.suddenMagnitude[1]),
                            np.random.uniform(low=self.suddenMagnitude[0], high=self.suddenMagnitude[1])]

        #prba to include the different type of noise
        proba_noiseBase = self.noiseBasePars[2] if len(self.noiseBasePars) == 3 else 1
        proba_swallow = self.swallowFrequency[2] if len(self.swallowFrequency) == 3 else 1
        proba_sudden = self.suddenFrequency[2] if len(self.suddenFrequency) == 3 else 1
        do_noise, do_swallow, do_sudden = False, False, False
        while (do_noise or do_swallow or do_sudden) is False: #at least one is not false
            do_noise = np.random.uniform() <= proba_noiseBase
            do_swallow = np.random.uniform() <= proba_swallow
            do_sudden = np.random.uniform() <= proba_sudden
        if do_noise is False: noiseBasePars = 0
        if do_swallow is False: swallowFrequency = 0
        if do_sudden is False: suddenFrequency = 0

        #print('simulate FITpars')
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
                rand_shifts = int(round(np.random.rand() * self.nT))
                rolled = np.roll(swallowTraceBase, rand_shifts, axis=0)
                swallowTrace += rolled

            fitpars[2, :] += swallowMagnitude[0] * swallowTrace
            fitpars[3, :] += swallowMagnitude[1] * swallowTrace

        # add in random sudden movements in any direction
        if suddenFrequency > 0:
            suddenTrace = np.zeros(fitpars.shape)

            for i in range(suddenFrequency):
                iT_sudden = int(np.ceil(np.random.rand() * self.nT))
                to_add = np.asarray([suddenMagnitude[0] * (2 * np.random.random(3) - 1),
                                     suddenMagnitude[1] * (2 * np.random.random(3) - 1)]).reshape((-1, 1))
                suddenTrace[:, iT_sudden:] = np.add(suddenTrace[:, iT_sudden:], to_add)

            fitpars += suddenTrace

        if self.displacement_shift > 0:
            to_substract = fitpars[:, int(round(self.nT / 2))]
            fitpars = np.subtract(fitpars, to_substract[..., np.newaxis])
            self.displacement_substract = to_substract

        if self.preserve_center_frequency_pct:
            center = np.int(np.floor( fitpars.shape[1] /2 ))
            nbpts =  np.int(np.floor(fitpars.shape[1] * self.preserve_center_frequency_pct/2))
            fitpars[:, center-nbpts:center+nbpts] = 0

        self.fitpars = fitpars
        #print(f' in _simul_motionfitpar shape fitpars {fitpars.shape}')
        simu_param = dict(noisPar=noiseBasePars,maxDisp=maxDisp,maxRot=maxRot,
                          swallowFrequency=swallowFrequency, swallowMagnitudeT=swallowMagnitude[0], swallowMagnitudeR=swallowMagnitude[1],
                          suddenFrequency=suddenFrequency,suddenMagnitudeT=suddenMagnitude[0], suddenMagnitude=suddenMagnitude[1])
        self.simu_param = simu_param
        fitpars = self._interpolate_space_timing(fitpars)
        fitpars = self._tile_params_to_volume_dims(fitpars)

        return fitpars

    def _interpolate_space_timing_1D(self, fitpars):
        n_phase= self.phase_encoding_shape[0]
        nT = self.nT
        # Time steps
        mg_total = np.linspace(0,1,n_phase)
        # Equidistant time spacing
        teq = np.linspace(0, 1, nT)
        # Actual interpolation
        fitpars_interp = np.asarray([np.interp(mg_total, teq, params) for params in fitpars])
        # Reshaping to phase encoding dimensions
        self.fitpars_interp = fitpars_interp
        # Add missing dimension
        fitpars_interp = np.expand_dims(fitpars_interp, axis= [self.frequency_encoding_dim + 1,self.phase_encoding_dims[1] + 1])
        return fitpars_interp

    def _interpolate_space_timing(self, fitpars):
        n_phase, n_slice = self.phase_encoding_shape[0], self.phase_encoding_shape[1]
        # Time steps
        t_steps = n_phase * self.tr
        # Echo spacing dimension
        dim_es = np.cumsum(self.es * np.ones(n_slice)) - self.es
        dim_tr = np.cumsum(self.tr * np.ones(n_phase)) - self.tr
        # Build grid
        mg_es, mg_tr = np.meshgrid(*[dim_es, dim_tr])
        mg_total = mg_es + mg_tr  # MP-rage timing
        # Flatten grid and sort values
        mg_total = np.sort(mg_total.reshape(-1))
        # Equidistant time spacing
        teq = np.linspace(0, t_steps, self.nT)
        # Actual interpolation
        fitpars_interp = np.asarray([np.interp(mg_total, teq, params) for params in fitpars])
        # Reshaping to phase encoding dimensions
        fitpars_interp = fitpars_interp.reshape([6] + self.phase_encoding_shape)
        self.fitpars_interp = fitpars_interp
        # Add missing dimension
        fitpars_interp = np.expand_dims(fitpars_interp, axis=self.frequency_encoding_dim + 1)
        return fitpars_interp

    def _tile_params_to_volume_dims(self, params_to_reshape):
        target_shape = [6] + self.im_shape
        data_shape = params_to_reshape.shape
        tiles = np.floor_divide(target_shape, data_shape, dtype=int)
        return np.tile(params_to_reshape, reps=tiles)

    def _translate_freq_domain(self, freq_domain, inv_transfo=False):
        """
        image domain translation by adding phase shifts in frequency domain
        :param freq_domain - frequency domain data 3d numpy array:
        :return frequency domain array with phase shifts added according to self.translations:
        """
        translations = -self.translations if inv_transfo else self.translations

        lin_spaces = [torch.linspace(-.5, .5, x) for x in freq_domain.shape[:-1]]
        meshgrids = torch.meshgrid(*lin_spaces)
        grid_coords = torch.stack([mg.flatten() for mg in meshgrids], 0)
        phase_shift = torch.mul(grid_coords, translations).sum(0)
        exp_phase_shift = imag_exp(-2 * math.pi * phase_shift, -1)
        if self.cuda:
            exp_phase_shift = exp_phase_shift.cuda()
        freq_domain_translated = complex_mult(exp_phase_shift, freq_domain.reshape((-1, 2)), -1)
        return freq_domain_translated.reshape(freq_domain.shape)

    def _rotate_coordinates(self, inv_transfo=False):
        """
        :return: grid_coordinates after applying self.rotations
        """

        rotations = -self.rotations if inv_transfo else self.rotations

        center = [math.ceil((x - 1) / 2) for x in self.im_shape]

        [i1, i2, i3] = np.meshgrid(2*(np.arange(self.im_shape[0]) - center[0])/self.im_shape[0],
                                   2*(np.arange(self.im_shape[1]) - center[1])/self.im_shape[1],
                                   2*(np.arange(self.im_shape[2]) - center[2])/self.im_shape[2], indexing='ij')

        #to rotate coordinate between -1 and 1 is not equivalent to compute it betawe -100 and 100 and divide by 100
        #special thanks to the matlab code from Gallichan  https://github.com/dgallichan/retroMoCoBox

        grid_coordinates = np.array([i1.flatten('F'), i2.flatten('F'), i3.flatten('F')])

        #print('rotation size is {}'.format(self.rotations.shape))

        rotations = rotations.reshape([3] + self.im_shape)
        ix = (len(self.im_shape) + 1) * [slice(None)]
        ix[self.frequency_encoding_dim + 1] = 0  # dont need to rotate along freq encoding

        rotations = rotations[tuple(ix)].reshape(3, -1)
        rotation_matrices = np.apply_along_axis(create_rotation_matrix_3d, axis=0, arr=rotations).transpose([-1, 0, 1])
        rotation_matrices = rotation_matrices.reshape(self.phase_encoding_shape + [3, 3])
        rotation_matrices = np.expand_dims(rotation_matrices, self.frequency_encoding_dim)

        rotation_matrices = np.tile(rotation_matrices,
                                    reps=([self.im_shape[ self.frequency_encoding_dim] if i == self.frequency_encoding_dim else 1
                                           for i in range(5)]))  # tile in freq encoding dimension

        #bug fix same order F as for grid_coordinates where it will be multiply to
        rotation_matrices = rotation_matrices.reshape([-1, 3, 3], order='F')

        # tile grid coordinates for vectorizing computation
        grid_coordinates_tiled = np.tile(grid_coordinates, [3, 1])
        grid_coordinates_tiled = grid_coordinates_tiled.reshape([3, -1], order='F').T
        rotation_matrices = rotation_matrices.reshape([-1, 3]) #reshape for matrix multiplication, so no order F

        new_grid_coords = (rotation_matrices * grid_coordinates_tiled).sum(axis=1)

        # reshape new grid coords back to 3 x nvoxels
        new_grid_coords = new_grid_coords.reshape([3, -1], order='F')

        # scale data between -pi and pi
        max_vals = [1, 1, 1]
        new_grid_coordinates_scaled = [(new_grid_coords[i, :] / max_vals[i]) * math.pi for i in [0, 1, 2]]
#                                       range(new_grid_coords.shape[0])]
        #new_grid_coordinates_scaled = [np.asfortranarray(i) for i in new_grid_coordinates_scaled]
        #rrr why already flat ... ?

        #self.new_grid_coordinates_scaled = new_grid_coordinates_scaled
        #self.grid_coordinates = grid_coordinates
        #self.new_grid_coords = new_grid_coords
        return new_grid_coordinates_scaled, [grid_coordinates, new_grid_coords]

    def _nufft(self, freq_domain_data, iflag=1, eps=1E-7,  inv_transfo=False):
        """
        rotate coordinates and perform nufft
        :param freq_domain_data:
        :param iflag/eps: see finufftpy doc
        :param eps: precision of nufft
        :return: nufft of freq_domain_data after applying self.rotations
        """

        new_grid_coords = torch.from_numpy(np.asarray(self._rotate_coordinates(inv_transfo=inv_transfo)[0])).unsqueeze(0)
        adj_nufft = AdjKbNufft(im_size=freq_domain_data.shape[:-1], n_shift=(0, 0, 0))
        freq_domain_data = freq_domain_data.permute(3, 0, 1, 2).view((1, 1, 2, -1))
        if self.cuda:
            adj_nufft = adj_nufft.cuda()
            new_grid_coords = new_grid_coords.cuda()
        im_out = adj_nufft(freq_domain_data, new_grid_coords)
        im_out = torch.stack(torch.unbind(im_out.squeeze(), 0), -1)
        del adj_nufft, new_grid_coords
        return im_out

    def demean_fitpar(self,fitpars_interp, original_image):

        o_shape = original_image.shape
        #original_image = np.moveaxis(original_image.numpy(), 1, 2)
        tfi = np.abs(np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(original_image))).astype(np.complex128))
        #tfi = np.real((np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(original_image)))).astype(np.complex128)) does not work (make a shift)
        #ss = np.tile(tfi, (6, 1, 1, 1))
        ss = tfi #np.moveaxis(tfi, 1,2)  # to have the same order as previously

        # mean around kspace center
        #ss_center = np.zeros(ss.shape)
        #nbpts = 2
        #center = [np.floor((x - 1) / 2).astype(int) for x in o_shape]
        #ss_center[center[0]-nbpts:center[0]+nbpts,center[1]-nbpts:center[1]+nbpts,center[2]-nbpts:center[2]+nbpts] =  ss[center[0]-nbpts:center[0]+nbpts,center[1]-nbpts:center[1]+nbpts,center[2]-nbpts:center[2]+nbpts]
        #ss = ss_center

        ff = fitpars_interp
        #ff = np.moveaxis(fitpars_interp, 2, 3)  # because y is slowest axis  uselull to plot but not necessary if ff and ss are coherent

        to_substract = np.zeros(6)
        for i in range(0, 6):
            ffi = ff[i].reshape(-1)
            ssi = ss.reshape(-1)
            #xx = np.argwhere(ssi > (np.max(ssi) * 0.001)).reshape(-1)
            #to_substract[i] = np.sum(ffi[xx] * ssi[xx]) / np.sum(ssi[xx])
            to_substract[i] = np.sum(ffi * ssi) / np.sum(ssi)
        #print('Removing TF mean {} '.format(to_substract))

        #print('Removing {} OR {}'.format(to_substract, (to_substract+self.displacement_substract)))
        to_substract_tile = np.tile(to_substract[..., np.newaxis, np.newaxis, np.newaxis],
                               (1, o_shape[0], o_shape[1], o_shape[2]))
        fitpars_interp = np.subtract(fitpars_interp, to_substract_tile)
        return fitpars_interp, to_substract



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

    mat2 = np.array([[math.cos(angles[1]), 0., -math.sin(angles[1])],
                     [0., 1., 0.],
                     [math.sin(angles[1]), 0., math.cos(angles[1])]],
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


def calculate_mean_Disp_P(motion_params):
    """
    Same as previous, but without taking the diff between frame
    """
    translations = np.transpose(np.abs(motion_params[0:3, :]))
    rotations = np.transpose(np.abs(motion_params[3:6, :]))
    fd = np.mean(translations, axis=1) + (50 * np.pi / 180) * np.mean(rotations, axis=1)

    return np.mean(fd)


def calculate_mean_FD_J(motion_params):
    """
    Method to calculate framewise displacement as per Jenkinson et al. 2002
    """
    pm = np.zeros((motion_params.shape[1],16))
    for tt in range(motion_params.shape[1]):
        P = np.hstack((motion_params[:, tt], np.array([1, 1, 1, 0, 0, 0])))
        pm[tt,:] = spm_matrix(P, order=0).reshape(-1)

    # The default radius (as in FSL) of a sphere represents the brain
    rmax = 80.0

    T_rb_prev = pm[0].reshape(4, 4)

    fd = np.zeros(pm.shape[0])

    for i in range(1, pm.shape[0]):
        T_rb = pm[i].reshape(4, 4)
        M = np.dot(T_rb, np.linalg.inv(T_rb_prev)) - np.eye(4)
        A = M[0:3, 0:3]
        b = M[0:3, 3]
        fd[i] = np.sqrt( (rmax * rmax / 5) * np.trace(np.dot(A.T, A)) + np.dot(b.T, b) )
        T_rb_prev = T_rb

    return np.mean(fd)


def calculate_mean_RMSE_displacment(fit_pars, image=None):
    """
    very crude approximation where rotation in degree and translation are average ...
    """
    if image is None:
        r1 = np.sqrt(np.sum(fit_pars[0:3] * fit_pars[0:3], axis=0))
        rms1 = np.sqrt(np.mean(r1 * r1))
        r2 = np.sqrt(np.sum(fit_pars[3:6] * fit_pars[3:6], axis=0))
        rms2 = np.sqrt(np.mean(r2 * r2))
        res = (rms1 + rms2) / 2
    else:
        tfi = np.abs(np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(image))).astype(np.complex128))
        ss = tfi
        ff = fit_pars

        to_substract = np.zeros(6)
        for i in range(0, 6):
            ffi = ff[i].reshape(-1)
            ssi = ss.reshape(-1)
            # xx = np.argwhere(ssi > (np.max(ssi) * 0.001)).reshape(-1)
            # to_substract[i] = np.sum(ffi[xx] * ssi[xx]) / np.sum(ssi[xx])
            to_substract[i] = np.sqrt( np.sum(ffi * ffi * ssi) / np.sum(ssi) )
        res = np.mean(to_substract)

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

    [P[3], P[4], P[5]] = [P[3]*180/np.pi, P[4]*180/np.pi, P[5]*180/np.pi] #degre to radian

    T = np.array([[1,0,0,P[0]],[0,1,0,P[1]],[0,0,1,P[2]],[0,0,0,1]])
    R1 =  np.array([[1,0,0,0],
                    [0,np.cos(P[3]),np.sin(P[3]),0],#sing change compare to spm because neuro versus radio ?
                    [0,-np.sin(P[3]),np.cos(P[3]),0],
                    [0,0,0,1]])
    R2 =  np.array([[np.cos(P[4]),0,-np.sin(P[4]),0],
                    [0,1,0,0],
                    [np.sin(P[4]),0,np.cos(P[4]),0],
                    [0,0,0,1]])
    R3 =  np.array([[np.cos(P[5]),np.sin(P[5]),0,0],  #sing change compare to spm because neuro versus radio ?
                    [-np.sin(P[5]),np.cos(P[5]),0,0],
                    [0,0,1,0],
                    [0,0,0,1]])

    #R = R1.dot(R2.dot(R3))
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
