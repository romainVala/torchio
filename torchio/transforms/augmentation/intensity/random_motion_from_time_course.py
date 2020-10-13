import math
import torch
import warnings
import numpy as np
from typing import Dict, Tuple, List, Union, Optional
from scipy.interpolate import pchip_interpolate
try:
    import finufft
    _finufft = True
except ImportError:
    _finufft = False

from .. import RandomTransform


class RandomMotionFromTimeCourse(RandomTransform):

    def __init__(self, nT: int = 200, maxDisp: Tuple[float, float] = (2,5), maxRot: Tuple[float, float] = (2,5),
                 noiseBasePars: Tuple[float, float] = (5,15), swallowFrequency: Tuple[float, float] = (0,5),
                 swallowMagnitude: Tuple[float, float] = (2,6), suddenFrequency: Tuple[int, int] = (0,5),
                 suddenMagnitude: Tuple[float, float] = (2,6), fitpars: Union[List, np.ndarray] = None,
                 displacement_shift_strategy: str = None, freq_encoding_dim: List = [0], tr: float = 2.3, es: float = 4E-3,
                 nufft: bool = True,  oversampling_pct: float = 0.3, p: float = 1,
                 preserve_center_frequency_pct: float = 0, correct_motion: bool = False,  metrics: Dict = None,
                 keys: Optional[List[str]] = None, res_dir: str = None):
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
        :param displacement_shift (bool): whether or not to substract the time course by the values of the center of the kspace
        :param freq_encoding_dim (tuple of ints): potential frequency encoding dims to use (one of them is randomly chosen)
        :param tr (float): repetition time of the data acquisition (used for interpolating the time course movement)
        :param es (float): echo spacing time of the data acquisition (used for interpolating the time course movement)
        :param nufft (bool): whether or not to apply nufft (if false, no rotation is aaplyed ! )
        :param oversampling_pct (float): percentage with which the data will be oversampled in the image domain prior to applying the motion
        :param verbose (bool): verbose
        Note currently on freq_encoding_dim=0 give the same ringing direction for rotation and translation, dim 1 and 2 are not coherent
        Note fot suddenFrequency and swallowFrequency min max must differ and the max is never achieved, so to have 0 put (0,1)
        """
        super(RandomMotionFromTimeCourse, self).__init__(p=p, metrics=metrics, keys=keys)
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
        self.displacement_shift_strategy = displacement_shift_strategy
        self.preserve_center_frequency_pct = preserve_center_frequency_pct
        self.freq_encoding_choice = freq_encoding_dim
        self.frequency_encoding_dim = self._rand_choice(self.freq_encoding_choice)
        if fitpars is None:
            self.fitpars = None
            self.simulate_displacement = True
        else:
            self.fitpars = self.read_fitpars(fitpars)
            self.simulate_displacement = False
        self.nufft = nufft
        self.oversampling_pct = oversampling_pct
        if (not _finufft) and nufft:
            raise ImportError('finufft cannot be imported')
        self.correct_motion = correct_motion
        self.to_substract = None
        self.res_dir = res_dir
        self.nb_saved = 0

    def apply_transform(self, sample):
        for image_name, image_dict in sample.get_images_dict().items():
            image_data = np.squeeze(image_dict['data'])[..., np.newaxis, np.newaxis]
            original_image = np.squeeze(image_data[:, :, :, 0, 0])

            if self.oversampling_pct > 0.0:
                original_image_shape = original_image.shape
                original_image = self._oversample(original_image, self.oversampling_pct)

            self._calc_dimensions(original_image.shape)

            if self.simulate_displacement:
                self._simulate_random_trajectory()

            if self.fitpars.ndim == 4: #we assume the interpolation has been done on the input
                fitpars_interp = self.fitpars
            else:
                fitpars_interp = self._interpolate_space_timing(self.fitpars)
                fitpars_interp = self._tile_params_to_volume_dims(fitpars_interp)

            fitpars_interp = self.demean_fitpar(fitpars_interp, original_image)

            fitpars_vox = fitpars_interp.reshape((6, -1))
            translations, rotations = fitpars_vox[:3], np.radians(fitpars_vox[3:])
            # fft
            im_freq_domain = self._fft_im(original_image)
            translated_im_freq_domain = self._translate_freq_domain(freq_domain=im_freq_domain,
                                                                    translations=translations)
            # iNufft for rotations
            if self.nufft:
                corrupted_im = self._nufft(translated_im_freq_domain, rotations=rotations)
                corrupted_im = corrupted_im / corrupted_im.size  # normalize

            else:
                corrupted_im = self._ifft_im(translated_im_freq_domain)

            if self.correct_motion:
                corrected_im = self.do_correct_motion(corrupted_im)
                image_dict["data_cor"] = corrected_im[np.newaxis, ...]
                image_dict['data_cor'] = torch.from_numpy(image_dict['data_cor']).float()

            # magnitude
            corrupted_im = abs(corrupted_im)

            if self.oversampling_pct > 0.0:
                corrupted_im = self.crop_volume(corrupted_im, original_image_shape)

            image_dict["data"] = corrupted_im[np.newaxis, ...]
            image_dict['data'] = torch.from_numpy(image_dict['data']).float()

        if self.res_dir is not None:
            self.save_to_dir(image_dict)

        self._compute_motion_metrics(fitpars_interp=fitpars_interp, original_image=original_image)

        return sample

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
            frequency = 2**i
            this_npts = round(npts / frequency)

            if this_npts > 1:
                total += weights[i] * pchip_interpolate(np.linspace(0, 1, this_npts), self._rand_uniform(shape=this_npts)[..., np.newaxis],
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

        maxDisp = self._rand_uniform(min=self.maxDisp[0], max=self.maxDisp[1])
        maxRot = self._rand_uniform(min=self.maxRot[0], max=self.maxRot[1])
        noiseBasePars = self._rand_uniform(min=self.noiseBasePars[0], max=self.noiseBasePars[1])
        swallowMagnitude = [self._rand_uniform(min=self.swallowMagnitude[0], max=self.swallowMagnitude[1]),
                            self._rand_uniform(min=self.swallowMagnitude[0], max=self.swallowMagnitude[1])]

        suddenMagnitude = [self._rand_uniform(min=self.suddenMagnitude[0], max=self.suddenMagnitude[1]),
                           self._rand_uniform(min=self.suddenMagnitude[0], max=self.suddenMagnitude[1])]

        swallowFrequency = torch.randint(self.swallowFrequency[0], self.swallowFrequency[1], (1,)).item()
        suddenFrequency = torch.randint(self.suddenFrequency[0], self.suddenFrequency[1], (1,)).item()

        #prba to include the different type of noise
        proba_noiseBase = self.noiseBasePars[2] if len(self.noiseBasePars) == 3 else 1
        proba_swallow = self.swallowFrequency[2] if len(self.swallowFrequency) == 3 else 1
        proba_sudden = self.suddenFrequency[2] if len(self.suddenFrequency) == 3 else 1
        do_noise, do_swallow, do_sudden = False, False, False
        while (do_noise or do_swallow or do_sudden) is False: #at least one is not false
            do_noise = self._rand_uniform() <= proba_noiseBase
            do_swallow = self._rand_uniform() <= proba_swallow
            do_sudden = self._rand_uniform() <= proba_sudden
        if do_noise is False: noiseBasePars = 0
        if do_swallow is False: swallowFrequency = 0
        if do_sudden is False: suddenFrequency = 0

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
            center = np.int(np.floor(fitpars.shape[1]/2))
            nbpts = np.int(np.floor(fitpars.shape[1] * self.preserve_center_frequency_pct/2))
            fitpars[:, center-nbpts:center+nbpts] = 0
        self.fitpars = fitpars


    def _interpolate_space_timing_1D(self, fitpars):
        n_phase = self.phase_encoding_shape[0]
        nT = self.nT
        # Time steps
        mg_total = np.linspace(0,1,n_phase)
        # Equidistant time spacing
        teq = np.linspace(0, 1, nT)
        # Actual interpolation
        fitpars_interp = np.asarray([np.interp(mg_total, teq, params) for params in fitpars])
        # Reshaping to phase encoding dimensions
        # Add missing dimension
        fitpars_interp = np.expand_dims(fitpars_interp, axis=[self.frequency_encoding_dim + 1,self.phase_encoding_dims[1] + 1])
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
        # Add missing dimension
        fitpars_interp = np.expand_dims(fitpars_interp, axis=self.frequency_encoding_dim + 1)
        return fitpars_interp

    def _tile_params_to_volume_dims(self, params_to_reshape):
        target_shape = [6] + self.im_shape
        data_shape = params_to_reshape.shape
        tiles = np.floor_divide(target_shape, data_shape, dtype=int)
        return np.tile(params_to_reshape, reps=tiles)

    def _translate_freq_domain(self, freq_domain, translations, inv_transfo=False):
        """
        image domain translation by adding phase shifts in frequency domain
        :param freq_domain - frequency domain data 3d numpy array:
        :return frequency domain array with phase shifts added according to self.translations:
        """
        translations = -translations if inv_transfo else translations

        lin_spaces = [np.linspace(-0.5, 0.5, x) for x in freq_domain.shape] #todo it suposes 1 vox = 1mm
        meshgrids = np.meshgrid(*lin_spaces, indexing='ij')
        grid_coords = np.array([mg.flatten() for mg in meshgrids])

        phase_shift = np.multiply(grid_coords, translations).sum(axis=0)  # phase shift is added
        exp_phase_shift = np.exp(-2j * math.pi * phase_shift)
        freq_domain_translated = exp_phase_shift * freq_domain.reshape(-1)

        return freq_domain_translated.reshape(freq_domain.shape)

    def _rotate_coordinates(self, rotations, inv_transfo=False):
        """
        :return: grid_coordinates after applying self.rotations
        """

        rotations = -rotations if inv_transfo else rotations

        center = [math.ceil((x - 1) / 2) for x in self.im_shape]

        [i1, i2, i3] = np.meshgrid(2*(np.arange(self.im_shape[0]) - center[0])/self.im_shape[0],
                                   2*(np.arange(self.im_shape[1]) - center[1])/self.im_shape[1],
                                   2*(np.arange(self.im_shape[2]) - center[2])/self.im_shape[2], indexing='ij')

        #to rotate coordinate between -1 and 1 is not equivalent to compute it betawe -100 and 100 and divide by 100
        #special thanks to the matlab code from Gallichan  https://github.com/dgallichan/retroMoCoBox

        grid_coordinates = np.array([i1.flatten('F'), i2.flatten('F'), i3.flatten('F')])

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

    def _nufft(self, freq_domain_data, rotations, eps=1E-7,  inv_transfo=False):
        """
        rotate coordinates and perform nufft
        :param freq_domain_data:
        :param eps: see finufft doc
        :param eps: precision of nufft
        :return: nufft of freq_domain_data after applying self.rotations
        """
        if not _finufft:
            raise ImportError('finufft not available')

        new_grid_coords = self._rotate_coordinates(inv_transfo=inv_transfo, rotations=rotations)[0]
        # initialize array for nufft output
        f = np.zeros(self.im_shape, dtype=np.complex128, order='F')

        freq_domain_data_flat = np.asfortranarray(freq_domain_data.flatten(order='F'))

        finufft.nufft3d1(new_grid_coords[0], new_grid_coords[1], new_grid_coords[2], freq_domain_data_flat,
                           eps=eps, out=f, debug=0, spread_debug=0, spread_sort=2, fftw=0, modeord=0,
                           chkbnds=0, upsampfac=1.25)  # upsampling at 1.25 saves time at low precisions
        im_out = f.reshape(self.im_shape, order='F')

        return im_out

    def _rand_uniform(self, min=0.0, max=1.0, shape=1):
        rand = torch.FloatTensor(shape).uniform_(min, max)
        if shape == 1:
            return rand.item()
        return rand.numpy()

    def _rand_choice(self, array):
        chosen_idx = torch.randint(0, len(array), (1, ))
        return array[chosen_idx]

    def _compute_motion_metrics(self, fitpars_interp, original_image):
        self.mean_DispP = calculate_mean_Disp_P(self.fitpars)
        self.rmse_Disp = calculate_mean_RMSE_displacment(self.fitpars)
        self.mean_DispP_iterp = calculate_mean_Disp_P(fitpars_interp)
        self.rmse_Disp_iterp = calculate_mean_RMSE_displacment(fitpars_interp)

        #ff_interp, to_substract = self.demean_fitpar(fitpars_interp, original_image)
        #self.rmse_DispTF = calculate_mean_RMSE_displacment(ff_interp, original_image)

    def demean_fitpar(self, fitpars_interp, original_image):
        if self.displacement_shift_strategy == "demean":
            o_shape = original_image.shape
            tfi = np.abs(np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(original_image))).astype(np.complex128))
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

        elif self.displacement_shift_strategy == "demean_half":
            nb_pts_around = 31
            print('RR demean_center around {}'.format(nb_pts_around))
            # let's take the weight from the tf, but only in the center (+- 11 pts)
            o_shape = original_image.shape
            tfi = np.abs(np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(original_image))).astype(np.complex128))
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

        elif self.displacement_shift_strategy == "center_zero":
            dim = fitpars_interp.shape
            center = [int(round(dd / 2)) for dd in dim]
            to_substract = fitpars_interp[:, center[1], center[2], center[3]]
            to_substract_tile = np.tile(to_substract[..., np.newaxis, np.newaxis, np.newaxis], (1, dim[1], dim[2], dim[3]))
            fitpars_interp = fitpars_interp - to_substract_tile

        else:
            to_substract = np.array([0, 0, 0, 0, 0, 0])

        self.to_substract = to_substract
        return fitpars_interp



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
