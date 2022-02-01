import numpy as np
from transforms3d.euler import euler2mat

#does not work ... circular import
#from ..transforms.augmentation.intensity.random_motion_from_time_course import get_matrix_from_euler_and_trans
def change_affine_rotation_center(A, new_center):
    aff_center = np.eye(4);
    aff_center[:3, 3] = np.array(new_center)
    return np.dot(aff_center, np.dot(A, np.linalg.inv(aff_center)))

#warning duplicate function from random_motion_from_time_course ... (because cant import ...)
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
        T_rb = get_matrix_from_euler_and_trans(P)
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


def compute_motion_metrics( fitpars, img_fft, fast_dim=(0,2)):
    metrics = dict()
    metrics["mean_DispP"] = calculate_mean_Disp_P(fitpars)
    metrics["mean_DispJ"] = calculate_mean_Disp_J(fitpars)
    #metrics["rmse_Disp"] = calculate_mean_RMSE_displacment(fitpars)
    metrics["rmse_Trans"], metrics["rmse_Rot"] = calculate_mean_RMSE_trans_rot(fitpars)


    coef_TF = np.sum(abs(img_fft), axis=tuple(fast_dim)) ;
    coef_shaw = np.sqrt( np.sum(abs(img_fft)**2, axis=tuple(fast_dim)) ) ;
    # I do not see diff, but may be better to write with complex conjugate, here the fft is done on abs image, so I guess the
    # phase does not matter (Cf todd 2015)
    #print(f'averagin TF coef on dim {dim_to_average} shape coef {coef_TF.shape}')
    if fitpars.shape[1] != coef_TF.shape[0] :
        #just interpolate end to end. at image slowest dimention size
        fitpars = _interpolate_fitpars(fitpars, len_output=coef_TF.shape[0])
        print(f' WARNING SHOULD NOT HAPPEN rrr !!! interp fitpar for wcoef new shape {fitpars.shape}')

    metrics["meanDispJ_wTF"]  = calculate_mean_Disp_J(fitpars,  weights=coef_TF)
    metrics["meanDispJ_wSH"]  = calculate_mean_Disp_J(fitpars,  weights=coef_shaw)
    metrics["meanDispJ_wTF2"] = calculate_mean_Disp_J(fitpars,  weights=coef_TF**2)
    metrics["meanDispJ_wSH2"] = calculate_mean_Disp_J(fitpars,  weights=coef_shaw**2)

    metrics["meanDispP_wSH"] = calculate_mean_Disp_P(fitpars,  weights=coef_shaw)
    metrics["rmse_Trans_wSH"], metrics["rmse_Rot_wSH"] = calculate_mean_RMSE_trans_rot(fitpars, weights=coef_shaw)
    metrics["rmse_Trans_wTF2"], metrics["rmse_Rot_wTF2"] = calculate_mean_RMSE_trans_rot(fitpars, weights=coef_TF**2)

    #compute meand disp as weighted mean (weigths beeing TF coef)
    w_coef = np.abs(img_fft)

    ff = fitpars
    for i in range(0, 6):
        ffi = ff[i].reshape(-1)
        metrics[f'wTFshort_Disp_{i}'] = np.sum(ffi * coef_TF) / np.sum(coef_TF)
        metrics[f'wTFshort2_Disp_{i}'] = np.sum(ffi * coef_TF**2) / np.sum(coef_TF**2)
        metrics[f'wSH_Disp_{i}'] = np.sum(ffi * coef_shaw) / np.sum(coef_shaw)
        metrics[f'wSH2_Disp_{i}'] = np.sum(ffi * coef_shaw**2) / np.sum(coef_shaw**2)
        metrics[f'mean_Disp_{i}'] = np.mean(ffi)
        metrics[f'center_Disp_{i}'] = ffi[ffi.shape[0]//2]
    #at the end only SH and SH2 seems ok
    # TF2 == SH2  but TFshort==TF and TFshort2 < TF2 !

    return metrics