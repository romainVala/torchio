import numpy as np


class FourierTransform:

    @staticmethod
    def fourier_transform(array: np.ndarray) -> np.ndarray:
        transformed = np.fft.fftn(array)
        fshift = np.fft.fftshift(transformed)
        return fshift

    @staticmethod
    def inv_fourier_transform(fshift: np.ndarray) -> np.ndarray:
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifftn(f_ishift)
        return img_back

    #added for working with nufft (in random_motion_from_time_course), not sure why exactly, but we need different
    # fft shift
    @staticmethod
    def fourier_transform_for_nufft(image):
        output = (np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(image)))).astype(np.complex128)
        return output


    @staticmethod
    def inv_fourier_transform_for_nufft(freq_domain):
        output = np.fft.ifftshift(np.fft.ifftn(freq_domain))
        return output
        #fi_image = np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(fi_phase), axis=1))
