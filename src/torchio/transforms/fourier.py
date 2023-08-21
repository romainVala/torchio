import numpy as np
import torch


class FourierTransform:
    @staticmethod
    def fourier_transform(tensor: torch.Tensor) -> torch.Tensor:
        try:
            import torch.fft

            transformed = torch.fft.fftn(tensor)
            fshift = torch.fft.fftshift(transformed)
            return fshift
        except (ModuleNotFoundError, AttributeError):
            import torch

            transformed = np.fft.fftn(tensor)
            fshift = np.fft.fftshift(transformed)
            return torch.from_numpy(fshift)

    @staticmethod
    def inv_fourier_transform(tensor: torch.Tensor) -> torch.Tensor:
        try:
            import torch.fft

            f_ishift = torch.fft.ifftshift(tensor)
            img_back = torch.fft.ifftn(f_ishift)
            return img_back
        except (ModuleNotFoundError, AttributeError):
            import torch

            f_ishift = np.fft.ifftshift(tensor)
            img_back = np.fft.ifftn(f_ishift)
            return torch.from_numpy(img_back)


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
