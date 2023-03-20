import warnings
from pathlib import Path
from typing import Optional
from typing import Tuple
from typing import Union

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch

from ..constants import REPO_URL
from ..typing import TypeData
from ..typing import TypeDataAffine
from ..typing import TypeDirection
from ..typing import TypeDoubletInt
from ..typing import TypePath
from ..typing import TypeQuartetInt
from ..typing import TypeTripletFloat
from ..typing import TypeTripletInt


# Matrices used to switch between LPS and RAS
FLIPXY_33 = np.diag([-1, -1, 1])
FLIPXY_44 = np.diag([-1, -1, 1, 1])

# Image formats that are typically 2D
formats = ['.jpg', '.jpeg', '.bmp', '.png', '.tif', '.tiff']
IMAGE_2D_FORMATS = formats + [s.upper() for s in formats]


def read_image(path: TypePath) -> TypeDataAffine:
    try:
        result = _read_sitk(path)
    except RuntimeError as e:  # try with NiBabel
        message = (
            f'Error loading image with SimpleITK:\n{e}\n\nTrying NiBabel...'
        )
        warnings.warn(message)
        try:
            result = _read_nibabel(path)
        except nib.loadsave.ImageFileError as e:
            message = (
                f'File "{path}" not understood.'
                ' Check supported formats by at'
                ' https://simpleitk.readthedocs.io/en/master/IO.html#images'
                ' and https://nipy.org/nibabel/api.html#file-formats'
            )
            raise RuntimeError(message) from e
    return result


def _read_nibabel(path: TypePath) -> TypeDataAffine:
    img = nib.load(str(path), mmap=False)
    data = img.get_fdata(dtype=np.float32)
    if data.ndim == 5:
        data = data[..., 0, :]
        data = data.transpose(3, 0, 1, 2)
    data = check_uint_to_int(data)
    tensor = torch.as_tensor(data)
    affine = img.affine
    return tensor, affine


def _read_sitk(path: TypePath) -> TypeDataAffine:
    if Path(path).is_dir():  # assume DICOM
        image = _read_dicom(path)
    else:
        image = sitk.ReadImage(str(path))
    data, affine = sitk_to_nib(image, keepdim=True)
    data = check_uint_to_int(data)
    tensor = torch.as_tensor(data)
    return tensor, affine


def _read_dicom(directory: TypePath):
    directory = Path(directory)
    if not directory.is_dir():  # unreachable if called from _read_sitk
        raise FileNotFoundError(f'Directory "{directory}" not found')
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(directory))
    if not dicom_names:
        message = (
            f'The directory "{directory}"'
            ' does not seem to contain DICOM files'
        )
        raise FileNotFoundError(message)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return image


def read_shape(path: TypePath) -> TypeQuartetInt:
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    num_channels = reader.GetNumberOfComponents()
    num_dimensions = reader.GetDimension()
    assert 2 <= num_dimensions <= 4
    if num_dimensions == 2:
        spatial_shape_2d: TypeDoubletInt = reader.GetSize()
        assert len(spatial_shape_2d) == 2
        si, sj = spatial_shape_2d
        sk = 1
    elif num_dimensions == 4:
        # We assume bad NIfTI file (channels encoded as spatial dimension)
        spatial_shape_4d: TypeQuartetInt = reader.GetSize()
        assert len(spatial_shape_4d) == 4
        si, sj, sk, num_channels = spatial_shape_4d
    elif num_dimensions == 3:
        spatial_shape_3d: TypeTripletInt = reader.GetSize()
        assert len(spatial_shape_3d) == 3
        si, sj, sk = spatial_shape_3d
    shape = num_channels, si, sj, sk
    return shape


def read_affine(path: TypePath) -> np.ndarray:
    reader = get_reader(path)
    affine = get_ras_affine_from_sitk(reader)
    return affine


def get_reader(path: TypePath, read: bool = True) -> sitk.ImageFileReader:
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    if read:
        reader.ReadImageInformation()
    return reader


def write_image(
        tensor: torch.Tensor,
        affine: TypeData,
        path: TypePath,
        squeeze: Optional[bool] = None,
) -> None:
    args = tensor, affine, path
    try:
        _write_nibabel(*args)
    except RuntimeError:  # try with NiBabel
        _write_sitk(*args, squeeze=squeeze)

def _write_nibabel(
        tensor: torch.Tensor,
        affine: TypeData,
        path: TypePath,
) -> None:
    """
    Expects a path with an extension that can be used by nibabel.save
    to write a NIfTI-1 image, such as '.nii.gz' or '.img'
    """
    assert tensor.ndim == 4
    num_components = tensor.shape[0]

    # NIfTI components must be at the end, in a 5D array
    #I change taht, where this 5D comes from ?
    if num_components == 1:
        tensor = tensor[0]
    else:
        #tensor = tensor[np.newaxis].permute(2, 3, 4, 1, 0)
        tensor = tensor.permute(1, 2, 3, 0)
    if (tensor.dtype is torch.float16) or (tensor.dtype is torch.int64) :
        tensor = tensor.to(torch.float32)
    suffix = Path(str(path).replace('.gz', '')).suffix
    if '.nii' in suffix:
        img = nib.Nifti1Image(np.asarray(tensor), affine)
    elif '.hdr' in suffix or '.img' in suffix:
        img = nib.Nifti1Pair(np.asarray(tensor), affine)
    else:
        raise nib.loadsave.ImageFileError
    if num_components > 1:
        img.header.set_intent('vector')
    img.header['qform_code'] = 1
    img.header['sform_code'] = 1
    nib.save(img, str(path))


def _write_sitk(
        tensor: torch.Tensor,
        affine: TypeData,
        path: TypePath,
        use_compression: bool = True,
        squeeze: Optional[bool] = None,
) -> None:
    assert tensor.ndim == 4
    path = Path(path)
    if path.suffix in ('.png', '.jpg', '.jpeg', '.bmp'):
        warnings.warn(
            f'Casting to uint 8 before saving to {path}',
            RuntimeWarning,
        )
        tensor = tensor.numpy().astype(np.uint8)
    if squeeze is None:
        force_3d = path.suffix not in IMAGE_2D_FORMATS
    else:
        force_3d = not squeeze
    image = nib_to_sitk(tensor, affine, force_3d=force_3d)
    sitk.WriteImage(image, str(path), use_compression)


def read_matrix(path: TypePath):
    """Read an affine transform and convert to tensor."""
    path = Path(path)
    suffix = path.suffix
    if suffix in ('.tfm', '.h5'):  # ITK
        tensor = _read_itk_matrix(path)
    elif suffix in ('.txt', '.trsf'):  # NiftyReg, blockmatching
        tensor = _read_niftyreg_matrix(path)
    else:
        raise ValueError(f'Unknown suffix for transform file: "{suffix}"')
    return tensor


def write_matrix(matrix: torch.Tensor, path: TypePath):
    """Write an affine transform."""
    path = Path(path)
    suffix = path.suffix
    if suffix in ('.tfm', '.h5'):  # ITK
        _write_itk_matrix(matrix, path)
    elif suffix in ('.txt', '.trsf'):  # NiftyReg, blockmatching
        _write_niftyreg_matrix(matrix, path)


def _to_itk_convention(matrix):
    """RAS to LPS"""
    matrix = np.dot(FLIPXY_44, matrix)
    matrix = np.dot(matrix, FLIPXY_44)
    matrix = np.linalg.inv(matrix)
    return matrix


def _from_itk_convention(matrix):
    """LPS to RAS"""
    matrix = np.dot(matrix, FLIPXY_44)
    matrix = np.dot(FLIPXY_44, matrix)
    matrix = np.linalg.inv(matrix)
    return matrix


def _read_itk_matrix(path):
    """Read an affine transform in ITK's .tfm format"""
    transform = sitk.ReadTransform(str(path))
    parameters = transform.GetParameters()
    rotation_parameters = parameters[:9]
    rotation_matrix = np.array(rotation_parameters).reshape(3, 3)
    translation_parameters = parameters[9:]
    translation_vector = np.array(translation_parameters).reshape(3, 1)
    matrix = np.hstack([rotation_matrix, translation_vector])
    homogeneous_matrix_lps = np.vstack([matrix, [0, 0, 0, 1]])
    homogeneous_matrix_ras = _from_itk_convention(homogeneous_matrix_lps)
    return torch.as_tensor(homogeneous_matrix_ras)


def _write_itk_matrix(matrix, tfm_path):
    """The tfm file contains the matrix from floating to reference."""
    transform = _matrix_to_itk_transform(matrix)
    transform.WriteTransform(str(tfm_path))


def _matrix_to_itk_transform(matrix, dimensions=3):
    matrix = _to_itk_convention(matrix)
    rotation = matrix[:dimensions, :dimensions].ravel().tolist()
    translation = matrix[:dimensions, 3].tolist()
    transform = sitk.AffineTransform(rotation, translation)
    return transform


def _read_niftyreg_matrix(trsf_path):
    """Read a NiftyReg matrix and return it as a NumPy array"""
    matrix = np.loadtxt(trsf_path)
    matrix = np.linalg.inv(matrix)
    return torch.as_tensor(matrix)


def _write_niftyreg_matrix(matrix, txt_path):
    """Write an affine transform in NiftyReg's .txt format (ref -> flo)"""
    matrix = np.linalg.inv(matrix)
    np.savetxt(txt_path, matrix, fmt='%.8f')


def get_rotation_and_spacing_from_affine(
        affine: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    # From https://github.com/nipy/nibabel/blob/master/nibabel/orientations.py
    rotation_zoom = affine[:3, :3]
    spacing = np.sqrt(np.sum(rotation_zoom * rotation_zoom, axis=0))
    rotation = rotation_zoom / spacing
    return rotation, spacing


def nib_to_sitk(
        data: TypeData,
        affine: TypeData,
        force_3d: bool = False,
        force_4d: bool = False,
) -> sitk.Image:
    """Create a SimpleITK image from a tensor and a 4x4 affine matrix."""
    if data.ndim != 4:
        shape = tuple(data.shape)
        raise ValueError(f'Input must be 4D, but has shape {shape}')
    # Possibilities
    # (1, w, h, 1)
    # (c, w, h, 1)
    # (1, w, h, 1)
    # (c, w, h, d)
    array = np.asarray(data)
    affine = np.asarray(affine).astype(np.float64)

    is_multichannel = array.shape[0] > 1 and not force_4d
    is_2d = array.shape[3] == 1 and not force_3d
    if is_2d:
        array = array[..., 0]
    if not is_multichannel and not force_4d:
        array = array[0]
    array = array.transpose()  # (W, H, D, C) or (W, H, D)
    image = sitk.GetImageFromArray(array, isVector=is_multichannel)

    origin, spacing, direction = get_sitk_metadata_from_ras_affine(
        affine,
        is_2d=is_2d,
    )
    image.SetOrigin(origin)  # should I add a 4th value if force_4d?
    image.SetSpacing(spacing)
    image.SetDirection(direction)

    if data.ndim == 4:
        assert image.GetNumberOfComponentsPerPixel() == data.shape[0]
    num_spatial_dims = 2 if is_2d else 3
    assert image.GetSize() == data.shape[1:1 + num_spatial_dims]

    return image


def sitk_to_nib(
        image: sitk.Image,
        keepdim: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    data = sitk.GetArrayFromImage(image).transpose()
    data = check_uint_to_int(data)
    num_components = image.GetNumberOfComponentsPerPixel()
    if num_components == 1:
        data = data[np.newaxis]  # add channels dimension
    input_spatial_dims = image.GetDimension()
    if input_spatial_dims == 2:
        data = data[..., np.newaxis]
    elif input_spatial_dims == 4:  # probably a bad NIfTI (1, sx, sy, sz, c)
        # Try to fix it
        num_components = data.shape[-1]
        data = data[0]
        data = data.transpose(3, 0, 1, 2)
        input_spatial_dims = 3
    if not keepdim:
        data = ensure_4d(data, num_spatial_dims=input_spatial_dims)
    assert data.shape[0] == num_components
    affine = get_ras_affine_from_sitk(image)
    return data, affine


def get_ras_affine_from_sitk(
        sitk_object: Union[sitk.Image, sitk.ImageFileReader],
) -> np.ndarray:
    spacing = np.array(sitk_object.GetSpacing())
    direction_lps = np.array(sitk_object.GetDirection())
    origin_lps = np.array(sitk_object.GetOrigin())
    direction_length = len(direction_lps)
    if direction_length == 9:
        rotation_lps = direction_lps.reshape(3, 3)
    elif direction_length == 4:  # ignore last dimension if 2D (1, W, H, 1)
        rotation_lps_2d = direction_lps.reshape(2, 2)
        rotation_lps = np.eye(3)
        rotation_lps[:2, :2] = rotation_lps_2d
        spacing = np.append(spacing, 1)
        origin_lps = np.append(origin_lps, 0)
    elif direction_length == 16:  # probably a bad NIfTI. Let's try to fix it
        rotation_lps = direction_lps.reshape(4, 4)[:3, :3]
        spacing = spacing[:-1]
        origin_lps = origin_lps[:-1]
    rotation_ras = np.dot(FLIPXY_33, rotation_lps)
    rotation_ras_zoom = rotation_ras * spacing
    translation_ras = np.dot(FLIPXY_33, origin_lps)
    affine = np.eye(4)
    affine[:3, :3] = rotation_ras_zoom
    affine[:3, 3] = translation_ras
    return affine


def get_sitk_metadata_from_ras_affine(
        affine: np.ndarray,
        is_2d: bool = False,
        lps: bool = True,
) -> Tuple[TypeTripletFloat, TypeTripletFloat, TypeDirection]:
    direction_ras, spacing_array = get_rotation_and_spacing_from_affine(affine)
    origin_ras = affine[:3, 3]
    origin_lps = np.dot(FLIPXY_33, origin_ras)
    direction_lps = np.dot(FLIPXY_33, direction_ras)
    if is_2d:  # ignore orientation if 2D (1, W, H, 1)
        direction_lps = np.diag((-1, -1)).astype(np.float64)
        direction_ras = np.diag((1, 1)).astype(np.float64)
    origin_array = origin_lps if lps else origin_ras
    direction_array = direction_lps if lps else direction_ras
    direction_array = direction_array.flatten()
    # The following are to comply with mypy
    # (although there must be prettier ways to do this)
    ox, oy, oz = origin_array
    sx, sy, sz = spacing_array
    direction: TypeDirection
    if is_2d:
        d1, d2, d3, d4 = direction_array
        direction = d1, d2, d3, d4
    else:
        d1, d2, d3, d4, d5, d6, d7, d8, d9 = direction_array
        direction = d1, d2, d3, d4, d5, d6, d7, d8, d9
    origin = ox, oy, oz
    spacing = sx, sy, sz
    return origin, spacing, direction


def ensure_4d(tensor: TypeData, num_spatial_dims=None) -> TypeData:
    # I wish named tensors were properly supported in PyTorch
    tensor = torch.as_tensor(tensor)
    num_dimensions = tensor.ndim
    if num_dimensions == 4:
        pass
    elif num_dimensions == 5:  # hope (W, H, D, 1, C)
        if tensor.shape[-2] == 1:
            tensor = tensor[..., 0, :]
            tensor = tensor.permute(3, 0, 1, 2)
        else:
            raise ValueError('5D is not supported for shape[-2] > 1')
    elif num_dimensions == 2:  # assume 2D monochannel (W, H)
        tensor = tensor[np.newaxis, ..., np.newaxis]  # (1, W, H, 1)
    elif num_dimensions == 3:  # 2D multichannel or 3D monochannel?
        if num_spatial_dims == 2:
            tensor = tensor[..., np.newaxis]  # (C, W, H, 1)
        elif num_spatial_dims == 3:  # (W, H, D)
            tensor = tensor[np.newaxis]  # (1, W, H, D)
        else:  # try to guess
            shape = tensor.shape
            maybe_rgb = 3 in (shape[0], shape[-1])
            if maybe_rgb:
                if shape[-1] == 3:  # (W, H, 3)
                    tensor = tensor.permute(2, 0, 1)  # (3, W, H)
                tensor = tensor[..., np.newaxis]  # (3, W, H, 1)
            else:  # (W, H, D)
                tensor = tensor[np.newaxis]  # (1, W, H, D)
    else:
        message = (
            f'{num_dimensions}D images not supported yet. Please create an'
            f' issue in {REPO_URL} if you would like support for them'
        )
        raise ValueError(message)
    assert tensor.ndim == 4
    return tensor


def check_uint_to_int(array):
    # This is because PyTorch won't take uint16 nor uint32
    if array.dtype == np.uint16:
        return array.astype(np.int32)
    if array.dtype == np.uint32:
        return array.astype(np.int64)
    return array
