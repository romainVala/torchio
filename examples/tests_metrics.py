from torchio import Subject, Image, ImagesDataset
from torchio.transforms import RandomMotionFromTimeCourse
from torchio.metrics import SSIM3D, MetricWrapper
from torchio.metrics.ssim import functional_ssim
from torch.nn import MSELoss, L1Loss
from nibabel.viewers import OrthoSlicer3D as ov

t1_path = "/data/romain/HCPdata/suj_100307/T1w_acpc_dc_restore.nii"
mask_path = "/data/romain/HCPdata/suj_100307/cat12/fill_mask_head.nii.gz"
dataset = ImagesDataset([
    Subject({
        "T1": Image(t1_path),
        "mask": Image(mask_path)
    })])

metrics = {
    "L1": MetricWrapper("L1", L1Loss()),
    "L2": MetricWrapper("L2", MSELoss()),
    "SSIM": SSIM3D(return_map=False, discard_zeros=True),
    "SSIM_Wrapped": MetricWrapper("SSIM_wrapped", lambda x, y: functional_ssim(x, y, discard_zeros=True), use_mask=True, mask_key="mask"),
    "ssim_base": MetricWrapper('SSIM_base', ssim3D)
}

motion_trsfm = RandomMotionFromTimeCourse(verbose=True, compare_to_original=True, metrics=metrics,
                                          oversampling_pct=0.0)

dataset.set_transform(motion_trsfm)

tf = dataset[0]
computed_metrics = tf["T1"]["metrics"]
print("Computed metrics: {}".format(computed_metrics))

ov(tf['T1']['data'].squeeze().numpy())
