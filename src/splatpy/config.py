from dataclasses import dataclass
from dataclasses import field
from typing import Union, Literal
from gsplat.strategy import DefaultStrategy, MCMCStrategy


@dataclass
class TrainingConfig:
    colmap_data_dir: str = ".splatpy_dir"

    data_dir: str = "res/output/images/"       # where the colmap is stored
    data_factor:int = 1                 # downscaling factor for images
    results_dir: str = "res/results/"   # where to store results

    # the configuration below is mainly the same as in
    # https://github.com/nerfstudio-project/gsplat/blob/b60e917c95afc449c5be33a634f1f457e116ff5e/examples/simple_trainer.py#L44
    batch_size:int = 1
    # LR for 3D point positions
    means_lr: float = 1.6e-4
    # LR for Gaussian scale factors
    scales_lr: float = 5e-3
    # LR for alpha blending weights
    opacities_lr: float = 5e-2
    # LR for orientation (quaternions)
    quats_lr: float = 1e-3
    # LR for SH band 0 (brightness)
    sh0_lr: float = 2.5e-3
    # LR for higher-order SH (detail)
    shN_lr: float = 2.5e-3 / 20

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda = 0.2

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )

class QualityPreset:
    def __init__(self, quality:str):
        valid_qualities = ["test","low", "medium", "high", "ultra"]
        if quality not in valid_qualities:
            raise ValueError(f"Quality must be one of {valid_qualities}, got '{quality}'")
        self.quality = quality
        self.__set_preset(quality)

    def __set_preset(self, quality:str):
        if quality == "test":
            self.steps = 1_000
            self.sh_degree = 2
            self.frames_modulo = 30
            self.sift_features = 256
            self.description = "Quick test run"

        elif quality == "low":
            self.steps = 10_000
            self.frames_modulo = 30
            self.sh_degree = 3
            self.sift_features = 1024
            self.description = "Fast preview quality - good for testing"

        elif quality == "medium":
            self.steps = 20_000
            self.frames_modulo = 20
            self.sh_degree = 3
            self.sift_features = 2048
            self.description = "Balanced quality and speed"

        elif quality == "high":
            self.steps = 50_000
            self.frames_modulo = 15
            self.sh_degree = 3
            self.sift_features = 4096
            self.description = "High quality results"

        elif quality == "ultra":
            self.steps = 100_000
            self.frames_modulo = 10
            self.data_factor = 1
            self.sh_degree = 3
            self.sift_features = 8192
            self.description = "Maximum quality - slow but best results"
        self.name = quality

def get_quality_preset(quality: QualityPreset):
    preset = QualityPreset(quality)
    return preset

