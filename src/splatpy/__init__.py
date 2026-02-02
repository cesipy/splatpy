from .api import (
    video_to_splat,
    video_to_splat_advanced,
    video_to_gsplat,  # legacy
)
from .config import TrainingConfig, QualityPreset, get_quality_preset
from .trainer import Trainer

__all__ = [
    "video_to_splat",
    "video_to_splat_advanced",
    "video_to_gsplat",
    "TrainingConfig",
    "QualityPreset",
    "get_quality_preset",
    "Trainer",
    "check_installation",
]

__version__ = "0.1.0"


def check_installation():
    """Validate splatpy installation and CUDA availability.

    This function checks that splatpy and its dependencies are properly installed,
    and provides helpful feedback about CUDA support.

    Example:
        >>> import splatpy
        >>> splatpy.check_installation()
        ✓ splatpy installed: 0.1.0
        ✓ PyTorch version: 2.6.0+cu124
        ✓ CUDA available: True
        ✓ CUDA version: 12.4
        ✓ GPU: NVIDIA GeForce RTX 3090
    """
    import torch

    print(f"✓ splatpy installed: {__version__}")
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print("\nYour installation is ready to use!")
    else:
        print("\n✗ CUDA not available")
        print("\nTo fix this, you need to reinstall PyTorch with CUDA support:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
        print("\nThen reinstall splatpy:")
        print("  pip install splatpy")
        print("\nSee https://github.com/cesipy/splatpy#installation for details.")