# 🎥 Splatpy

**Transform videos into stunning 3D Gaussian Splats in minutes.**

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/CUDA-12.4-green.svg" alt="CUDA 12.4">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License">
</p>

## Installation

**Requirements:**
- CUDA 12.4+
- Python 3.10+
- FFmpeg

### Using uv (recommended)
```bash
uv sync --python 3.10
source .venv/bin/activate
python src/main.py
```

### Using pip
```bash
python3.10 -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install --no-build-isolation git+https://github.com/cesipy/splatpy.git
```

Note: `--no-build-isolation` is required because gsplat and fused-ssim need PyTorch during compilation.

## Usage
```python
from splatpy import video_to_gsplat

video_to_gsplat("path/to/video.mp4")
```

This will:
1. Extract frames from the video
2. Run COLMAP for camera poses and 3D reconstruction
3. Train Gaussian splats
4. Export results to `res/results/`

## Configuration

You can customize training parameters:
```python
from splatpy.config import TrainingConfig

config = TrainingConfig(
    data_dir="res/output/",
    results_dir="res/results/",
    sh_degree=3,
    means_lr=1.6e-4,
)
```

See `src/splatpy/config.py` for all available options.

## How it works

The pipeline is straightforward:

1. **Frame extraction** - sample frames from input video at regular intervals
2. **COLMAP** - feature extraction, matching, and sparse reconstruction
3. **Training** - optimize 3D Gaussians using differentiable rasterization
4. **Export** - save as .ply files compatible with standard viewers

Training uses a combination of L1, SSIM, and LPIPS losses. The Gaussians are initialized from COLMAP's sparse point cloud.

## Project structure
```
src/splatpy/
  api.py              - main entry point
  trainer.py          - training loop
  config.py           - configuration dataclass
  incremental_pipeline.py - COLMAP wrapper
  utils/
    colmap_datahandling.py
    utils.py
```

## TODOs

- [ ] Auto-detect frame count from video metadata
- [ ] Support pycolmap with CUDA 12
- [ ] Add progress callbacks
- [ ] Web viewer for results

## 🙏 Acknowledgments

Built with:
- [gsplat](https://github.com/nerfstudio-project/gsplat) - 3D Gaussian Splatting
- [COLMAP](https://colmap.github.io/) - Structure-from-Motion
- [PyTorch](https://pytorch.org/) - Deep Learning Framework

---

<p align="center">
  Made with ❤️ by the Splatpy team
</p>