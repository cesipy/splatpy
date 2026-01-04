# 🎥 Splatpy

**Transform videos into stunning 3D Gaussian Splats in minutes.**

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/CUDA-12.4-green.svg" alt="CUDA 12.4">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License">
</p>

<p align="center">
  <img src="res/assets/comparison.gif" width="90%" alt="Splatpy Comparison">
  <br>
  <small><b>Left:</b> Input Video | <b>Right:</b> 3D Gaussian Splat Result</small>
</p>

## Installation

**Requirements:**
- **NVIDIA Driver:** Compatible with CUDA 12.4+ (Driver version ≥ 550.54 on Linux / ≥ 551.61 on Windows)
- **CUDA Toolkit:** 12.4+ (required for compilation)
- **Python:** 3.10+
- **FFmpeg**

**⚠️ Note:** Due to CUDA requirements, this package cannot be installed via `pip install splatpy`.
Please follow the instructions below.


### Using uv (recommended)
```bash
uv sync --python 3.10
source .venv/bin/activate
python src/main.py
```
### Using pip
Install `torch` manually with the correct CUDA version, then install `splatpy` from source:

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install --no-build-isolation git+https://github.com/cesipy/splatpy.git
```

Note: `--no-build-isolation` is required because gsplat and fused-ssim need PyTorch during compilation.


## Quick Start
The following video formats are supported:
- MP4 (`.mp4`)
- AVI (`.avi`)
- MOV (`.mov`)

### Simple Usage (Recommended)

```python
from splatpy import video_to_splat

# Convert video to 3D Gaussian Splat with quality preset
output = video_to_splat("path/to/video.mp4", quality="medium")
print(f"Splat saved to: {output}")
```

Quality presets:
- `"test"`: Quick test run (1k steps, frames_modulo=30, sift_features=256)
- `"low"`: Fast preview (10k steps, frames_modulo=30, sift_features=1024)
- `"medium"`: Balanced quality (20k steps, frames_modulo=20, sift_features=2048) **[default]**
- `"high"`: High quality (50k steps, frames_modulo=15, sift_features=4096)
- `"ultra"`: Maximum quality (100k steps, frames_modulo=10, sift_features=8192)

### Advanced Usage

For full control over training parameters:

```python
from splatpy import video_to_splat_advanced

output = video_to_splat_advanced(
    video_path="path/to/video.mp4",
    output_dir="results/",
    training_steps=50_000,
    frames_modulo=10,        # Extract every 10th frame
    data_factor=1,           # Full resolution images
    colmap_mode="sequential",
    sh_degree=3,             # Spherical harmonics degree
    render_orbit=True,
    orbit_frames=240
)
```

### Custom Configuration

For maximum control, use the `TrainingConfig` class:

```python
from splatpy import video_to_splat_advanced, TrainingConfig
from gsplat.strategy import MCMCStrategy

config = TrainingConfig(
    data_dir="res/output/",
    data_factor=1,
    results_dir="res/results/",
    sh_degree=3,
    means_lr=1.6e-4,
    scales_lr=5e-3,
    opacities_lr=5e-2,
    strategy=MCMCStrategy()
)

output = video_to_splat_advanced(
    video_path="path/to/video.mp4",
    custom_config=config,
    training_steps=100_000
)
```


## How it works

The pipeline is straightforward:

1. **Frame extraction** - sample frames from input video at regular intervals
2. **COLMAP**  - Structure-from-Motion (SfM) to estimate camera poses and a sparse point cloud.
3. **Training** - optimize 3D Gaussians using differentiable rasterization
4. **Export** - save as .ply files compatible with standard viewers

Training uses a combination of L1, SSIM, and LPIPS losses. The Gaussians are initialized from COLMAP's sparse point cloud.

## Development
For Development you need extra dependencies (e.g. pytest):
```bash
# Install with dev dependencies (to run tests)
uv sync --python 3.10 --all-extras

# Run all tests
pytest
```

Bugs, issues, and contributions are welcome!

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
- [ ] Add progress callbacks
- [ ] Web viewer for results
- [ ] Support for pycolmap-cuda-12 (needs build from source)
- [ ] Automatically detect number of frames extracted from video
- [ ] Add evaluation metrics (PSNR, SSIM, LPIPS)
- [ ] Support for image folder input (not just video)
- [ ] Automatic downscaling of images
- [ ] CLI tool
- [ ] Dockerize
- [ ] pylint

## 🙏 Acknowledgments

Built with:
- [gsplat](https://github.com/nerfstudio-project/gsplat) - 3D Gaussian Splatting
- [COLMAP](https://colmap.github.io/) - Structure-from-Motion
- [PyTorch](https://pytorch.org/) - Deep Learning Framework

Test video:
- "House 360 Aerial Orbit" by [Aerial Photography & Drone Video](https://www.youtube.com/watch?v=zvpMpzBh0Y8)


---

<p align="center">
  Made with ❤️ by the Splatpy team
</p>