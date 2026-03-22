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
- **NVIDIA GPU** with CUDA 12.4+ support (Driver version ≥ 550.54 on Linux / ≥ 551.61 on Windows)
- **Python:** 3.10+
- **FFmpeg**

### Installing from PyPI
PyTorch with CUDA is not available on standard PyPI, so you need a two-step installation:

```bash
# Step 1: Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Step 2: Install splatpy
pip install splatpy
```

**Verify installation:**
```bash
python -c "import splatpy; splatpy.check_installation()"
```

### Development Setup
For development, `uv` handles CUDA dependencies automatically:

```bash
uv sync --python 3.10 --all-extras
source .venv/bin/activate
pytest  # Run tests
```

**Note:** If you get a CUDA error when running splatpy, make sure you installed PyTorch with CUDA support using the instructions above.


## Quick Start
The following video formats are supported:
- MP4 (`.mp4`)
- AVI (`.avi`)
- MOV (`.mov`)

### CLI (Recommended)

```bash
splatpy video.mp4
splatpy video.mp4 --quality high --output my_results/
splatpy video.mp4 --no-orbit
splatpy video.mp4 --depth-prior   # denser point cloud via Depth Anything V2
```

To view your result, drag and drop the `.ply` file into **[supersplat.playcanvas.com](https://supersplat.playcanvas.com)**.

#### Depth Prior (optional)

The `--depth-prior` flag uses [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) to estimate dense depth maps for each frame, scale-aligns them to the COLMAP sparse reconstruction, and merges the unprojected points into the Gaussian initialization. This gives a much denser starting point and helps especially with textureless surfaces (walls, floors, smooth objects) where COLMAP struggles.

First install the extra dependency:

```bash
pip install transformers accelerate
# or
pip install "splatpy[depth]"
```

Then run:

```bash
splatpy video.mp4 --depth-prior --quality high
```

### Python API

```python
from splatpy import video_to_splat

# Convert video to 3D Gaussian Splat with quality preset
output = video_to_splat("path/to/video.mp4", quality="medium")
print(f"Splat saved to: {output}")
```

Quality presets:
- `"test"`: Quick test run (1k steps, 5% frame extraction, sift_features=256)
- `"low"`: Fast preview (7k steps, 5% frame extraction, sift_features=2048)
- `"medium"`: Balanced quality (15k steps, 10% frame extraction, sift_features=4096) **[default]**
- `"high"`: High quality (28k steps, 15% frame extraction, sift_features=8192)
- `"ultra"`: Maximum quality (35k steps, 30% frame extraction, sift_features=16384)

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