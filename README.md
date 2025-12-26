# Splatpy


## TODOs
- [ ] support for pycolmap-cuda-12, needs build from source.
- [ ] automatically detect num of frames extracted from video

## Installation

A CUDA installation is required. This project uses `uv` for dependency management and was tested on Python 3.10.
```bash
uv sync --python 3.10
```

To run:
```bash
uv run python src/trainer.py
```

Or activate the venv manually:
```bash
source .venv/bin/activate
python src/trainer.py
```