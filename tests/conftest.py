"""Shared pytest fixtures for splatpy tests."""
import os
import tempfile
from pathlib import Path
import pytest
import numpy as np
import torch
from tests.fixtures.gpu_mocks import (
    mock_gsplat_rendering,
    mock_lpips_loss,
    mock_fused_ssim,
    mock_gsplat_export,
    mock_pycolmap_operations,
    mock_video_io,
    mock_image_io,
    mock_gsplat_strategies,
)
from tests.fixtures.mock_colmap_data import create_mock_reconstruction


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def device():
    """Return CPU or CUDA device based on availability."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seeds for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture(autouse=True)
def allow_cpu_for_tests(monkeypatch):
    """Allow CPU mode for all tests by setting SPLATPY_ALLOW_CPU environment variable."""
    monkeypatch.setenv("SPLATPY_ALLOW_CPU", "1")


@pytest.fixture
def sample_intrinsics():
    """Sample camera intrinsics matrix (800x600 image, f=800)."""
    return np.array([
        [800.0, 0.0, 400.0],
        [0.0, 800.0, 300.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)


@pytest.fixture
def sample_camera_pose():
    """Sample camera-to-world transformation (camera at z=-5)."""
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 3] = np.array([0, 0, -5])
    return c2w


@pytest.fixture
def sample_image():
    """Sample RGB image (480x640x3)."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


# Mock fixtures - shared by both unit and integration tests
@pytest.fixture
def mock_pycolmap_reconstruction(mocker):
    """Mock pycolmap reconstruction with realistic data."""
    return create_mock_reconstruction(num_images=10, num_points=100)


@pytest.fixture
def mock_all_gpu_operations(mocker):
    """Mock all GPU and rendering operations."""
    return {
        "rendering": mock_gsplat_rendering(mocker),
        "lpips": mock_lpips_loss(mocker),
        "ssim": mock_fused_ssim(mocker),
        "export": mock_gsplat_export(mocker),
        "strategies": mock_gsplat_strategies(mocker),
    }


@pytest.fixture
def mock_all_pycolmap(mocker):
    """Mock all pycolmap operations."""
    return mock_pycolmap_operations(mocker)


@pytest.fixture
def mock_all_io(mocker):
    """Mock all I/O operations."""
    return {
        **mock_image_io(mocker),
        "video": mock_video_io(mocker),
    }
