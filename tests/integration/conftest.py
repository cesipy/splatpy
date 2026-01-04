"""Integration test specific fixtures."""
import pytest
from pathlib import Path
from tests.fixtures.synthetic_data import (
    create_synthetic_video,
    create_synthetic_images,
)


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory):
    """Create a synthetic test video (session-scoped for reuse)."""
    tmpdir = tmp_path_factory.mktemp("videos")
    video_path = tmpdir / "test_video.mp4"
    return create_synthetic_video(video_path, num_frames=30)


@pytest.fixture(scope="session")
def synthetic_images(tmp_path_factory):
    """Create synthetic test images (session-scoped for reuse)."""
    tmpdir = tmp_path_factory.mktemp("images")
    return create_synthetic_images(tmpdir, num_images=10)
