"""Real end-to-end test with actual video file."""
import os

import pytest
from pathlib import Path
from splatpy.api import video_to_splat, video_to_splat_advanced
from splatpy.config import TrainingConfig
import torch


@pytest.mark.integration
@pytest.mark.slow
class TestRealVideoWorkflow:
    """Test with real video file - minimal mocking."""

    @pytest.fixture
    def house_video(self):
        """Path to the real house video."""
        video_path = Path(__file__).parent.parent.parent / "res" / "input" / "house.mp4"
        if not video_path.exists():
            pytest.skip(f"Test video not found at {video_path}")
        return video_path

    def test_video_to_splat_real_video_minimal_steps(
        self,
        temp_dir,
        house_video,
        mocker
    ):
        """Test complete pipeline with real video (very short training for speed)."""
        # Mock backward and strategy to avoid GPU requirement in tests
        # mocker.patch.object(torch.Tensor, 'backward')
        # from gsplat.strategy import DefaultStrategy
        # mocker.patch.object(DefaultStrategy, 'step_post_backward', return_value=None)
        output_path = video_to_splat(
            str(house_video),
            quality="test",  # Low quality = fast
            output_dir=str(temp_dir / "output"),
            render_orbit=False
        )

        assert output_path is not None
        assert "final.ply" in output_path
        output_file = Path(output_path)
        assert output_file.exists()
        assert os.path.getsize(output_file) > 0

    def test_video_to_splat_advanced_real_video(
        self,
        temp_dir,
        house_video,
        mocker
    ):
        """Test advanced API with real video and custom parameters."""
        # Mock backward and strategy to avoid GPU requirement
        # mocker.patch.object(torch.Tensor, 'backward')
        # from gsplat.strategy import DefaultStrategy
        # mocker.patch.object(DefaultStrategy, 'step_post_backward', return_value=None)


        output_path = video_to_splat_advanced(
            str(house_video),
            training_steps=10,  # Very minimal for fast test
            frames_modulo=30,   # Extract few frames
            colmap_mode="sequential",
            sh_degree=2,
            render_orbit=False
        )

        # Verify output
        assert output_path is not None
        assert "final.ply" in output_path
        assert Path(output_path).exists()
        assert os.path.getsize(output_path) > 0

    def test_custom_config_real_video(
        self,
        temp_dir,
        house_video,
        mocker
    ):
        """Test with custom TrainingConfig on real video."""
        # Mock backward and strategy to avoid GPU requirement
        # mocker.patch.object(torch.Tensor, 'backward')
        # from gsplat.strategy import DefaultStrategy
        # mocker.patch.object(DefaultStrategy, 'step_post_backward', return_value=None)


        custom_config = TrainingConfig(
            sh_degree=2,
            results_dir=str(temp_dir / "custom_output"),
        )

        output_path = video_to_splat_advanced(
            str(house_video),
            custom_config=custom_config,
            training_steps=10,
            frames_modulo=30,
            render_orbit=False
        )

        # Verify output in custom directory
        assert "final.ply" in output_path
        assert Path(temp_dir / "custom_output" / "final.ply").exists()
        assert os.path.getsize(temp_dir / "custom_output" / "final.ply") > 0
