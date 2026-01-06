"""Unit tests for API functions."""
import pytest
from pathlib import Path
from splatpy.api import video_to_splat, video_to_splat_advanced
from splatpy.config import TrainingConfig


@pytest.mark.unit
class TestVideoToSplat:
    """Test video_to_splat function."""

    def test_raises_error_for_nonexistent_video(self):
        """Test that FileNotFoundError is raised for missing video."""
        with pytest.raises(FileNotFoundError, match="Video path"):
            video_to_splat("nonexistent.mp4")

    def test_accepts_valid_quality_presets(
        self,
        temp_dir,
        mock_all_pycolmap,
        mock_all_gpu_operations,
        mock_all_io,
        mocker
    ):
        """Test that valid quality presets work."""
        # Create fake video file
        video_path = temp_dir / "test.mp4"
        video_path.touch()

        # Mock COLMAP processor
        mock_processor = mocker.MagicMock()
        mocker.patch(
            "splatpy.api.incremental_pipeline.COLMAP_Processor",
            return_value=mock_processor
        )

        # Mock Trainer
        mock_trainer = mocker.MagicMock()
        mocker.patch("splatpy.api.Trainer", return_value=mock_trainer)

        output = video_to_splat(
            str(video_path),
            quality="medium",
            output_dir=str(temp_dir / "results")
        )

        # Verify outputs
        assert "final.ply" in output
        mock_processor.create_colmap.assert_called_once()
        mock_trainer.train.assert_called_once()
        mock_processor.clean_up.assert_called()

    @pytest.mark.parametrize("quality", ["low", "medium", "high", "ultra"])
    def test_all_quality_presets(
        self,
        quality,
        temp_dir,
        mock_all_pycolmap,
        mock_all_gpu_operations,
        mocker
    ):
        """Test all quality presets."""
        video_path = temp_dir / "test.mp4"
        video_path.touch()

        # Mock dependencies
        mock_processor = mocker.MagicMock()
        mocker.patch(
            "splatpy.api.incremental_pipeline.COLMAP_Processor",
            return_value=mock_processor
        )
        mocker.patch("splatpy.api.Trainer", return_value=mocker.MagicMock())

        # Should not raise
        output = video_to_splat(str(video_path), quality=quality)
        assert "final.ply" in output

    def test_cleanup_called_on_error(
        self,
        temp_dir,
        mock_all_pycolmap,
        mocker
    ):
        """Test that cleanup is called even when error occurs."""
        video_path = temp_dir / "test.mp4"
        video_path.touch()

        # Mock COLMAP processor
        mock_processor = mocker.MagicMock()
        mock_processor.create_colmap.side_effect = RuntimeError("COLMAP failed")
        mocker.patch(
            "splatpy.api.incremental_pipeline.COLMAP_Processor",
            return_value=mock_processor
        )

        # Should raise error but still call cleanup
        with pytest.raises(RuntimeError, match="COLMAP failed"):
            video_to_splat(str(video_path), quality="low")

        mock_processor.clean_up.assert_called()

    def test_render_orbit_parameter(
        self,
        temp_dir,
        mock_all_pycolmap,
        mock_all_gpu_operations,
        mocker
    ):
        """Test render_orbit parameter controls orbit rendering."""
        video_path = temp_dir / "test.mp4"
        video_path.touch()

        mock_processor = mocker.MagicMock()
        mocker.patch(
            "splatpy.api.incremental_pipeline.COLMAP_Processor",
            return_value=mock_processor
        )

        mock_trainer = mocker.MagicMock()
        mocker.patch("splatpy.api.Trainer", return_value=mock_trainer)

        # Test with render_orbit=True
        video_to_splat(str(video_path), quality="low", render_orbit=True)
        mock_trainer.render_orbit.assert_called_once()

        # Test with render_orbit=False
        mock_trainer.reset_mock()
        video_to_splat(str(video_path), quality="low", render_orbit=False)
        mock_trainer.render_orbit.assert_not_called()


@pytest.mark.unit
class TestVideoToSplatAdvanced:
    """Test video_to_splat_advanced function."""

    def test_raises_error_for_nonexistent_video(self):
        """Test that FileNotFoundError is raised for missing video."""
        with pytest.raises(FileNotFoundError, match="Video path"):
            video_to_splat_advanced("nonexistent.mp4")

    def test_custom_training_steps(
        self,
        temp_dir,
        mock_all_pycolmap,
        mock_all_gpu_operations,
        mocker
    ):
        """Test custom training steps parameter."""
        video_path = temp_dir / "test.mp4"
        video_path.touch()

        mock_trainer = mocker.MagicMock()
        mocker.patch("splatpy.api.Trainer", return_value=mock_trainer)
        mocker.patch(
            "splatpy.api.incremental_pipeline.COLMAP_Processor",
            return_value=mocker.MagicMock()
        )

        video_to_splat_advanced(
            str(video_path),
            training_steps=5000
        )

        mock_trainer.train.assert_called_once_with(5000)

    def test_custom_config_overrides_parameters(
        self,
        temp_dir,
        mock_all_pycolmap,
        mock_all_gpu_operations,
        mocker
    ):
        """Test that custom_config overrides other parameters."""
        video_path = temp_dir / "test.mp4"
        video_path.touch()

        custom_config = TrainingConfig(sh_degree=2, data_factor=4)

        mock_processor = mocker.MagicMock()
        mocker.patch(
            "splatpy.api.incremental_pipeline.COLMAP_Processor",
            return_value=mock_processor
        )

        trainer_class_mock = mocker.MagicMock()
        mock_trainer_instance = mocker.MagicMock()
        trainer_class_mock.return_value = mock_trainer_instance
        mocker.patch("splatpy.api.Trainer", trainer_class_mock)

        video_to_splat_advanced(
            str(video_path),
            sh_degree=3,  # This should be ignored
            data_factor=1,  # This should be ignored
            custom_config=custom_config
        )

        # Verify custom config was used
        call_kwargs = trainer_class_mock.call_args[1]
        assert call_kwargs["config"].sh_degree == 2
        assert call_kwargs["config"].data_factor == 4

    def test_colmap_mode_parameter(
        self,
        temp_dir,
        mock_all_pycolmap,
        mock_all_gpu_operations,
        mocker
    ):
        """Test colmap_mode parameter (sequential vs exhaustive)."""
        video_path = temp_dir / "test.mp4"
        video_path.touch()

        mock_processor = mocker.MagicMock()
        mocker.patch(
            "splatpy.api.incremental_pipeline.COLMAP_Processor",
            return_value=mock_processor
        )
        mocker.patch("splatpy.api.Trainer", return_value=mocker.MagicMock())

        # Test sequential mode
        video_to_splat_advanced(str(video_path), colmap_mode="sequential")
        call_kwargs = mock_processor.create_colmap.call_args[1]
        assert call_kwargs["mode"] == "sequential"

        # Test exhaustive mode
        mock_processor.reset_mock()
        video_to_splat_advanced(str(video_path), colmap_mode="exhaustive")
        call_kwargs = mock_processor.create_colmap.call_args[1]
        assert call_kwargs["mode"] == "exhaustive"

    def test_extraction_rate_parameter(
        self,
        temp_dir,
        mock_all_pycolmap,
        mock_all_gpu_operations,
        mocker
    ):
        """Test extraction_rate parameter controls frame extraction."""
        video_path = temp_dir / "test.mp4"
        video_path.touch()

        mock_processor = mocker.MagicMock()
        mocker.patch(
            "splatpy.api.incremental_pipeline.COLMAP_Processor",
            return_value=mock_processor
        )
        mocker.patch("splatpy.api.Trainer", return_value=mocker.MagicMock())

        video_to_splat_advanced(str(video_path), extraction_rate=0.20)

        call_kwargs = mock_processor.create_colmap.call_args[1]
        assert call_kwargs["extraction_rate"] == 0.20
