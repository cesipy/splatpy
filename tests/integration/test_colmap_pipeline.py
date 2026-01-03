"""Integration tests for COLMAP pipeline - ROBUSTNESS FOCUS."""
import pytest
from pathlib import Path
from splatpy.incremental_pipeline import COLMAP_Processor
from tests.fixtures.synthetic_data import create_synthetic_video


@pytest.mark.integration
class TestCOLMAPPipelineIntegration:
    """Test full COLMAP processing pipeline with synthetic data."""

    def test_full_pipeline_with_synthetic_video(
        self,
        temp_dir,
        synthetic_video,
        mock_all_pycolmap,
        mocker
    ):
        """Test complete COLMAP pipeline with synthetic video."""
        # Mock sqlite3 for match counting
        mock_cursor = mocker.MagicMock()
        mock_cursor.fetchone.return_value = (50,)
        mock_conn = mocker.MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = mocker.MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = mocker.MagicMock()
        mocker.patch("sqlite3.connect", return_value=mock_conn)

        # Run pipeline
        processor = COLMAP_Processor(save_dir=str(temp_dir / "colmap"))
        processor.create_colmap(
            str(synthetic_video),
            frames_modulo=10,
            mode="sequential"
        )

        # Verify directories were created
        assert Path(processor.frame_path).exists()
        assert Path(processor.output_path_sparse).exists()

        # Verify pycolmap operations were called
        mock_all_pycolmap["extract_features"].assert_called()
        mock_all_pycolmap["match_sequential"].assert_called()
        mock_all_pycolmap["incremental_mapping"].assert_called()

    def test_pipeline_with_different_frames_modulo(
        self,
        temp_dir,
        synthetic_video,
        mock_all_pycolmap,
        mocker
    ):
        """Test pipeline with various frame extraction rates."""
        mock_cursor = mocker.MagicMock()
        mock_cursor.fetchone.return_value = (30,)
        mock_conn = mocker.MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = mocker.MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = mocker.MagicMock()
        mocker.patch("sqlite3.connect", return_value=mock_conn)

        processor = COLMAP_Processor(save_dir=str(temp_dir / "colmap"))

        # Test with high frame modulo (few frames)
        processor.create_colmap(
            str(synthetic_video),
            frames_modulo=20,  # Only ~1-2 frames from 30-frame video
            mode="sequential"
        )

        # Should complete without error
        assert Path(processor.frame_path).exists()

    def test_sequential_vs_exhaustive_matching(
        self,
        temp_dir,
        synthetic_video,
        mock_all_pycolmap,
        mocker
    ):
        """Test both sequential and exhaustive matching modes."""
        mock_cursor = mocker.MagicMock()
        mock_cursor.fetchone.return_value = (40,)
        mock_conn = mocker.MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = mocker.MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = mocker.MagicMock()
        mocker.patch("sqlite3.connect", return_value=mock_conn)

        # Test sequential
        processor = COLMAP_Processor(save_dir=str(temp_dir / "colmap_seq"))
        processor.create_colmap(
            str(synthetic_video),
            frames_modulo=10,
            mode="sequential"
        )
        mock_all_pycolmap["match_sequential"].assert_called()

        # Reset mocks
        for mock_func in mock_all_pycolmap.values():
            if hasattr(mock_func, "reset_mock"):
                mock_func.reset_mock()

        # Test exhaustive
        processor = COLMAP_Processor(save_dir=str(temp_dir / "colmap_exh"))
        processor.create_colmap(
            str(synthetic_video),
            frames_modulo=10,
            mode="exhaustive"
        )
        mock_all_pycolmap["match_exhaustive"].assert_called()

    @pytest.mark.slow
    def test_pipeline_with_very_short_video(
        self,
        temp_dir,
        mock_all_pycolmap,
        mocker
    ):
        """Test pipeline with very short video (3 frames)."""
        # Create very short video
        short_video = temp_dir / "short_video.mp4"
        create_synthetic_video(short_video, num_frames=3)

        mock_cursor = mocker.MagicMock()
        mock_cursor.fetchone.return_value = (2,)  # Very few matches
        mock_conn = mocker.MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = mocker.MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = mocker.MagicMock()
        mocker.patch("sqlite3.connect", return_value=mock_conn)

        processor = COLMAP_Processor(save_dir=str(temp_dir / "colmap"))

        # Should handle short video without crashing
        processor.create_colmap(
            str(short_video),
            frames_modulo=1,  # Extract all 3 frames
            mode="sequential"
        )

        assert Path(processor.frame_path).exists()

    def test_pipeline_cleanup_after_completion(
        self,
        temp_dir,
        synthetic_video,
        mock_all_pycolmap,
        mocker
    ):
        """Test that cleanup works after successful pipeline run."""
        mock_cursor = mocker.MagicMock()
        mock_cursor.fetchone.return_value = (50,)
        mock_conn = mocker.MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = mocker.MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = mocker.MagicMock()
        mocker.patch("sqlite3.connect", return_value=mock_conn)

        save_dir = temp_dir / "colmap"
        processor = COLMAP_Processor(save_dir=str(save_dir))

        processor.create_colmap(
            str(synthetic_video),
            frames_modulo=10,
            mode="sequential"
        )

        # Directories should exist
        assert save_dir.exists()

        # Cleanup
        processor.clean_up()

        # Directories should be removed
        assert not save_dir.exists()

    def test_pipeline_with_no_matches_edge_case(
        self,
        temp_dir,
        synthetic_video,
        mock_all_pycolmap,
        mocker
    ):
        """Test handling when feature matching finds no matches."""
        # Mock database to return 0 matches
        mock_cursor = mocker.MagicMock()
        mock_cursor.fetchone.return_value = (0,)
        mock_conn = mocker.MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = mocker.MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = mocker.MagicMock()
        mocker.patch("sqlite3.connect", return_value=mock_conn)

        processor = COLMAP_Processor(save_dir=str(temp_dir / "colmap"))

        # Should complete without error (COLMAP handles this internally)
        processor.create_colmap(
            str(synthetic_video),
            frames_modulo=10,
            mode="sequential"
        )

    @pytest.mark.slow
    def test_pipeline_with_high_frames_modulo(
        self,
        temp_dir,
        synthetic_video,
        mock_all_pycolmap,
        mocker
    ):
        """Test pipeline with very high frames_modulo (extracting only 2 frames)."""
        mock_cursor = mocker.MagicMock()
        mock_cursor.fetchone.return_value = (1,)
        mock_conn = mocker.MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = mocker.MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = mocker.MagicMock()
        mocker.patch("sqlite3.connect", return_value=mock_conn)

        processor = COLMAP_Processor(save_dir=str(temp_dir / "colmap"))

        # With 30-frame video and frames_modulo=25, only 2 frames extracted
        processor.create_colmap(
            str(synthetic_video),
            frames_modulo=25,
            mode="sequential"
        )

        # Should complete
        assert Path(processor.frame_path).exists()


@pytest.mark.integration
class TestCOLMAPErrorHandling:
    """Test COLMAP pipeline error handling and recovery."""

    def test_pipeline_handles_feature_extraction_failure(
        self,
        temp_dir,
        synthetic_video,
        mocker
    ):
        """Test handling when feature extraction fails."""
        # Mock pycolmap to raise error on feature extraction
        mocker.patch("pycolmap.extract_features", side_effect=RuntimeError("Feature extraction failed"))

        processor = COLMAP_Processor(save_dir=str(temp_dir / "colmap"))

        # Should propagate error
        with pytest.raises(RuntimeError, match="Feature extraction failed"):
            processor.create_colmap(
                str(synthetic_video),
                frames_modulo=10,
                mode="sequential"
            )

    def test_cleanup_after_pipeline_failure(
        self,
        temp_dir,
        synthetic_video,
        mocker
    ):
        """Test that cleanup works even after pipeline failure."""
        # Mock to cause failure
        mocker.patch("pycolmap.extract_features", side_effect=RuntimeError("Mock failure"))

        save_dir = temp_dir / "colmap"
        processor = COLMAP_Processor(save_dir=str(save_dir))

        # Pipeline should fail
        try:
            processor.create_colmap(
                str(synthetic_video),
                frames_modulo=10,
                mode="sequential"
            )
        except RuntimeError:
            pass

        # Cleanup should still work
        processor.clean_up()
        # Don't check if dir exists, as it may not have been fully created
