"""Unit tests for COLMAP_Processor with emphasis on robustness."""
import pytest
import numpy as np
from pathlib import Path
from splatpy.incremental_pipeline import COLMAP_Processor


@pytest.mark.unit
class TestCOLMAPProcessorInitialization:
    """Test COLMAP_Processor initialization."""

    def test_initialization_creates_directories(self, temp_dir):
        """Test processor initialization creates required directories."""
        save_dir = temp_dir / "colmap_data"
        processor = COLMAP_Processor(save_dir=str(save_dir))

        assert processor.save_dir == str(save_dir)
        assert Path(processor.frame_path).exists()
        assert Path(processor.output_path_sparse).exists()
        assert processor.frame_path == str(save_dir / "images")
        assert processor.output_path_sparse == str(save_dir / "sparse")
        assert processor.db_path == str(save_dir / "database.db")


@pytest.mark.unit
class TestExtractImages:
    """Test frame extraction from video - ROBUSTNESS FOCUS."""

    def test_extract_images_from_valid_video(
        self,
        temp_dir,
        mock_all_io
    ):
        """Test extracting frames from valid video."""
        processor = COLMAP_Processor(save_dir=str(temp_dir / "colmap"))

        # Create fake video
        video_path = temp_dir / "test.mp4"
        video_path.touch()

        frame_path = processor.extract_images(
            str(video_path),
            frames_modulo=5
        )

        # Verify frames were extracted
        assert frame_path == processor.frame_path
        assert Path(frame_path).exists()

    @pytest.mark.parametrize("extension", ["mp4", "avi", "mov", "MP4", "AVI", "MOV"])
    def test_valid_video_formats(self, temp_dir, mock_all_io, extension):
        """Test that all valid video formats are accepted."""
        processor = COLMAP_Processor(save_dir=str(temp_dir / "colmap"))

        video_path = temp_dir / f"test.{extension}"
        video_path.touch()

        # Should not raise assertion error
        processor.extract_images(str(video_path), frames_modulo=10)

    @pytest.mark.parametrize("extension", ["txt", "png", "jpg", "mkv", "webm"])
    def test_invalid_video_formats_raise_error(self, temp_dir, mock_all_io, extension):
        """Test that invalid video formats raise AssertionError."""
        processor = COLMAP_Processor(save_dir=str(temp_dir / "colmap"))

        invalid_video = temp_dir / f"test.{extension}"
        invalid_video.touch()

        with pytest.raises(AssertionError, match="unsupported video format"):
            processor.extract_images(str(invalid_video))

    def test_nonexistent_video_raises_error(self, temp_dir):
        """Test that non-existent video raises AssertionError."""
        processor = COLMAP_Processor(save_dir=str(temp_dir / "colmap"))

        with pytest.raises(AssertionError, match="does not exist"):
            processor.extract_images(str(temp_dir / "nonexistent.mp4"))

    def test_corrupted_video_raises_io_error(self, temp_dir, mocker):
        """Test that corrupted video raises IOError."""
        processor = COLMAP_Processor(save_dir=str(temp_dir / "colmap"))

        # Create fake corrupted video
        video_path = temp_dir / "corrupted.mp4"
        video_path.touch()

        # Mock VideoCapture to simulate corrupted video
        mock_capture = mocker.MagicMock()
        mock_capture.isOpened.return_value = False
        mocker.patch("cv2.VideoCapture", return_value=mock_capture)

        with pytest.raises(IOError, match="Couldn't open video file"):
            processor.extract_images(str(video_path))

    def test_frames_modulo_logic(self, temp_dir, mocker):
        """Test frame extraction sampling logic with frames_modulo."""
        processor = COLMAP_Processor(save_dir=str(temp_dir / "colmap"))

        video_path = temp_dir / "test.mp4"
        video_path.touch()

        # Mock VideoCapture to return specific number of frames
        frames_returned = []

        class MockVideoCapture:
            def __init__(self, path):
                self.frame_idx = 0
                self.total_frames = 30

            def isOpened(self):
                return True

            def read(self):
                if self.frame_idx < self.total_frames:
                    self.frame_idx += 1
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    return True, frame
                return False, None

            def release(self):
                pass

        mocker.patch("cv2.VideoCapture", MockVideoCapture)

        # Mock cv2.imwrite to track which frames are saved
        saved_frames = []

        def mock_imwrite(path, frame):
            saved_frames.append(path)

        mocker.patch("cv2.imwrite", side_effect=mock_imwrite)

        # Extract every 5th frame from 30 frames
        processor.extract_images(str(video_path), frames_modulo=5)

        # Should save frames at indices 4, 9, 14, 19, 24, 29 (0-indexed)
        # That's 6 frames total
        assert len(saved_frames) == 6

    def test_empty_video_no_frames(self, temp_dir, mocker):
        """Test handling of video with zero frames."""
        processor = COLMAP_Processor(save_dir=str(temp_dir / "colmap"))

        video_path = temp_dir / "empty.mp4"
        video_path.touch()

        class MockEmptyVideo:
            def __init__(self, path):
                pass

            def isOpened(self):
                return True

            def read(self):
                return False, None  # No frames

            def release(self):
                pass

        mocker.patch("cv2.VideoCapture", MockEmptyVideo)
        saved_frames = []
        mocker.patch("cv2.imwrite", side_effect=lambda p, f: saved_frames.append(p))

        processor.extract_images(str(video_path))

        # No frames should be saved
        assert len(saved_frames) == 0


@pytest.mark.unit
class TestFeatureExtractAndMatch:
    """Test feature extraction and matching - ROBUSTNESS FOCUS."""

    def test_feature_extract_sequential_mode(
        self,
        temp_dir,
        mock_all_pycolmap,
        mocker
    ):
        """Test feature extraction and matching in sequential mode."""
        processor = COLMAP_Processor(save_dir=str(temp_dir / "colmap"))

        # Mock sqlite3 database query
        mock_cursor = mocker.MagicMock()
        mock_cursor.fetchone.return_value = (100,)  # 100 matches
        mock_conn = mocker.MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = mocker.MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = mocker.MagicMock()
        mocker.patch("sqlite3.connect", return_value=mock_conn)

        processor.feature_extract_and_match(
            images_path=str(temp_dir),
            mode="sequential"
        )

        # Verify sequential matching was called
        mock_all_pycolmap["match_sequential"].assert_called_once()
        mock_all_pycolmap["match_exhaustive"].assert_not_called()

    def test_feature_extract_exhaustive_mode(
        self,
        temp_dir,
        mock_all_pycolmap,
        mocker
    ):
        """Test feature extraction and matching in exhaustive mode."""
        processor = COLMAP_Processor(save_dir=str(temp_dir / "colmap"))

        # Mock sqlite3
        mock_cursor = mocker.MagicMock()
        mock_cursor.fetchone.return_value = (50,)
        mock_conn = mocker.MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = mocker.MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = mocker.MagicMock()
        mocker.patch("sqlite3.connect", return_value=mock_conn)

        processor.feature_extract_and_match(
            images_path=str(temp_dir),
            mode="exhaustive"
        )

        # Verify exhaustive matching was called
        mock_all_pycolmap["match_exhaustive"].assert_called_once()
        mock_all_pycolmap["match_sequential"].assert_not_called()

    def test_no_matches_found(
        self,
        temp_dir,
        mock_all_pycolmap,
        mocker
    ):
        """Test handling when no feature matches are found."""
        processor = COLMAP_Processor(save_dir=str(temp_dir / "colmap"))

        # Mock database to return 0 matches
        mock_cursor = mocker.MagicMock()
        mock_cursor.fetchone.return_value = (0,)
        mock_conn = mocker.MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = mocker.MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = mocker.MagicMock()
        mocker.patch("sqlite3.connect", return_value=mock_conn)

        # Should not raise, just log 0 matches
        processor.feature_extract_and_match(
            images_path=str(temp_dir),
            mode="sequential"
        )


@pytest.mark.unit
class TestReconstruct:
    """Test COLMAP reconstruction."""

    def test_reconstruct_creates_output_dir(
        self,
        temp_dir,
        mock_all_pycolmap
    ):
        """Test that reconstruct creates output directory."""
        processor = COLMAP_Processor(save_dir=str(temp_dir / "colmap"))

        # Remove sparse directory
        import shutil
        if Path(processor.output_path_sparse).exists():
            shutil.rmtree(processor.output_path_sparse)

        processor.reconstruct(images_path=str(temp_dir))

        # Verify directory was created
        assert Path(processor.output_path_sparse).exists()

    def test_reconstruct_returns_reconstruction(
        self,
        temp_dir,
        mock_all_pycolmap
    ):
        """Test that reconstruct returns reconstruction object."""
        processor = COLMAP_Processor(save_dir=str(temp_dir / "colmap"))

        result = processor.reconstruct(images_path=str(temp_dir))

        # Verify reconstruction object is returned
        assert result is not None
        assert hasattr(result, "cameras")
        assert hasattr(result, "images")
        assert hasattr(result, "points3D")


@pytest.mark.unit
class TestCleanup:
    """Test cleanup operations - ROBUSTNESS FOCUS."""

    def test_clean_up_removes_all_files(self, temp_dir):
        """Test that cleanup removes all generated files."""
        save_dir = temp_dir / "colmap"
        processor = COLMAP_Processor(save_dir=str(save_dir))

        # Create some fake files
        (save_dir / "database.db").touch()
        (save_dir / "images" / "frame_00000.png").touch()
        (save_dir / "sparse" / "cameras.txt").touch()

        processor.clean_up()

        # Verify all files are removed
        assert not (save_dir / "database.db").exists()
        assert not (save_dir / "images").exists()
        assert not (save_dir / "sparse").exists()
        assert not save_dir.exists()

    def test_clean_up_handles_missing_files(self, temp_dir):
        """Test that cleanup handles already-deleted files gracefully."""
        save_dir = temp_dir / "colmap"
        processor = COLMAP_Processor(save_dir=str(save_dir))

        # Remove directories manually
        import shutil
        if save_dir.exists():
            shutil.rmtree(save_dir)

        # Should not raise error
        processor.clean_up()

    def test_clean_up_handles_partial_cleanup(self, temp_dir):
        """Test cleanup when only some files exist."""
        save_dir = temp_dir / "colmap"
        processor = COLMAP_Processor(save_dir=str(save_dir))

        # Remove some directories but not others
        import shutil
        if (save_dir / "images").exists():
            shutil.rmtree(save_dir / "images")

        # Database and sparse still exist, should not raise
        processor.clean_up()

        # All should be removed
        assert not save_dir.exists()


@pytest.mark.unit
class TestCreateCOLMAP:
    """Test full COLMAP pipeline orchestration."""

    def test_create_colmap_calls_all_steps(
        self,
        temp_dir,
        mock_all_pycolmap,
        mock_all_io,
        mocker
    ):
        """Test that create_colmap calls all pipeline steps."""
        processor = COLMAP_Processor(save_dir=str(temp_dir / "colmap"))

        video_path = temp_dir / "test.mp4"
        video_path.touch()

        # Mock sqlite3
        mock_cursor = mocker.MagicMock()
        mock_cursor.fetchone.return_value = (50,)
        mock_conn = mocker.MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = mocker.MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = mocker.MagicMock()
        mocker.patch("sqlite3.connect", return_value=mock_conn)

        processor.create_colmap(
            str(video_path),
            frames_modulo=10,
            mode="sequential"
        )

        # Verify all steps were called
        mock_all_pycolmap["extract_features"].assert_called()
        mock_all_pycolmap["match_sequential"].assert_called()
        mock_all_pycolmap["incremental_mapping"].assert_called()
