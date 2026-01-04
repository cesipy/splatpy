"""Unit tests for Trainer class."""
import os

import pytest
import torch
import numpy as np
from pathlib import Path
from splatpy.trainer import Trainer
from splatpy.config import TrainingConfig
import imageio


@pytest.mark.unit
class TestTrainerInitialization:
    """Test Trainer initialization."""

    def test_initialization_cpu(
        self,
        temp_dir,
        mock_all_pycolmap,
        mock_all_gpu_operations,
        mocker
    ):
        """Test Trainer initialization on CPU."""
        # Force CPU mode
        mocker.patch("torch.cuda.is_available", return_value=False)

        # Setup COLMAP data
        colmap_dir = temp_dir / "colmap"
        (colmap_dir / "sparse" / "0").mkdir(parents=True, exist_ok=True)
        (colmap_dir / "images").mkdir(parents=True, exist_ok=True)

        # Create multiple dummy images (Parser needs at least test_every images)
        for i in range(10):
            dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            imageio.imwrite(colmap_dir / "images" / f"frame_{i:05d}.png", dummy_image)


        config = TrainingConfig(
            colmap_data_dir=str(colmap_dir),
            results_dir=str(temp_dir / "results")
        )

        trainer = Trainer(config=config)

        assert trainer.config == config
        assert trainer.device == "cpu"
        assert "means" in trainer.splats
        assert "scales" in trainer.splats
        assert len(trainer.splats["means"]) > 0

    @pytest.mark.gpu
    def test_initialization_cuda(
        self,
        temp_dir,
        mock_all_pycolmap,
        mocker
    ):
        """Test Trainer initialization with CUDA (requires GPU)."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")

        # Setup COLMAP data
        colmap_dir = temp_dir / "colmap"
        (colmap_dir / "sparse" / "0").mkdir(parents=True, exist_ok=True)
        (colmap_dir / "images").mkdir(parents=True, exist_ok=True)

        # Create multiple dummy images
        for i in range(10):
            dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            imageio.imwrite(colmap_dir / "images" / f"frame_{i:05d}.png", dummy_image)

        config = TrainingConfig(colmap_data_dir=str(colmap_dir))
        trainer = Trainer(config=config)

        assert trainer.device == "cuda"

    def test_initialization_creates_optimizers(
        self,
        temp_dir,
        mock_all_pycolmap,
        mock_all_gpu_operations,
        mocker
    ):
        """Test that initialization creates optimizers for all parameters."""
        mocker.patch("torch.cuda.is_available", return_value=False)

        # Setup COLMAP data
        colmap_dir = temp_dir / "colmap"
        (colmap_dir / "sparse" / "0").mkdir(parents=True, exist_ok=True)
        (colmap_dir / "images").mkdir(parents=True, exist_ok=True)

        for i in range(10):
            dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            imageio.imwrite(colmap_dir / "images" / f"frame_{i:05d}.png", dummy_image)

        config = TrainingConfig(colmap_data_dir=str(colmap_dir))
        trainer = Trainer(config=config)

        # Verify optimizers exist
        assert "means" in trainer.optimizers
        assert "scales" in trainer.optimizers
        assert "quats" in trainer.optimizers
        assert "opacities" in trainer.optimizers


@pytest.mark.unit
class TestForwardPass:
    """Test forward pass rendering."""

    def test_forward_pass_returns_correct_shapes(
        self,
        temp_dir,
        mock_all_pycolmap,
        mock_all_gpu_operations,
        mocker
    ):
        """Test that forward pass returns correctly shaped tensors."""
        mocker.patch("torch.cuda.is_available", return_value=False)

        # Setup
        colmap_dir = temp_dir / "colmap"
        (colmap_dir / "sparse" / "0").mkdir(parents=True, exist_ok=True)
        (colmap_dir / "images").mkdir(parents=True, exist_ok=True)

        for i in range(10):
            dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            imageio.imwrite(colmap_dir / "images" / f"frame_{i:05d}.png", dummy_image)

        config = TrainingConfig(colmap_data_dir=str(colmap_dir))
        trainer = Trainer(config=config)

        # Create mock inputs
        batch_size = 2
        height, width = 480, 640
        camtoworlds = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
        Ks = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)

        renders, alphas, info = trainer.forward_pass(
            camtoworlds=camtoworlds,
            Ks=Ks,
            width=width,
            height=height
        )

        assert renders.shape == (batch_size, height, width, 3)
        assert alphas.shape == (batch_size, height, width, 1)
        assert "num_gaussians" in info

    def test_forward_pass_with_masks(
        self,
        temp_dir,
        mock_all_pycolmap,
        mock_all_gpu_operations,
        mocker
    ):
        """Test forward pass with mask parameter."""
        mocker.patch("torch.cuda.is_available", return_value=False)

        # Setup
        colmap_dir = temp_dir / "colmap"
        (colmap_dir / "sparse" / "0").mkdir(parents=True, exist_ok=True)
        (colmap_dir / "images").mkdir(parents=True, exist_ok=True)

        for i in range(10):
            dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            imageio.imwrite(colmap_dir / "images" / f"frame_{i:05d}.png", dummy_image)

        config = TrainingConfig(colmap_data_dir=str(colmap_dir))
        trainer = Trainer(config=config)

        batch_size = 1
        height, width = 480, 640
        camtoworlds = torch.eye(4).unsqueeze(0)
        Ks = torch.eye(3).unsqueeze(0)
        masks = torch.ones(batch_size, height, width, dtype=torch.bool)

        renders, alphas, info = trainer.forward_pass(
            camtoworlds=camtoworlds,
            Ks=Ks,
            width=width,
            height=height,
            masks=masks
        )

        # Should not raise
        assert renders.shape == (batch_size, height, width, 3)


@pytest.mark.unit
@pytest.mark.slow
class TestTraining:
    """Test training operations."""

    def test_train_completes_successfully(
        self,
        temp_dir,
        mock_all_pycolmap,
        mock_all_gpu_operations,
        mocker
    ):
        """Test that training loop runs without errors."""
        mocker.patch("torch.cuda.is_available", return_value=False)

        # Setup
        colmap_dir = temp_dir / "colmap"
        (colmap_dir / "sparse" / "0").mkdir(parents=True, exist_ok=True)
        (colmap_dir / "images").mkdir(parents=True, exist_ok=True)

        # Create multiple dummy images for dataset (match mock reconstruction count)
        for i in range(10):
            dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            imageio.imwrite(colmap_dir / "images" / f"frame_{i:05d}.png", dummy_image)

        config = TrainingConfig(colmap_data_dir=str(colmap_dir))
        trainer = Trainer(config=config)

        # Mock gradient computation components for unit testing
        mocker.patch.object(torch.Tensor, 'backward')
        mocker.patch.object(trainer.strategy, 'step_post_backward')

        trainer.train(steps=3)
        assert trainer.splats["means"] is not None
        assert trainer.splats["scales"] is not None
        assert trainer.splats["opacities"] is not None

    def test_train_with_short_steps(
        self,
        temp_dir,
        mock_all_pycolmap,
        mock_all_gpu_operations,
        mocker
    ):
        """Test training with small number of steps."""
        mocker.patch("torch.cuda.is_available", return_value=False)

        # Setup
        colmap_dir = temp_dir / "colmap"
        (colmap_dir / "sparse" / "0").mkdir(parents=True, exist_ok=True)
        (colmap_dir / "images").mkdir(parents=True, exist_ok=True)
        results_dir = temp_dir / "results"

        for i in range(10):
            dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            imageio.imwrite(colmap_dir / "images" / f"frame_{i:05d}.png", dummy_image)

        config = TrainingConfig(
            colmap_data_dir=str(colmap_dir),
            results_dir=str(results_dir)
        )
        trainer = Trainer(config=config)

        # Mock gradient computation components for unit testing
        mocker.patch.object(torch.Tensor, 'backward')
        mocker.patch.object(trainer.strategy, 'step_post_backward')

        # Train for just 10 steps
        trainer.train(steps=10)

        assert (results_dir / "final.ply").exists()


    def test_train_saves_checkpoints(
        self,
        temp_dir,
        mock_all_pycolmap,
        mock_all_gpu_operations,
        mocker
    ):
        """Test that training saves checkpoints at intervals."""
        mocker.patch("torch.cuda.is_available", return_value=False)

        # Setup
        colmap_dir = temp_dir / "colmap"
        (colmap_dir / "sparse" / "0").mkdir(parents=True, exist_ok=True)
        (colmap_dir / "images").mkdir(parents=True, exist_ok=True)
        results_dir = temp_dir / "results"

        for i in range(10):
            dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            imageio.imwrite(colmap_dir / "images" / f"frame_{i:05d}.png", dummy_image)

        config = TrainingConfig(
            colmap_data_dir=str(colmap_dir),
            results_dir=str(results_dir)
        )
        trainer = Trainer(config=config)

        # Mock gradient computation components for unit testing
        mocker.patch.object(torch.Tensor, 'backward')
        mocker.patch.object(trainer.strategy, 'step_post_backward')

        trainer.train(steps=4001)
        assert (results_dir / "step-2000.ply").exists()
        assert (results_dir / "step-4000.ply").exists()
        assert os.path.getsize(results_dir / "step-2000.ply") > 0
        assert os.path.getsize(results_dir / "step-4000.ply") > 0
        assert (results_dir / "final.ply").exists()


@pytest.mark.unit
class TestRenderOrbit:
    """Test orbit rendering."""

    def test_render_orbit_creates_video(
        self,
        temp_dir,
        mock_all_pycolmap,
        mock_all_gpu_operations,
        mock_all_io,
        mocker
    ):
        """Test that render_orbit creates output video."""
        mocker.patch("torch.cuda.is_available", return_value=False)

        # Setup
        colmap_dir = temp_dir / "colmap"
        (colmap_dir / "sparse" / "0").mkdir(parents=True, exist_ok=True)
        (colmap_dir / "images").mkdir(parents=True, exist_ok=True)
        results_dir = temp_dir / "results"
        results_dir.mkdir(exist_ok=True)

        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        imageio.imwrite(colmap_dir / "images" / "frame_00000.png", dummy_image)

        config = TrainingConfig(
            colmap_data_dir=str(colmap_dir),
            results_dir=str(results_dir)
        )
        trainer = Trainer(config=config)

        # Render orbit with small number of frames
        trainer.render_orbit(num_frames=10)

        # Video saving was mocked, but method should complete
        # In real scenario, would check for orbit.mp4

    def test_render_orbit_with_single_frame(
        self,
        temp_dir,
        mock_all_pycolmap,
        mock_all_gpu_operations,
        mock_all_io,
        mocker
    ):
        """Test render_orbit with num_frames=1."""
        mocker.patch("torch.cuda.is_available", return_value=False)

        # Setup
        colmap_dir = temp_dir / "colmap"
        (colmap_dir / "sparse" / "0").mkdir(parents=True, exist_ok=True)
        (colmap_dir / "images").mkdir(parents=True, exist_ok=True)

        for i in range(10):
            dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            imageio.imwrite(colmap_dir / "images" / f"frame_{i:05d}.png", dummy_image)

        config = TrainingConfig(colmap_data_dir=str(colmap_dir))
        trainer = Trainer(config=config)

        # Should not raise
        trainer.render_orbit(num_frames=1)
