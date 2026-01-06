"""GPU mocking utilities for CPU-only testing."""
import torch
import numpy as np


def mock_gsplat_rendering(mocker):
    """Mock gsplat.rendering.rasterization for CPU testing.

    Args:
        mocker: pytest-mock fixture

    Returns:
        Mock rasterization function
    """
    def mock_rasterize(*args, **kwargs):
        """CPU-compatible rasterization mock."""
        height = kwargs.get("height", 480)
        width = kwargs.get("width", 640)

        # Get viewmats to determine batch size
        viewmats = kwargs.get("viewmats")
        if viewmats is not None:
            batch_size = viewmats.shape[0]
        else:
            batch_size = 1

        # Get number of Gaussians from means
        means = kwargs.get("means")
        num_gaussians = len(means) if means is not None else 1000

        # Create outputs that maintain gradient flow from inputs
        # Use a trivial computation involving the input parameters to ensure gradients flow
        means_input = kwargs.get("means")
        colors_input = kwargs.get("colors")

        # Create renders with gradient flow from inputs
        if means_input is not None and means_input.requires_grad:
            # Trivial computation: sum all means and add to random base
            # This maintains gradient connection
            base_render = torch.rand(batch_size, height, width, 3, requires_grad=False)
            mean_contribution = means_input.mean() * 0.0  # multiplied by 0 so it doesn't affect values
            renders = base_render + mean_contribution
        else:
            renders = torch.rand(batch_size, height, width, 3)

        if colors_input is not None and colors_input.requires_grad:
            # Add color contribution (also scaled by 0 to not affect output)
            color_contribution = colors_input.mean() * 0.0
            renders = renders + color_contribution

        alphas = torch.rand(batch_size, height, width, 1)

        # Create gradient tracking tensors for strategy
        if means_input is not None and means_input.requires_grad:
            # Create means2d with gradient flow
            means2d_base = torch.rand(batch_size, num_gaussians, 2, requires_grad=False)
            means2d = means2d_base + means_input[:, :2].mean() * 0.0
            means2d = means2d.detach().requires_grad_(True)
        else:
            means2d = torch.rand(batch_size, num_gaussians, 2)

        radii = torch.randint(0, 10, (num_gaussians,))
        gaussian_ids = torch.arange(num_gaussians)

        info = {
            "num_gaussians": num_gaussians,
            "radii": radii,
            "means2d": means2d,
            "width": width,
            "height": height,
            "n_cameras": batch_size,
            "gaussian_ids": gaussian_ids,
        }
        return renders, alphas, info

    # Patch at the location where it's imported in trainer.py
    return mocker.patch("splatpy.trainer.rasterization", side_effect=mock_rasterize)


def mock_lpips_loss(mocker):
    """Mock lpips.LPIPS loss function.

    Args:
        mocker: pytest-mock fixture

    Returns:
        Mock LPIPS class
    """
    class MockLPIPS:
        def __init__(self, *args, **kwargs):
            self.net = kwargs.get("net", "alex")

        def to(self, device):
            return self

        def __call__(self, x, y):
            # Return a small positive loss value
            batch_size = x.shape[0] if len(x.shape) > 3 else 1
            return torch.tensor(0.1).expand(batch_size)

    # Patch both the module and where it's used in trainer
    mocker.patch("lpips.LPIPS", MockLPIPS)
    return mocker.patch("splatpy.trainer.lpips.LPIPS", MockLPIPS)


def mock_fused_ssim(mocker):
    """Mock fused_ssim loss function.

    Args:
        mocker: pytest-mock fixture

    Returns:
        Mock fused_ssim function
    """
    def mock_ssim(x, y, **kwargs):
        # Return high SSIM (close to 1.0)
        return torch.tensor(0.9)

    # Patch at the location where it's imported in trainer.py
    return mocker.patch("splatpy.trainer.ssim", side_effect=mock_ssim)


def mock_gsplat_export(mocker):
    """Mock gsplat.export_splats function.

    Args:
        mocker: pytest-mock fixture

    Returns:
        Mock export function
    """
    def mock_export(*args, **kwargs):
        # Create a minimal valid PLY file
        save_to = kwargs.get("save_to")
        if save_to:
            from pathlib import Path
            Path(save_to).parent.mkdir(parents=True, exist_ok=True)
            # Write minimal PLY header and data so file is not empty
            with open(save_to, 'w') as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write("element vertex 1\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("end_header\n")
                f.write("0.0 0.0 0.0\n")

    # Patch both the module and where it's used in trainer
    mocker.patch("gsplat.export_splats", side_effect=mock_export)
    return mocker.patch("splatpy.trainer.gsplat.export_splats", side_effect=mock_export)


def mock_pycolmap_operations(mocker):
    """Mock all pycolmap operations.

    Args:
        mocker: pytest-mock fixture

    Returns:
        Dict of mocked functions
    """
    from tests.fixtures.mock_colmap_data import create_mock_reconstruction

    # Mock feature extraction
    mock_extract = mocker.patch("pycolmap.extract_features")

    # Mock matching
    mock_match_exhaustive = mocker.patch("pycolmap.match_exhaustive")
    mock_match_sequential = mocker.patch("pycolmap.match_sequential")

    # Mock reconstruction
    mock_recon = create_mock_reconstruction()
    mock_mapping = mocker.patch("pycolmap.incremental_mapping", return_value=mock_recon)
    mock_recon_class = mocker.patch("pycolmap.Reconstruction", return_value=mock_recon)

    # Mock options classes
    mocker.patch("pycolmap.FeatureExtractionOptions")
    mocker.patch("pycolmap.FeatureMatchingOptions")
    mocker.patch("pycolmap.SequentialPairingOptions")

    return {
        "extract_features": mock_extract,
        "match_exhaustive": mock_match_exhaustive,
        "match_sequential": mock_match_sequential,
        "incremental_mapping": mock_mapping,
        "Reconstruction": mock_recon_class,
        "reconstruction": mock_recon,
    }


def mock_video_io(mocker, num_frames=30):
    """Mock cv2.VideoCapture for video reading.

    Args:
        mocker: pytest-mock fixture
        num_frames: Number of frames to return

    Returns:
        Mock VideoCapture class
    """
    class MockVideoCapture:
        def __init__(self, path):
            self.path = path
            self.frame_idx = 0
            self.total_frames = num_frames
            self.width = 640
            self.height = 480

        def isOpened(self):
            return True

        def read(self):
            if self.frame_idx < self.total_frames:
                self.frame_idx += 1
                frame = np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)
                return True, frame
            return False, None

        def get(self, prop):
            """Get video property (mimics cv2.VideoCapture.get)."""
            # prop 7 is CV_CAP_PROP_FRAME_COUNT
            if prop == 7:
                return self.total_frames
            return None

        def set(self, prop, value):
            """Set video property (mimics cv2.VideoCapture.set)."""
            # prop 1 is CV_CAP_PROP_POS_FRAMES
            if prop == 1:
                self.frame_idx = int(value)

        def release(self):
            pass

    return mocker.patch("cv2.VideoCapture", MockVideoCapture)


def mock_image_io(mocker):
    """Mock imageio for image reading/writing.

    Args:
        mocker: pytest-mock fixture

    Returns:
        Dict of mocked functions
    """
    def mock_imread(path):
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def mock_imwrite(path, data):
        pass

    def mock_mimsave(path, frames, **kwargs):
        pass

    mock_read = mocker.patch("imageio.v2.imread", side_effect=mock_imread)
    mock_write = mocker.patch("imageio.v2.imwrite", side_effect=mock_imwrite)
    mock_video = mocker.patch("imageio.mimsave", side_effect=mock_mimsave)
    mocker.patch("cv2.imwrite", return_value=True)

    return {
        "imread": mock_read,
        "imwrite": mock_write,
        "mimsave": mock_video,
    }


def mock_gsplat_strategies(mocker):
    """Mock gsplat strategy classes for testing.

    Args:
        mocker: pytest-mock fixture

    Returns:
        Dict of mocked strategy classes
    """
    class MockStrategy:
        """Mock strategy that does nothing but tracks calls."""

        def __init__(self, *args, **kwargs):
            self.refine_start_iter = kwargs.get('refine_start_iter', 500)
            self.refine_stop_iter = kwargs.get('refine_stop_iter', 15000)
            self.refine_every = kwargs.get('refine_every', 100)
            self.reset_every = kwargs.get('reset_every', 3000)
            self.absgrad = False

        def initialize_state(self, scene_scale=1.0):
            """Initialize empty state dict."""
            return {}

        def step_pre_backward(self, params, optimizers, state, step, info):
            """Mock pre-backward step - do nothing."""
            pass

        def step_post_backward(self, params, optimizers, state, step, info, packed=False):
            """Mock post-backward step - do nothing."""
            pass

    # Mock both DefaultStrategy and MCMCStrategy
    mock_default = mocker.patch("splatpy.trainer.DefaultStrategy", MockStrategy)
    mock_mcmc = mocker.patch("splatpy.trainer.MCMCStrategy", MockStrategy)

    return {
        "DefaultStrategy": mock_default,
        "MCMCStrategy": mock_mcmc,
    }
