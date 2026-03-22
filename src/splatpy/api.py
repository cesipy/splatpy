import os
from typing import Optional, Literal
from . import incremental_pipeline
from .config import TrainingConfig, QualityPreset, get_quality_preset
from .trainer import Trainer
from .utils.colmap_datahandling import Parser, Dataset, test


def video_to_splat(
    video_path: str,
    quality: str = "medium",
    output_dir: str = "results",
    render_orbit: bool = True,
    orbit_frames: int = 240,
    use_depth_prior: bool = False,
) -> str:
    """Convert a video to a 3D Gaussian Splat.
    Works with different presets for speed vs. quality trade-offs.

    Args:
        video_path: Path to input video file (mp4, avi, or mov)
        quality: Quality preset - one of "low", "medium", "high", "ultra"
            - "low": Fast preview (10k steps, ~10-15min)
            - "medium": Balanced quality (30k steps, ~20-30 min) [default]
            - "high": High quality (50k steps, ~45-60 min)
            - "ultra": Maximum quality (100k steps, ~60-120 min)
        output_dir: Directory to save results (default: "res/results/")
        render_orbit: Whether to render a 360° orbit video (default: True)
        orbit_frames: Number of frames in orbit video (default: 240)

    Returns:
        Path to the final .ply file

    Example:
        >>> from splatpy import video_to_splat
        >>> output = video_to_splat("my_video.mp4", quality="high")
        >>> print(f"Splat saved to: {output}")
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video path '{video_path}' does not exist")

    preset = get_quality_preset(quality)
    print(f"\nUsing quality preset '{quality}': {preset.description}")
    print(f"  - Training steps: {preset.steps:,}")
    print(f"  - Frame extraction: {preset.extraction_rate*100:.0f}% of frames")
    # print(f"  - Image downscaling: {preset.data_factor}x")
    print(f"  - SH degree: {preset.sh_degree}\n")

    config = TrainingConfig(
        results_dir=output_dir,
        sh_degree=preset.sh_degree,
    )

    ip = incremental_pipeline.COLMAP_Processor(save_dir=config.colmap_data_dir)
    try:
        ip.clean_up()
        ip.create_colmap(
            video_path,
            extraction_rate=preset.extraction_rate,
            mode="sequential",
            sift_num_max_features=preset.sift_features,
        )

        dense_points, dense_colors = None, None
        if use_depth_prior:
            dense_points, dense_colors = ip.densify_with_depth()

        trainer = Trainer(config=config, dense_points=dense_points, dense_colors=dense_colors)
        trainer.train(preset.steps)

        if render_orbit:
            trainer.render_orbit(num_frames=orbit_frames)

        final_path = os.path.join(output_dir, "final.ply")
        return final_path

    except Exception as e:
        raise e
    finally:
        ip.clean_up()


def video_to_splat_advanced(
    video_path: str,
    output_dir: str = "results",
    training_steps: int = 25_000,
    extraction_rate: float = 0.15,
    data_factor: int = 1,
    colmap_mode: Literal["sequential", "exhaustive"] = "sequential",
    sh_degree: int = 3,
    means_lr: float = 1.6e-4,
    scales_lr: float = 5e-3,
    opacities_lr: float = 5e-2,
    quats_lr: float = 1e-3,
    render_orbit: bool = True,
    orbit_frames: int = 240,
    sift_max_num_features: int = 4096,
    custom_config: Optional[TrainingConfig] = None
) -> str:
    """Convert a video to a 3D Gaussian Splat.
    adcanced function allowing fine-tuning of training parameters.

    Either call with custom values directly (`video_to_splat_advanced(video_path, trainig_step=<val>)`)
    or call with a config (see splatpy.TrainingConfig).

    Args:
        video_path: Path to input video file (mp4, avi, or mov)
        output_dir: Directory to save results
        training_steps: Number of training iterations
        extraction_rate: Percentage of frames to extract (0.1 = 10%, 0.5 = 50%). Lower values = faster but may miss details. Higher values = better quality but slower processing.
        data_factor: Image downscaling factor (1 = full res, 2 = half res, etc.)
        colmap_mode: COLMAP matching mode - "sequential" (fast) or "exhaustive" (slow but thorough)
        sh_degree: Spherical harmonics degree for appearance (0-3, higher = more detail)
        means_lr: Learning rate for Gaussian positions
        scales_lr: Learning rate for Gaussian scales
        opacities_lr: Learning rate for Gaussian opacities
        quats_lr: Learning rate for Gaussian rotations
        render_orbit: Whether to render a 360° orbit video
        orbit_frames: Number of frames in orbit video
        custom_config: Optional custom TrainingConfig (overrides other parameters)

    Returns:
        Path to the final .ply file

    Example:
        >>> from splatpy import video_to_splat_advanced
        >>> output = video_to_splat_advanced(
        ...     "my_video.mp4",
        ...     training_steps=50_000,
        ...     extraction_rate=0.10,
        ...     data_factor=1
        ... )
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video path '{video_path}' does not exist")

    try:
        if custom_config is not None:
            config = custom_config
        else:
            config = TrainingConfig(
                data_dir="res/output/images/",
                data_factor=data_factor,
                results_dir=output_dir,
                sh_degree=sh_degree,
                means_lr=means_lr,
                scales_lr=scales_lr,
                opacities_lr=opacities_lr,
                quats_lr=quats_lr
            )
        ip = incremental_pipeline.COLMAP_Processor(save_dir=config.colmap_data_dir)
        ip.clean_up()
        ip.create_colmap(
            video_path,
            extraction_rate=extraction_rate,
            mode=colmap_mode,
            sift_num_max_features=sift_max_num_features,
        )

        trainer = Trainer(config=config)
        trainer.train(training_steps)

        if render_orbit:
            trainer.render_orbit(num_frames=orbit_frames)

        final_path = os.path.join(output_dir, "final.ply")
        return final_path

    except Exception as e:
        raise e
    finally:
        ip.clean_up()


# Legacy function name for backwards compatibility
def video_to_gsplat(path: str):
    """Legacy function - use video_to_splat() instead."""
    return video_to_splat(path, quality="ultra")

