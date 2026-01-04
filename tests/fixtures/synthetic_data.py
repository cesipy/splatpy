"""Generate synthetic test data (videos, images, PLY files)."""
import numpy as np
import cv2
from pathlib import Path
import imageio.v2 as imageio


def create_synthetic_video(
    output_path: Path,
    num_frames: int = 30,
    width: int = 640,
    height: int = 480,
    fps: int = 30
) -> Path:
    #create a simple synthetic video for testing
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (50, 50, 50)  # gray background
        x = int((width - 100) * (i / max(num_frames - 1, 1)))
        y = height // 2 - 50
        cv2.rectangle(frame, (x, y), (x + 100, y + 100), (0, 255, 0), -1)

        out.write(frame)

    out.release()
    return output_path


def create_synthetic_images(
    output_dir: Path,
    num_images: int = 10,
    width: int = 640,
    height: int = 480
) -> Path:
    #create a set of synthetic images for COLMAP testing.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_images):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:, :] = (100, 100, 100)  # gray background

        # colored square
        x = int((width - 100) * (i / max(num_images - 1, 1)))
        y = height // 2 - 50
        img[y:y+100, x:x+100] = (0, 255, 0)  # green square

        # Save as PNG
        imageio.imwrite(output_dir / f"frame_{i:05d}.png", img)

    return output_dir


def create_dummy_ply(output_path: Path, num_points: int = 100) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Random points
    points = np.random.rand(num_points, 3).astype(np.float32)
    colors = (np.random.rand(num_points, 3) * 255).astype(np.uint8)

    # ply header
    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        # Write points
        for i in range(num_points):
            x, y, z = points[i]
            r, g, b = colors[i]
            f.write(f"{x} {y} {z} {r} {g} {b}\n")

    return output_path
