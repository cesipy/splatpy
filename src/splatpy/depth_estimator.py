"""Depth map estimation for point cloud densification.

Uses Depth Anything V2 (via HuggingFace transformers) to estimate dense depth
maps from extracted frames, scale-aligns them to metric COLMAP depths, and
unprojects to 3D for a denser Gaussian initialization.

Optional dependency - install with:
    pip install transformers accelerate
"""

import os
import numpy as np


def _require_transformers():
    try:
        import transformers  # noqa: F401
    except ImportError:
        raise ImportError(
            "depth prior requires the 'transformers' package.\n"
            "Install it with:\n"
            "  pip install transformers accelerate"
        )


def _load_depth_pipe(device: str):
    from transformers import pipeline as hf_pipeline
    print("Loading Depth Anything V2 (first run downloads ~300 MB)...")
    pipe = hf_pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=0 if device == "cuda" else -1,
    )
    return pipe


def _estimate_depth(rgb: np.ndarray, pipe, target_h: int, target_w: int) -> np.ndarray:
    """Run depth estimation and resize output to match the input image dimensions."""
    import cv2
    from PIL import Image

    result = pipe(Image.fromarray(rgb))
    depth = result["predicted_depth"].squeeze().cpu().numpy().astype(np.float32)

    # Resize to match original image if the model changed the resolution
    if depth.shape != (target_h, target_w):
        depth = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    return depth


def _scale_align(
    depth_map: np.ndarray,
    pixels_uv: np.ndarray,
    colmap_depths: np.ndarray,
) -> float | None:
    """Compute median scale factor to align relative depth map to metric COLMAP depths."""
    h, w = depth_map.shape
    us = np.clip(pixels_uv[:, 0].astype(int), 0, w - 1)
    vs = np.clip(pixels_uv[:, 1].astype(int), 0, h - 1)
    sampled = depth_map[vs, us]
    valid = (sampled > 1e-6) & (colmap_depths > 0)
    if valid.sum() < 3:
        return None
    ratios = colmap_depths[valid] / sampled[valid]
    return float(np.median(ratios))


def _get_K(camera) -> np.ndarray:
    """Extract 3x3 intrinsics matrix from a pycolmap camera."""
    if hasattr(camera, "focal_length_x"):
        fx, fy = camera.focal_length_x, camera.focal_length_y
    else:
        fx = fy = camera.focal_length
    cx, cy = camera.principal_point_x, camera.principal_point_y
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def _get_dist_coeffs(camera) -> np.ndarray | None:
    """Extract OpenCV-format distortion coefficients from a pycolmap camera.

    Returns:
        Coefficient array compatible with cv2.undistort, or None for fisheye
        cameras (which need cv2.fisheye and are skipped).
    """
    params = camera.params
    model_name = camera.model_name if hasattr(camera, "model_name") else str(camera.model)

    if model_name in ("SIMPLE_PINHOLE", "PINHOLE"):
        return np.zeros(4)
    elif model_name == "SIMPLE_RADIAL":
        # params: [f, cx, cy, k1]
        return np.array([params[3], 0.0, 0.0, 0.0])
    elif model_name == "RADIAL":
        # params: [f, cx, cy, k1, k2]
        return np.array([params[3], params[4], 0.0, 0.0])
    elif model_name == "OPENCV":
        # params: [fx, fy, cx, cy, k1, k2, p1, p2]
        return np.array([params[4], params[5], params[6], params[7]])
    elif model_name == "FULL_OPENCV":
        # params: [fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6]
        return np.array(params[4:12])
    elif "FISHEYE" in model_name:
        return None  # Requires cv2.fisheye — skip these images
    else:
        return np.zeros(4)  # Unknown model, assume no distortion


def _undistort(
    rgb: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    uv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Undistort the image and the corresponding distorted pixel coordinates.

    Args:
        rgb:  (H, W, 3) distorted image.
        K:    3x3 camera intrinsics.
        dist: OpenCV distortion coefficients.
        uv:   (N, 2) distorted pixel coordinates (e.g. COLMAP p2d.xy).

    Returns:
        rgb_undist: (H, W, 3) undistorted image.
        uv_undist:  (N, 2) pixel coordinates in the undistorted image.
    """
    import cv2

    rgb_undist = cv2.undistort(rgb, K, dist)
    uv_arr = uv.astype(np.float32).reshape(-1, 1, 2)
    uv_undist = cv2.undistortPoints(uv_arr, K, dist, P=K).reshape(-1, 2)
    return rgb_undist, uv_undist


def _get_c2w(image) -> np.ndarray:
    """Extract 4x4 camera-to-world matrix from a pycolmap image."""
    cam_from_world = image.cam_from_world()
    R = cam_from_world.rotation.matrix()
    t = cam_from_world.translation
    c2w = np.eye(4)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = -R.T @ t
    return c2w


def _unproject(
    depth: np.ndarray,
    rgb: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
    stride: int,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Unproject a metric depth map to world-space 3D points with colors."""
    h, w = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    us = np.arange(0, w, stride)
    vs = np.arange(0, h, stride)
    uu, vv = np.meshgrid(us, vs)

    d = depth[vv, uu]
    valid = (d > 0.05) & (d < max_depth)
    d = d[valid]
    uu = uu[valid].astype(float)
    vv = vv[valid].astype(float)

    x_cam = (uu - cx) / fx * d
    y_cam = (vv - cy) / fy * d
    pts_cam = np.stack([x_cam, y_cam, d, np.ones_like(d)], axis=1)  # (N, 4)
    pts_world = (c2w @ pts_cam.T).T[:, :3]                          # (N, 3)
    colors = rgb[vv.astype(int), uu.astype(int)]                     # (N, 3)

    return pts_world, colors


def densify_point_cloud(
    reconstruction,
    images_dir: str,
    device: str = "cuda",
    stride: int = 8,
    max_points: int = 200_000,
    max_depth: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Densify a COLMAP sparse point cloud using Depth Anything V2.

    For each reconstructed image:
      1. Estimate depth with Depth Anything V2.
      2. Scale-align to metric using visible COLMAP sparse points.
      3. Unproject to world-space 3D.

    Args:
        reconstruction: pycolmap.Reconstruction object (already loaded).
        images_dir:     Path to the extracted frame images.
        device:         "cuda" or "cpu".
        stride:         Pixel stride when unprojecting (lower = denser, slower).
        max_points:     Cap on total output points; random subsample if exceeded.
        max_depth:      Discard points beyond this distance (in COLMAP metric units).

    Returns:
        points: (N, 3) float32 world-space coordinates.
        colors: (N, 3) float32 RGB values in [0, 1].
    """
    import cv2
    _require_transformers()

    depth_pipe = _load_depth_pipe(device)

    all_points = []
    all_colors = []
    images = reconstruction.images
    n = len(images)

    print(f"Estimating depth for {n} images...")

    for i, (img_id, image) in enumerate(images.items()):
        img_path = os.path.join(images_dir, image.name)
        if not os.path.exists(img_path):
            continue

        bgr = cv2.imread(img_path)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        camera = reconstruction.cameras[image.camera_id]
        K = _get_K(camera)
        dist = _get_dist_coeffs(camera)

        # Skip fisheye cameras (dist == None) — cv2.undistort doesn't handle them
        if dist is None:
            continue

        # Collect visible COLMAP 3D points for scale alignment (distorted coords)
        cam_from_world = image.cam_from_world()
        R = cam_from_world.rotation.matrix()
        t = cam_from_world.translation
        uv_list, d_list = [], []
        for p2d in image.points2D:
            if not p2d.has_point3D():
                continue
            pt3d = reconstruction.points3D[p2d.point3D_id].xyz
            d_cam = (R @ pt3d + t)[2]
            if d_cam > 0:
                uv_list.append(p2d.xy)
                d_list.append(d_cam)

        if len(uv_list) < 3:
            continue

        # Undistort image and keypoints so that depth estimation runs on a
        # clean pinhole image and the K matrix is valid for unprojection
        uv_arr = np.array(uv_list, dtype=np.float32)
        if np.any(dist != 0):
            rgb_input, uv_aligned = _undistort(rgb, K, dist, uv_arr)
        else:
            rgb_input, uv_aligned = rgb, uv_arr

        depth_map = _estimate_depth(rgb_input, depth_pipe, target_h=h, target_w=w)

        scale = _scale_align(depth_map, uv_aligned, np.array(d_list))
        if scale is None or scale <= 0:
            continue

        c2w = _get_c2w(image)
        pts, cols = _unproject(depth_map * scale, rgb_input, K, c2w, stride=stride, max_depth=max_depth)

        if len(pts) > 0:
            all_points.append(pts)
            all_colors.append(cols)

        print(f"  [{i + 1}/{n}] {image.name}: {len(pts):,} points")

    if not all_points:
        print("Warning: depth estimation produced no points, falling back to COLMAP sparse only.")
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    points = np.concatenate(all_points, axis=0).astype(np.float32)
    colors = np.concatenate(all_colors, axis=0).astype(np.float32) / 255.0

    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points, colors = points[idx], colors[idx]

    print(f"Dense point cloud: {len(points):,} points total")
    return points, colors
