"""Generate mock COLMAP reconstruction data for testing."""
import numpy as np
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class MockPoint3D:
    """Mock COLMAP Point3D."""
    xyz: np.ndarray
    color: np.ndarray
    error: float
    track: "MockTrack"


@dataclass
class MockTrack:
    """Mock COLMAP track."""
    elements: List["MockTrackElement"]


@dataclass
class MockTrackElement:
    """Mock COLMAP track element."""
    image_id: int
    point2D_idx: int


@dataclass
class MockImage:
    """Mock COLMAP Image."""
    name: str
    camera_id: int
    _cam_from_world: "MockPose"

    def cam_from_world(self):
        return self._cam_from_world


@dataclass
class MockPose:
    """Mock COLMAP Pose."""
    _rotation: np.ndarray
    _translation: np.ndarray

    @property
    def rotation(self):
        return MockRotation(self._rotation)

    @property
    def translation(self):
        return self._translation


@dataclass
class MockRotation:
    """Mock COLMAP Rotation."""
    _matrix: np.ndarray

    def matrix(self):
        return self._matrix


@dataclass
class MockCamera:
    """Mock COLMAP Camera."""
    model: "MockCameraModel"
    focal_length_x: float
    focal_length_y: float
    principal_point_x: float
    principal_point_y: float
    width: int
    height: int
    params: np.ndarray


@dataclass
class MockCameraModel:
    """Mock COLMAP camera model."""
    name: str = "PINHOLE"


def create_mock_reconstruction(num_images=10, num_points=100):
    """Create a mock COLMAP reconstruction.

    Args:
        num_images: Number of camera images
        num_points: Number of 3D points

    Returns:
        Mock reconstruction object with realistic structure
    """
    reconstruction = type('Reconstruction', (), {})()

    # Create mock cameras
    cameras = {}
    camera = MockCamera(
        model=MockCameraModel(),
        focal_length_x=800.0,
        focal_length_y=800.0,
        principal_point_x=400.0,
        principal_point_y=300.0,
        width=800,
        height=600,
        params=np.array([])
    )
    cameras[1] = camera
    reconstruction.cameras = cameras

    # Create mock images with realistic camera poses
    images = {}
    for i in range(num_images):
        # Create a circular camera trajectory
        angle = 2 * np.pi * i / num_images
        radius = 5.0

        # Camera position
        cam_x = radius * np.cos(angle)
        cam_y = radius * np.sin(angle)
        cam_z = 0.0

        # Rotation matrix (camera looking at origin)
        # Simple rotation around Z axis
        rot = np.array([
            [np.cos(angle + np.pi/2), -np.sin(angle + np.pi/2), 0],
            [np.sin(angle + np.pi/2), np.cos(angle + np.pi/2), 0],
            [0, 0, 1]
        ], dtype=np.float32)

        trans = np.array([cam_x, cam_y, cam_z], dtype=np.float32)

        pose = MockPose(_rotation=rot, _translation=trans)

        image = MockImage(
            name=f"frame_{i:05d}.png",
            camera_id=1,
            _cam_from_world=pose
        )
        images[i] = image
    reconstruction.images = images

    # Create mock 3D points
    points3D = {}
    for i in range(num_points):
        # Random points in a sphere
        theta = np.random.rand() * 2 * np.pi
        phi = np.random.rand() * np.pi
        r = np.random.rand() * 2.0

        xyz = np.array([
            r * np.sin(phi) * np.cos(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi)
        ], dtype=np.float32)

        color = np.random.randint(0, 255, 3, dtype=np.uint8)

        # Create mock track (point visible in 3-5 images)
        num_views = min(np.random.randint(3, 6), num_images)
        view_ids = np.random.choice(num_images, num_views, replace=False)

        track_elements = [
            MockTrackElement(image_id=int(j), point2D_idx=i)
            for j in view_ids
        ]
        track = MockTrack(elements=track_elements)

        point = MockPoint3D(
            xyz=xyz,
            color=color,
            error=0.5,
            track=track
        )
        points3D[i] = point
    reconstruction.points3D = points3D

    return reconstruction
