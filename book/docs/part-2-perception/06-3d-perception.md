---
id: ch-2-06
title: 3D Perception
sidebar_position: 3
difficulty: intermediate
estimated_time: 90
prerequisites: [ch-2-04, ch-2-05]
---

# 3D Perception: Seeing the World in Three Dimensions

> *"The world is not flat, and neither should our robot's understanding of it be. Depth is not just another dimension—it's the difference between seeing and understanding."*

When a human reaches for a coffee cup, they don't calculate distances consciously. Yet somewhere in the brain, an exquisitely accurate 3D model enables the hand to arrive at precisely the right location. For robots, this seemingly effortless feat requires sophisticated 3D perception—the ability to reconstruct and understand the three-dimensional structure of the environment from sensor data.

## The Third Dimension: Why Depth Matters

### From 2D to 3D: A Paradigm Shift

A camera image captures rich visual information but fundamentally loses the depth dimension. Two objects at different distances may project to the same pixels. This ambiguity, inherent in 2D imaging, creates fundamental challenges for robot interaction with the physical world.

```
The Depth Ambiguity Problem
===========================

    Reality:                          2D Image (what camera sees):

         Small close                  ┌─────────────────────┐
           ┌───┐                      │                     │
           │   │ ←── 1 meter          │      ┌───┐          │
           └───┘                      │      │   │          │
                                      │      └───┘          │
              Large far               │                     │
         ┌─────────┐                  │  Same apparent size │
         │         │ ←── 5 meters     │                     │
         └─────────┘                  └─────────────────────┘

    Without depth information, we cannot distinguish:
    - Small nearby object vs. large distant object
    - Flat surface vs. 3D structure
    - Passable gap vs. solid obstacle
```

**Why 3D Perception is Essential:**

| Robot Task | Why Depth Required | Consequence of 2D Only |
|------------|-------------------|------------------------|
| **Grasping** | Precise hand positioning | Collision or miss |
| **Navigation** | Obstacle distance | Path planning fails |
| **Manipulation** | Object geometry | Cannot plan contacts |
| **SLAM** | Scale estimation | Map scale drift |
| **Human interaction** | Personal space | Unsafe proximity |

### Depth Sensing Technologies

The quest for depth information has driven development of diverse sensing technologies, each with distinct principles and tradeoffs.

```
Depth Sensing Methods
====================

    PASSIVE METHODS                    ACTIVE METHODS
    (Use ambient light)                (Emit their own energy)

    ┌──────────────────┐               ┌──────────────────┐
    │   Stereo Vision  │               │   Structured     │
    │   👁️         👁️  │               │     Light        │
    │    ↘       ↙    │               │   ◢◣◢◣◢◣◢◣     │
    │      object      │               │    ↓ pattern     │
    │   triangulation  │               │      object      │
    └──────────────────┘               └──────────────────┘

    ┌──────────────────┐               ┌──────────────────┐
    │   Monocular      │               │   Time of Flight │
    │   Depth (AI)     │               │      (ToF)       │
    │      👁️          │               │   )))  →→→  (((  │
    │   learned cues   │               │   measure delay  │
    └──────────────────┘               └──────────────────┘

                                       ┌──────────────────┐
                                       │   LiDAR          │
                                       │   ═══════════    │
                                       │   scanning laser │
                                       └──────────────────┘
```

**Comparison of Depth Sensing Technologies:**

| Technology | Range | Resolution | Cost | Outdoor | Key Limitation |
|------------|-------|------------|------|---------|----------------|
| **Stereo camera** | 0.5-20m | High | Low | Yes | Textureless surfaces |
| **Structured light** | 0.3-5m | Very high | Medium | No (IR interference) | Indoor only |
| **ToF camera** | 0.1-10m | Medium | Medium | Limited | Multi-path errors |
| **Spinning LiDAR** | 1-200m | Medium-High | High | Yes | Mechanical wear |
| **Solid-state LiDAR** | 1-150m | High | Medium | Yes | Limited FOV |
| **Monocular (AI)** | Any | Low-Medium | Very low | Yes | Scale ambiguity |

## Point Clouds: The Language of 3D Perception

### Understanding Point Cloud Data

A point cloud is a collection of 3D points representing surfaces in the environment. Unlike images with regular grids, point clouds are unstructured—points can be anywhere in 3D space.

```
Point Cloud Structure
====================

    Image (structured):              Point Cloud (unstructured):

    ┌─┬─┬─┬─┬─┬─┬─┬─┐                      ·    ·
    ├─┼─┼─┼─┼─┼─┼─┼─┤                   ·    ··  ·
    ├─┼─┼─┼─┼─┼─┼─┼─┤                  ·  ·    ·   ·
    ├─┼─┼─┼─┼─┼─┼─┼─┤                    · ·  · ·
    ├─┼─┼─┼─┼─┼─┼─┼─┤                     ·  ·
    ├─┼─┼─┼─┼─┼─┼─┼─┤                  ·      ·  ·
    ├─┼─┼─┼─┼─┼─┼─┼─┤                   ·  ·    ·
    └─┴─┴─┴─┴─┴─┴─┴─┘

    Regular grid                     Irregular sampling
    Pixel = (row, col, color)        Point = (x, y, z, [color, normal, ...])
    Fixed resolution                 Variable density
    2D neighborhood clear            3D neighborhood must be computed
```

**Point Cloud Data Formats:**

| Format | Extension | Features | Common Use |
|--------|-----------|----------|------------|
| **PCD** | .pcd | PCL native, binary/ASCII | ROS, research |
| **PLY** | .ply | Colors, normals, faces | 3D scanning |
| **LAS** | .las | Geospatial metadata | Surveying, mapping |
| **XYZ** | .xyz | Simple, ASCII | Quick exchange |
| **E57** | .e57 | Compressed, images | BIM, architecture |

### Point Cloud Processing Fundamentals

Point cloud processing involves operations to clean, structure, and extract meaning from 3D data.

**Loading and Basic Operations:**

Point clouds can be loaded and manipulated using libraries like Open3D, which provides efficient implementations of common operations.

```python
import numpy as np
import open3d as o3d

def load_and_explore_pointcloud(file_path: str) -> o3d.geometry.PointCloud:
    """
    Load a point cloud and display basic statistics.
    This is typically the first step in any point cloud pipeline.
    """
    # Load point cloud from file
    pcd = o3d.io.read_point_cloud(file_path)

    # Get basic statistics
    points = np.asarray(pcd.points)
    print(f"Point count: {len(points)}")
    print(f"Bounding box: {pcd.get_min_bound()} to {pcd.get_max_bound()}")
    print(f"Center: {pcd.get_center()}")

    # Check for additional attributes
    if pcd.has_colors():
        print("Has RGB colors")
    if pcd.has_normals():
        print("Has surface normals")

    return pcd

def visualize_pointcloud(pcd: o3d.geometry.PointCloud,
                         window_name: str = "Point Cloud") -> None:
    """
    Visualize point cloud with interactive controls.
    Use mouse to rotate, scroll to zoom, shift+mouse to pan.
    """
    o3d.visualization.draw_geometries(
        [pcd],
        window_name=window_name,
        point_show_normal=pcd.has_normals()
    )
```

**Downsampling for Efficiency:**

Raw point clouds often contain millions of points. Downsampling reduces computational load while preserving structure.

```python
def downsample_pointcloud(pcd: o3d.geometry.PointCloud,
                          voxel_size: float = 0.05) -> o3d.geometry.PointCloud:
    """
    Reduce point cloud density using voxel grid downsampling.

    The voxel size determines the resolution:
    - 0.01m: High detail, large data
    - 0.05m: Balanced for indoor scenes
    - 0.10m: Coarse, for large outdoor areas

    Each voxel (3D cube) is replaced by a single point
    at the centroid of all points within it.
    """
    return pcd.voxel_down_sample(voxel_size=voxel_size)
```

```
Voxel Grid Downsampling
======================

    Original Points:                 Voxel Grid Applied:

    ·  · ·   ·  ·                   ┌───┬───┬───┬───┐
      · ··  ·  ·  ·                 │ · │   │ · │   │
    ·  · ·  · ·   ·                 ├───┼───┼───┼───┤
      ·  · ·   ·                    │ · │ · │   │ · │
    ·    ·  ·  · ·                  ├───┼───┼───┼───┤
                                    │   │ · │ · │   │
    Many points                     └───┴───┴───┴───┘
    (10,000+)
                                    One point per voxel
                                    (reduced to ~100)
```

**Computing Surface Normals:**

Surface normals are vectors perpendicular to the local surface, essential for many algorithms.

```python
def compute_normals(pcd: o3d.geometry.PointCloud,
                    radius: float = 0.1,
                    max_neighbors: int = 30) -> o3d.geometry.PointCloud:
    """
    Estimate surface normals for each point.

    Normals are computed by fitting a plane to local neighborhoods.
    The radius and max_neighbors control the neighborhood size:
    - Larger values: Smoother normals, less detail
    - Smaller values: More detail, noisier
    """
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=max_neighbors
        )
    )

    # Orient normals consistently (pointing outward)
    pcd.orient_normals_consistent_tangent_plane(k=15)

    return pcd
```

### Statistical Outlier Removal

Raw sensor data often contains noise and outliers that must be filtered.

```python
def remove_outliers(pcd: o3d.geometry.PointCloud,
                    nb_neighbors: int = 20,
                    std_ratio: float = 2.0) -> o3d.geometry.PointCloud:
    """
    Remove statistical outliers from point cloud.

    Points are removed if their average distance to neighbors
    exceeds (mean + std_ratio * standard_deviation) of the dataset.

    Parameters:
    - nb_neighbors: How many neighbors to consider
    - std_ratio: Threshold multiplier (higher = keep more points)
    """
    filtered_pcd, indices = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )

    removed_count = len(pcd.points) - len(filtered_pcd.points)
    print(f"Removed {removed_count} outlier points")

    return filtered_pcd
```

```
Statistical Outlier Removal
==========================

    Before:                          After:

        ·
    ·  · ···  ·                      ·  · ···  ·
      ·····  ·                         ·····  ·
    ·  ·····  ·                      ·  ·····  ·
      ·····                            ·····
         ·
    ·

    Isolated points are              Clean point cloud
    statistical outliers             with outliers removed
```

## Plane Segmentation: Finding Flat Surfaces

### The RANSAC Algorithm

Plane segmentation identifies dominant planar surfaces like floors, tables, and walls. RANSAC (Random Sample Consensus) is the standard approach.

```
RANSAC for Plane Detection
=========================

    1. Randomly select      2. Fit plane to       3. Count inliers
       3 points                these points          (points near plane)

       ·  ·                    ·  ·                    ·  ·
      ·  ·  ·                 ─────────────          ─────●─────
     ● ·  ·  ·               ·  ●  ●  ●  ·            ●  ●  ●  ●
       ●    ●                   ·    ·                   ●    ●
      ·  ·   ·                 ·  ·   ·                ·  ·   ·

    4. Repeat many times, keep plane with most inliers

    Why it works:
    - Random sampling eventually picks good points
    - Good hypotheses get high inlier counts
    - Bad hypotheses (including outliers) get rejected
```

```python
def segment_dominant_plane(pcd: o3d.geometry.PointCloud,
                           distance_threshold: float = 0.02,
                           ransac_n: int = 3,
                           num_iterations: int = 1000):
    """
    Segment the largest planar surface from the point cloud.

    Commonly used to remove ground planes or table surfaces
    before processing remaining objects.

    Returns:
    - plane_model: [a, b, c, d] where ax + by + cz + d = 0
    - inlier_cloud: Points belonging to the plane
    - outlier_cloud: Points not on the plane (remaining objects)
    """
    # Run RANSAC plane fitting
    plane_model, inlier_indices = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )

    # Extract plane equation coefficients
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")

    # Separate inliers (plane) from outliers (objects)
    inlier_cloud = pcd.select_by_index(inlier_indices)
    outlier_cloud = pcd.select_by_index(inlier_indices, invert=True)

    # Color for visualization
    inlier_cloud.paint_uniform_color([0.5, 0.5, 0.5])  # Gray for plane
    outlier_cloud.paint_uniform_color([1.0, 0.0, 0.0])  # Red for objects

    return plane_model, inlier_cloud, outlier_cloud
```

### Multi-Plane Segmentation

Real environments contain multiple planes. Iterative segmentation extracts them one by one.

```python
def segment_all_planes(pcd: o3d.geometry.PointCloud,
                       min_points: int = 100,
                       distance_threshold: float = 0.02,
                       max_planes: int = 10):
    """
    Extract multiple planes from a point cloud iteratively.

    Continues until remaining cloud is too small or max_planes reached.
    """
    planes = []
    remaining = pcd

    for i in range(max_planes):
        if len(remaining.points) < min_points:
            break

        # Segment next largest plane
        plane_model, inliers, remaining = segment_dominant_plane(
            remaining,
            distance_threshold=distance_threshold
        )

        if len(inliers.points) < min_points:
            # No more significant planes
            break

        planes.append({
            'model': plane_model,
            'points': inliers,
            'size': len(inliers.points)
        })

        print(f"Plane {i+1}: {len(inliers.points)} points")

    return planes, remaining
```

## Object Clustering: Finding Distinct Objects

### DBSCAN Clustering

After removing ground planes, remaining points often correspond to distinct objects. DBSCAN (Density-Based Spatial Clustering of Applications with Noise) groups nearby points into clusters.

```
DBSCAN Clustering
================

    Parameters:
    - eps (ε): Maximum distance between neighbors
    - min_points: Minimum cluster size

    ┌───────────────────────────────────────┐
    │                                       │
    │    ··●·           ●●●                 │
    │   ●●●●●           ●●                  │
    │    ●●●             ●                  │
    │                                       │
    │              ·                        │
    │                       ●●●●            │
    │        ·              ●●●●●           │
    │                        ●●●            │
    │                                       │
    └───────────────────────────────────────┘

    Result: 3 clusters (dense regions)
            2 noise points (isolated dots)
```

```python
def cluster_objects(pcd: o3d.geometry.PointCloud,
                    eps: float = 0.05,
                    min_points: int = 50):
    """
    Cluster point cloud into distinct objects using DBSCAN.

    Parameters:
    - eps: Maximum distance between points in same cluster
           (smaller = more, smaller clusters)
    - min_points: Minimum points to form a cluster
                  (larger = fewer, larger clusters)

    Returns list of point clouds, one per detected object.
    """
    # Run DBSCAN clustering
    labels = np.array(pcd.cluster_dbscan(
        eps=eps,
        min_points=min_points,
        print_progress=True
    ))

    # Find number of clusters (label -1 is noise)
    max_label = labels.max()
    print(f"Found {max_label + 1} clusters")

    # Extract each cluster
    clusters = []
    for i in range(max_label + 1):
        cluster_indices = np.where(labels == i)[0]
        cluster = pcd.select_by_index(cluster_indices)

        # Assign unique color for visualization
        color = plt.cm.tab10(i / 10)[:3]
        cluster.paint_uniform_color(color)

        clusters.append({
            'cloud': cluster,
            'num_points': len(cluster_indices),
            'center': cluster.get_center(),
            'bbox': cluster.get_axis_aligned_bounding_box()
        })

    return clusters
```

### Euclidean Cluster Extraction

An alternative clustering approach uses connected components in a graph where edges connect nearby points.

```python
def euclidean_cluster_extraction(pcd: o3d.geometry.PointCloud,
                                 cluster_tolerance: float = 0.02,
                                 min_cluster_size: int = 100,
                                 max_cluster_size: int = 25000):
    """
    Extract clusters using Euclidean distance threshold.

    Similar to DBSCAN but with explicit size bounds.
    Often used in pick-and-place applications.
    """
    # Build KD-tree for efficient neighbor search
    tree = o3d.geometry.KDTreeFlann(pcd)
    points = np.asarray(pcd.points)
    n_points = len(points)

    # Track which points have been assigned
    processed = np.zeros(n_points, dtype=bool)
    clusters = []

    for seed_idx in range(n_points):
        if processed[seed_idx]:
            continue

        # Start new cluster from this seed
        cluster_indices = []
        queue = [seed_idx]

        while queue:
            current_idx = queue.pop(0)
            if processed[current_idx]:
                continue

            processed[current_idx] = True
            cluster_indices.append(current_idx)

            # Find neighbors within tolerance
            [k, idx, _] = tree.search_radius_vector_3d(
                points[current_idx],
                cluster_tolerance
            )

            for neighbor_idx in idx:
                if not processed[neighbor_idx]:
                    queue.append(neighbor_idx)

        # Check cluster size bounds
        if min_cluster_size <= len(cluster_indices) <= max_cluster_size:
            cluster = pcd.select_by_index(cluster_indices)
            clusters.append(cluster)

    return clusters
```

## Depth Estimation from Images

### Stereo Vision Fundamentals

Stereo vision mimics human binocular vision, computing depth from the disparity between two viewpoints.

```
Stereo Vision Geometry
=====================

    Left Camera          Right Camera
         │                    │
         │ ←── baseline b ───→│
         │                    │
         ▼                    ▼
    ┌─────────┐          ┌─────────┐
    │    ●    │          │  ●      │
    │   (xL)  │          │ (xR)    │
    └─────────┘          └─────────┘
         │                    │
         └────────┬───────────┘
                  │
                  ▼
             3D Point (X, Y, Z)

    Disparity: d = xL - xR

    Depth formula: Z = (b × f) / d

    Where:
    - b = baseline (distance between cameras)
    - f = focal length
    - d = disparity (pixel difference)

    Key insight: Nearby objects have LARGE disparity
                 Distant objects have SMALL disparity
```

**Stereo Matching with OpenCV:**

```python
import cv2
import numpy as np

def compute_stereo_disparity(left_image: np.ndarray,
                             right_image: np.ndarray,
                             num_disparities: int = 128,
                             block_size: int = 5) -> np.ndarray:
    """
    Compute disparity map from rectified stereo image pair.

    Uses Semi-Global Block Matching (SGBM) for robust results.
    Input images must be rectified (epipolar lines horizontal).

    Parameters:
    - num_disparities: Maximum disparity (must be divisible by 16)
    - block_size: Matching window size (must be odd)
    """
    # Convert to grayscale if needed
    if len(left_image.shape) == 3:
        left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
    else:
        left_gray, right_gray = left_image, right_image

    # Create SGBM matcher with tuned parameters
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * block_size**2,    # Smoothness penalty
        P2=32 * 3 * block_size**2,   # Larger smoothness penalty
        disp12MaxDiff=1,              # Left-right consistency
        uniquenessRatio=10,           # Uniqueness threshold
        speckleWindowSize=100,        # Speckle filter
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # Compute disparity (result is fixed-point with 4 fractional bits)
    disparity = stereo.compute(left_gray, right_gray)

    # Convert to float and normalize
    disparity = disparity.astype(np.float32) / 16.0

    return disparity

def disparity_to_depth(disparity: np.ndarray,
                       baseline: float,
                       focal_length: float) -> np.ndarray:
    """
    Convert disparity map to depth map.

    Parameters:
    - baseline: Distance between cameras in meters
    - focal_length: Focal length in pixels

    Returns depth map in meters.
    """
    # Avoid division by zero
    disparity_safe = np.where(disparity > 0, disparity, 0.1)

    # Apply stereo geometry formula
    depth = (baseline * focal_length) / disparity_safe

    # Set invalid disparities to zero
    depth = np.where(disparity > 0, depth, 0)

    return depth
```

### Deep Learning Depth Estimation

Modern neural networks can estimate depth from single images by learning geometric cues.

```
Monocular Depth Cues
===================

    How AI learns depth from single images:

    1. RELATIVE SIZE                2. TEXTURE GRADIENT
    ┌────────────────────┐          ┌────────────────────┐
    │  ╔═══╗             │          │ ░░░░░░░░░░░░░░░░░ │
    │  ║   ║   ╔═╗       │          │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ │
    │  ╚═══╝   ╚═╝       │          │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │
    │  (near)  (far)     │          │ ████████████ │
    └────────────────────┘          └────────────────────┘
                                    (closer = larger pattern)

    3. OCCLUSION                    4. LINEAR PERSPECTIVE
    ┌────────────────────┐          ┌────────────────────┐
    │       ┌───┐        │          │   ╲             ╱  │
    │    ┌──┤   │        │          │    ╲           ╱   │
    │    │  │   │        │          │     ╲         ╱    │
    │    └──┴───┘        │          │      ╲───────╱     │
    │ (occluder in front)│          │  (converging lines)│
    └────────────────────┘          └────────────────────┘
```

```python
import torch
import torchvision.transforms as T

def estimate_depth_monocular(image: np.ndarray,
                             model_type: str = "DPT_Large") -> np.ndarray:
    """
    Estimate depth from a single image using pretrained model.

    Uses MiDaS (or similar) for relative depth estimation.
    Note: Returns relative depth, not absolute metric depth.
    """
    # Load pretrained model (downloads on first use)
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.eval()

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device)

    # Get model-specific transforms
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if "DPT" in model_type:
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    # Preprocess image
    input_batch = transform(image).to(device)

    # Inference
    with torch.no_grad():
        prediction = midas(input_batch)

        # Resize to original resolution
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()

    # Convert to numpy (inverse depth)
    depth = prediction.cpu().numpy()

    return depth
```

## Occupancy Grids: Discretizing 3D Space

### 2D Occupancy Grids for Navigation

Occupancy grids divide space into cells, each storing the probability of occupancy. They're fundamental for robot navigation.

```
2D Occupancy Grid
================

    Continuous space:               Discretized grid:

    ┌─────────────────────┐         ┌─┬─┬─┬─┬─┬─┬─┬─┐
    │                     │         │░│░│░│░│░│░│░│░│
    │   ┌──────┐          │         ├─┼─┼─┼─┼─┼─┼─┼─┤
    │   │      │          │         │░│█│█│█│█│░│░│░│
    │   │      │          │         ├─┼─┼─┼─┼─┼─┼─┼─┤
    │   └──────┘          │         │░│█│█│█│█│░│░│░│
    │          ┌────┐     │         ├─┼─┼─┼─┼─┼─┼─┼─┤
    │          │    │     │         │░│░│░│░│░│█│█│░│
    │          └────┘     │         ├─┼─┼─┼─┼─┼─┼─┼─┤
    │                     │         │░│░│░│░│░│█│█│░│
    └─────────────────────┘         └─┴─┴─┴─┴─┴─┴─┴─┘

    Obstacles (walls, objects)       █ = occupied (p > 0.5)
                                     ░ = free (p < 0.5)
                                     ? = unknown (p ≈ 0.5)
```

```python
import numpy as np
from typing import Tuple

class OccupancyGrid2D:
    """
    2D occupancy grid for robot navigation.

    Each cell stores log-odds probability of occupancy,
    updated incrementally from sensor observations.
    """

    def __init__(self,
                 resolution: float = 0.05,
                 size: Tuple[float, float] = (20.0, 20.0),
                 origin: Tuple[float, float] = (-10.0, -10.0)):
        """
        Initialize grid.

        Parameters:
        - resolution: Cell size in meters
        - size: Grid dimensions in meters (width, height)
        - origin: Position of grid corner in world frame
        """
        self.resolution = resolution
        self.origin = np.array(origin)

        # Grid dimensions in cells
        self.width = int(size[0] / resolution)
        self.height = int(size[1] / resolution)

        # Initialize with log-odds = 0 (probability = 0.5)
        self.log_odds = np.zeros((self.height, self.width))

        # Sensor model parameters
        self.log_odds_hit = 0.9     # Log-odds update for hit
        self.log_odds_miss = -0.4   # Log-odds update for miss
        self.log_odds_max = 5.0     # Clamping bounds
        self.log_odds_min = -5.0

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid indices."""
        gx = int((x - self.origin[0]) / self.resolution)
        gy = int((y - self.origin[1]) / self.resolution)
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Convert grid indices to world coordinates (cell center)."""
        x = self.origin[0] + (gx + 0.5) * self.resolution
        y = self.origin[1] + (gy + 0.5) * self.resolution
        return x, y

    def update_ray(self, start: Tuple[float, float],
                   end: Tuple[float, float],
                   hit: bool = True) -> None:
        """
        Update grid along a ray from sensor.

        Cells along the ray (except end) are marked free.
        End cell is marked occupied if hit=True.
        Uses Bresenham's line algorithm for efficiency.
        """
        # Convert to grid coordinates
        x0, y0 = self.world_to_grid(*start)
        x1, y1 = self.world_to_grid(*end)

        # Get cells along the ray using Bresenham
        cells = self._bresenham_line(x0, y0, x1, y1)

        # Mark traversed cells as free (except last)
        for gx, gy in cells[:-1]:
            if self._in_bounds(gx, gy):
                self.log_odds[gy, gx] = np.clip(
                    self.log_odds[gy, gx] + self.log_odds_miss,
                    self.log_odds_min,
                    self.log_odds_max
                )

        # Mark end cell as occupied or free
        gx, gy = cells[-1]
        if self._in_bounds(gx, gy):
            update = self.log_odds_hit if hit else self.log_odds_miss
            self.log_odds[gy, gx] = np.clip(
                self.log_odds[gy, gx] + update,
                self.log_odds_min,
                self.log_odds_max
            )

    def get_probability_map(self) -> np.ndarray:
        """Convert log-odds to probability map [0, 1]."""
        return 1.0 / (1.0 + np.exp(-self.log_odds))

    def _in_bounds(self, gx: int, gy: int) -> bool:
        """Check if grid coordinates are valid."""
        return 0 <= gx < self.width and 0 <= gy < self.height

    def _bresenham_line(self, x0, y0, x1, y1):
        """Bresenham's line algorithm for grid traversal."""
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            cells.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        return cells
```

### 3D Voxel Grids and OctoMap

For full 3D navigation and manipulation, 3D occupancy grids (voxel grids) extend the concept to three dimensions. OctoMap provides efficient hierarchical representation.

```
OctoMap Hierarchical Structure
=============================

    Full resolution:        OctoMap (octree):

    ┌──┬──┬──┬──┐           ┌────────────────┐
    ├──┼──┼──┼──┤           │                │
    ├──┼──┼──┼──┤           │    ┌──┬──┐     │
    ├──┼──┼──┼──┤           │    ├──┼──┤     │
    └──┴──┴──┴──┘           │    └──┴──┘     │
                            │                │
    64 cells                └────────────────┘
    (memory = O(n³))
                            Only subdivide where needed
                            (memory = O(n²) typical)

    Benefits:
    - Efficient for sparse environments
    - Multi-resolution queries
    - Handles large outdoor scenes
```

## Object Pose Estimation

### The 6-DoF Pose Problem

Object pose estimation determines an object's position (x, y, z) and orientation (roll, pitch, yaw) in 3D space—essential for robotic manipulation.

```
6-DoF Object Pose
================

    Pose = Position + Orientation

              Z (yaw)
              │    ╱ Y (pitch)
              │   ╱
              │  ╱
              │ ╱
              ├───────── X (roll)
             ╱│
            ╱ │
           ╱  │

    Position: (x, y, z) - where is the object?
    Orientation: (roll, pitch, yaw) - how is it rotated?

    Total: 6 Degrees of Freedom (6-DoF)

    Applications:
    - Grasping: Know where to grab
    - Assembly: Align parts precisely
    - Tracking: Follow moving objects
```

### ICP: Iterative Closest Point

ICP aligns a model point cloud to observed data, estimating the transformation (pose).

```
ICP Algorithm
=============

    1. Initial alignment           2. Find correspondences

    Model ○○○○                     ○───────●
    Data  ●●●●                     ○───────●
           ↕                       ○───────●
    (rough alignment)              ○───────●
                                   (nearest neighbors)

    3. Compute transformation      4. Iterate until converged

    ○○○○  ──rotation──▶  ○         ○●
        ──translation──▶   ○       ○●
                            ○      ○●
                             ○     ○●
    (minimize distance)            (model aligned to data)
```

```python
import open3d as o3d
import numpy as np

def estimate_pose_icp(source_cloud: o3d.geometry.PointCloud,
                      target_cloud: o3d.geometry.PointCloud,
                      initial_transform: np.ndarray = None,
                      max_correspondence_distance: float = 0.05,
                      max_iterations: int = 50) -> np.ndarray:
    """
    Estimate 6-DoF pose using ICP registration.

    Parameters:
    - source_cloud: Object model point cloud
    - target_cloud: Observed scene point cloud
    - initial_transform: Starting guess (4x4 matrix), identity if None
    - max_correspondence_distance: Maximum point-to-point distance

    Returns:
    - 4x4 transformation matrix (rotation + translation)
    """
    if initial_transform is None:
        initial_transform = np.eye(4)

    # Ensure both clouds have normals for point-to-plane ICP
    if not source_cloud.has_normals():
        source_cloud.estimate_normals()
    if not target_cloud.has_normals():
        target_cloud.estimate_normals()

    # Run ICP
    result = o3d.pipelines.registration.registration_icp(
        source_cloud,
        target_cloud,
        max_correspondence_distance,
        initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iterations
        )
    )

    print(f"ICP fitness: {result.fitness:.4f}")
    print(f"ICP RMSE: {result.inlier_rmse:.4f}")

    return result.transformation

def extract_pose_from_transform(transform: np.ndarray):
    """
    Extract position and orientation from 4x4 transform matrix.

    Returns:
    - position: (x, y, z) in meters
    - rotation_matrix: 3x3 rotation matrix
    - euler_angles: (roll, pitch, yaw) in radians
    """
    # Position is the translation vector
    position = transform[:3, 3]

    # Rotation is the upper-left 3x3
    rotation_matrix = transform[:3, :3]

    # Convert to Euler angles (ZYX convention)
    sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        roll = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        yaw = 0

    euler_angles = (roll, pitch, yaw)

    return position, rotation_matrix, euler_angles
```

### Global Registration: Finding Initial Alignment

ICP requires a good initial guess. Global registration methods find this alignment without prior knowledge.

```python
def global_registration_ransac(source: o3d.geometry.PointCloud,
                               target: o3d.geometry.PointCloud,
                               voxel_size: float = 0.05):
    """
    Find initial alignment using RANSAC-based global registration.

    Uses FPFH (Fast Point Feature Histograms) for matching.
    """
    # Downsample for efficiency
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)

    # Estimate normals
    source_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    target_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )

    # Compute FPFH features
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
    )
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
    )

    # RANSAC global registration
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=voxel_size * 1.5,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    )

    return result.transformation
```

## Deep Learning for 3D Perception

### PointNet: Learning Directly on Point Clouds

Traditional neural networks require structured input (images, grids). PointNet pioneered learning directly on unordered point sets.

```
PointNet Architecture
====================

    Input: N × 3 points        Per-point features         Global feature

    ●  (x₁, y₁, z₁)  ───▶  ┌────────────┐  ───▶   ┌───────────┐
    ●  (x₂, y₂, z₂)  ───▶  │   Shared   │  ───▶   │  Max      │
    ●  (x₃, y₃, z₃)  ───▶  │   MLP      │  ───▶   │  Pooling  │
    ...                     │            │         │           │
    ●  (xₙ, yₙ, zₙ)  ───▶  └────────────┘  ───▶   └───────────┘
                                                        │
                            Per-point: 64 → 128 → 1024  │
                                                        ▼
                                               Global: 1024-dim
                                                        │
                                               ┌────────┴────────┐
                                               │   Classifier    │
                                               │   or Segmenter  │
                                               └─────────────────┘

    Key insight: Max pooling over points creates permutation invariance
    (order of points doesn't matter)
```

### 3D Object Detection Networks

Modern 3D detection combines multiple representations:

| Architecture | Input | Key Innovation | Use Case |
|--------------|-------|----------------|----------|
| **PointPillars** | LiDAR | Points to pillars | Autonomous driving |
| **VoxelNet** | LiDAR | Voxel encoding | General detection |
| **PointRCNN** | LiDAR | Point-based proposals | High accuracy |
| **SECOND** | LiDAR | Sparse convolutions | Efficiency |
| **CenterPoint** | LiDAR | Center heatmaps | State-of-the-art |
| **BEVFormer** | Camera + LiDAR | Transformer BEV | Multi-modal |

## Industry Perspectives: 3D Perception in Practice

### Warehouse Robotics

Amazon, Ocado, and other warehouse operators deploy 3D perception for picking and packing:

```
Warehouse Pick System
====================

    ┌─────────────────────────────────────────┐
    │             Overhead LiDAR              │
    │                   │                     │
    │   ┌───────────────┼───────────────┐     │
    │   │               ▼               │     │
    │   │  ┌─────┬─────┬─────┬─────┐   │     │
    │   │  │ Box │ Box │ Box │ Box │   │ Bin │
    │   │  └─────┴─────┴─────┴─────┘   │     │
    │   │       ◄─── RGB-D camera ───► │     │
    │   │               │               │     │
    │   │               ▼               │     │
    │   │        ┌──────────┐          │     │
    │   │        │  Robot   │          │     │
    │   │        │   Arm    │          │     │
    │   │        └──────────┘          │     │
    │   └───────────────────────────────┘     │
    └─────────────────────────────────────────┘

    Pipeline:
    1. Segment bin from items (plane removal)
    2. Cluster individual items
    3. Estimate pose for grasping
    4. Plan collision-free motion
```

### Autonomous Vehicle Perception

Self-driving cars combine multiple 3D perception methods:

| Component | Method | Purpose |
|-----------|--------|---------|
| 3D detection | PointPillars + camera | Find vehicles, pedestrians |
| Tracking | Multi-object tracking | Maintain identity over time |
| Mapping | LiDAR SLAM | Build HD maps |
| Localization | Point cloud matching | Centimeter-level positioning |
| Free space | Occupancy grid | Drivable area detection |

### Humanoid Robot Applications

For humanoids operating in human environments:

| Task | 3D Perception Need | Typical Approach |
|------|-------------------|------------------|
| **Stair climbing** | Step height/depth | LiDAR + IMU |
| **Object grasping** | Precise pose | RGB-D + deep learning |
| **Door opening** | Handle detection | Camera + point cloud |
| **Crowd navigation** | Human detection | Multi-sensor fusion |

## Summary: The 3D Perception Pipeline

3D perception transforms raw sensor data into actionable understanding of the environment. The key concepts we've covered:

**Key Takeaways:**

1. **Depth is essential**: 2D vision cannot support robust interaction with the 3D world. Active or passive depth sensing is required.

2. **Point clouds are the native representation**: Unlike images, point clouds directly capture 3D geometry. Processing them requires specialized algorithms.

3. **Segmentation reveals structure**: Plane removal and clustering decompose scenes into meaningful components.

4. **Pose estimation enables manipulation**: Knowing where objects are and how they're oriented is fundamental to grasping.

5. **Deep learning revolutionizes 3D**: Networks like PointNet and their successors learn directly from point data.

6. **Integration is key**: Production systems combine multiple representations (point clouds, voxels, meshes, occupancy grids) for different tasks.

```
Complete 3D Perception Pipeline
==============================

    Raw Sensor Data
          │
          ▼
    ┌─────────────┐
    │ Preprocessing│  Downsample, filter outliers
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │Segmentation │  Remove ground, find surfaces
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │ Clustering  │  Identify individual objects
    └──────┬──────┘
           │
           ├──────────────────────────┐
           ▼                          ▼
    ┌─────────────┐           ┌─────────────┐
    │Recognition  │           │Pose Estimate│
    │(what is it?)│           │(where is it)│
    └──────┬──────┘           └──────┬──────┘
           │                          │
           └──────────┬───────────────┘
                      │
                      ▼
              Action (grasp, avoid, navigate)
```

---

## Further Reading

**Foundational Texts:**
- Szeliski, "Computer Vision: Algorithms and Applications" (2nd ed.) - Excellent depth estimation chapters
- Hartley & Zisserman, "Multiple View Geometry" - Mathematical foundations

**Key Papers:**
- Qi et al., "PointNet: Deep Learning on Point Sets" (2017)
- Rusu & Cousins, "3D is here: Point Cloud Library (PCL)" (2011)
- Lang et al., "PointPillars: Fast Encoders for Object Detection" (2019)

**Software Resources:**
- [Open3D Documentation](http://www.open3d.org/docs/)
- [Point Cloud Library (PCL)](https://pointclouds.org/)
- [OctoMap Library](https://octomap.github.io/)

**Online Courses:**
- [3D Machine Learning (GitHub Collection)](https://github.com/timzhang642/3D-Machine-Learning)
- ETH Zurich - 3D Vision Course Materials
