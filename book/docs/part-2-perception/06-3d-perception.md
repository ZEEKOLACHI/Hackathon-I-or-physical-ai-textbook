---
id: ch-2-06
title: 3D Perception
sidebar_position: 3
difficulty: intermediate
estimated_time: 35
prerequisites: [ch-2-04, ch-2-05]
---

# 3D Perception

3D perception enables robots to understand the three-dimensional structure of their environment, essential for navigation and manipulation.

## Point Cloud Processing

### Point Cloud Basics

```python
import numpy as np
import open3d as o3d

def load_point_cloud(file_path):
    """Load and visualize a point cloud."""
    pcd = o3d.io.read_point_cloud(file_path)
    o3d.visualization.draw_geometries([pcd])
    return pcd

def downsample_point_cloud(pcd, voxel_size=0.05):
    """Reduce point cloud density using voxel grid."""
    return pcd.voxel_down_sample(voxel_size=voxel_size)
```

### Plane Segmentation

```python
def segment_plane(pcd, distance_threshold=0.01):
    """Segment the dominant plane (e.g., ground or table)."""
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=1000
    )
    [a, b, c, d] = plane_model
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    return plane_model, inlier_cloud, outlier_cloud
```

### Clustering

```python
def cluster_objects(pcd, eps=0.02, min_points=10):
    """Cluster point cloud into distinct objects."""
    labels = np.array(pcd.cluster_dbscan(
        eps=eps,
        min_points=min_points
    ))

    max_label = labels.max()
    clusters = []

    for i in range(max_label + 1):
        cluster = pcd.select_by_index(np.where(labels == i)[0])
        clusters.append(cluster)

    return clusters
```

## Depth Estimation

### Stereo Vision

```python
def compute_disparity(left_image, right_image):
    """Compute disparity map from stereo images."""
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,
        blockSize=5,
        P1=8 * 3 * 5**2,
        P2=32 * 3 * 5**2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    disparity = stereo.compute(left_image, right_image)
    return disparity

def disparity_to_depth(disparity, baseline, focal_length):
    """Convert disparity to depth."""
    depth = (baseline * focal_length) / (disparity + 1e-6)
    return depth
```

### Monocular Depth Estimation

```python
import torch

def estimate_depth_mono(image, model):
    """Estimate depth from a single image using deep learning."""
    input_tensor = preprocess(image)

    with torch.no_grad():
        depth = model(input_tensor)

    return depth.squeeze().numpy()
```

## Occupancy Grids

```python
class OccupancyGrid:
    def __init__(self, resolution=0.05, size=(10, 10)):
        self.resolution = resolution
        self.grid = np.zeros((
            int(size[0] / resolution),
            int(size[1] / resolution)
        ))

    def update(self, point_cloud, robot_pose):
        """Update grid with new observations."""
        for point in point_cloud:
            grid_x = int(point[0] / self.resolution)
            grid_y = int(point[1] / self.resolution)
            if 0 <= grid_x < self.grid.shape[0] and \
               0 <= grid_y < self.grid.shape[1]:
                self.grid[grid_x, grid_y] = 1.0
```

## Object Pose Estimation

```python
def estimate_object_pose(point_cloud, object_model):
    """Estimate 6-DoF pose using ICP registration."""
    result = o3d.pipelines.registration.registration_icp(
        point_cloud,
        object_model,
        max_correspondence_distance=0.02,
        estimation_method=o3d.pipelines.registration.
            TransformationEstimationPointToPoint()
    )
    return result.transformation
```

## Summary

- Point cloud processing is fundamental to 3D perception
- Plane segmentation separates surfaces from objects
- Clustering identifies distinct objects
- Depth can be estimated from stereo or monocular images
- Occupancy grids represent the environment for planning

## Further Reading

- Rusu, R.B. & Cousins, S. "3D is here: Point Cloud Library"
- Open3D Documentation
