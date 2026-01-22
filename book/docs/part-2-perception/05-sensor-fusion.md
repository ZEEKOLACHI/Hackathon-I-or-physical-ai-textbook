---
id: ch-2-05
title: Sensor Fusion
sidebar_position: 2
difficulty: intermediate
estimated_time: 30
prerequisites: [ch-2-04]
---

# Sensor Fusion

Sensor fusion combines data from multiple sensors to achieve more accurate and robust perception than any single sensor alone.

## Why Sensor Fusion?

Different sensors have complementary strengths and weaknesses:

| Sensor | Strengths | Weaknesses |
|--------|-----------|------------|
| Camera | Rich visual information, color | Affected by lighting, no depth |
| LiDAR | Accurate depth, works in dark | Sparse, no color, expensive |
| Radar | All-weather, velocity | Low resolution |
| IMU | High frequency, orientation | Drift over time |

## Kalman Filter

The Kalman filter is the foundation of sensor fusion:

```python
import numpy as np

class KalmanFilter:
    def __init__(self, state_dim, measurement_dim):
        self.x = np.zeros(state_dim)  # State estimate
        self.P = np.eye(state_dim)    # State covariance
        self.F = np.eye(state_dim)    # State transition
        self.H = np.zeros((measurement_dim, state_dim))  # Observation
        self.Q = np.eye(state_dim)    # Process noise
        self.R = np.eye(measurement_dim)  # Measurement noise

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        y = z - self.H @ self.x  # Innovation
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain

        self.x = self.x + K @ y
        self.P = (np.eye(len(self.x)) - K @ self.H) @ self.P
```

## Extended Kalman Filter

For nonlinear systems:

```python
class ExtendedKalmanFilter(KalmanFilter):
    def __init__(self, state_dim, measurement_dim):
        super().__init__(state_dim, measurement_dim)

    def predict(self, f, jacobian_F):
        self.x = f(self.x)
        F = jacobian_F(self.x)
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z, h, jacobian_H):
        H = jacobian_H(self.x)
        y = z - h(self.x)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(len(self.x)) - K @ H) @ self.P
```

## Camera-LiDAR Fusion

```python
def fuse_camera_lidar(image, point_cloud, camera_matrix, transform):
    """
    Project LiDAR points onto camera image and fuse.
    """
    # Transform points to camera frame
    points_camera = transform @ point_cloud

    # Project to image
    pixels = camera_matrix @ points_camera[:3, :]
    pixels = pixels[:2, :] / pixels[2, :]

    # Associate points with image features
    fused_data = []
    for i, (u, v) in enumerate(pixels.T):
        if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
            fused_data.append({
                'pixel': (int(u), int(v)),
                'depth': point_cloud[2, i],
                'color': image[int(v), int(u)]
            })

    return fused_data
```

## ROS 2 Message Synchronization

```python
from message_filters import Subscriber, ApproximateTimeSynchronizer

class FusionNode(Node):
    def __init__(self):
        super().__init__('fusion_node')

        self.image_sub = Subscriber(self, Image, '/camera/image')
        self.lidar_sub = Subscriber(self, PointCloud2, '/lidar/points')

        self.sync = ApproximateTimeSynchronizer(
            [self.image_sub, self.lidar_sub],
            queue_size=10,
            slop=0.1
        )
        self.sync.registerCallback(self.fusion_callback)

    def fusion_callback(self, image_msg, lidar_msg):
        # Process synchronized messages
        fused = self.fuse_data(image_msg, lidar_msg)
        self.publish_fused(fused)
```

## Summary

- Sensor fusion improves perception robustness
- Kalman filters provide optimal estimation
- Camera-LiDAR fusion combines visual and depth information
- ROS 2 message filters enable time synchronization

## Further Reading

- Thrun, S. "Probabilistic Robotics"
- Bar-Shalom, Y. "Estimation with Applications to Tracking and Navigation"
