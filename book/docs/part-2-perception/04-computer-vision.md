---
id: ch-2-04
title: Computer Vision for Robotics
sidebar_position: 1
difficulty: intermediate
estimated_time: 35
prerequisites: [ch-1-03]
---

# Computer Vision for Robotics

Computer vision enables robots to perceive and understand visual information from cameras. This chapter covers the fundamental techniques used in robotic perception.

## Introduction

Visual perception is crucial for robots to:
- Navigate environments safely
- Recognize and manipulate objects
- Interact with humans
- Understand scene semantics

## Camera Models

### Pinhole Camera Model

The pinhole camera model describes how 3D points project onto a 2D image plane:

```python
import numpy as np

def project_point(point_3d, camera_matrix):
    """
    Project a 3D point to 2D image coordinates.

    Args:
        point_3d: [X, Y, Z] point in camera frame
        camera_matrix: 3x3 intrinsic camera matrix

    Returns:
        [u, v] pixel coordinates
    """
    # Homogeneous coordinates
    point_homogeneous = point_3d / point_3d[2]

    # Apply camera intrinsics
    pixel = camera_matrix @ point_homogeneous

    return pixel[:2]

# Example camera matrix
K = np.array([
    [fx, 0,  cx],
    [0,  fy, cy],
    [0,  0,  1]
])
```

### Camera Calibration

```python
import cv2
import numpy as np

def calibrate_camera(images, pattern_size=(9, 6)):
    """
    Calibrate camera using checkerboard pattern.
    """
    objpoints = []  # 3D points in world coordinates
    imgpoints = []  # 2D points in image plane

    # Prepare object points
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    return mtx, dist
```

## Image Processing

### Edge Detection

```python
def detect_edges(image):
    """Apply Canny edge detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges
```

### Feature Detection

```python
def detect_features(image):
    """Detect ORB features for matching."""
    orb = cv2.ORB_create(nfeatures=1000)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors
```

## Object Detection

### Deep Learning Approaches

Modern object detection uses neural networks like YOLO, SSD, and Faster R-CNN:

```python
import torch

def detect_objects(image, model, threshold=0.5):
    """
    Detect objects using a pre-trained model.
    """
    # Preprocess image
    input_tensor = preprocess(image)

    # Run inference
    with torch.no_grad():
        predictions = model(input_tensor)

    # Filter by confidence
    boxes = []
    for pred in predictions:
        if pred['score'] > threshold:
            boxes.append({
                'bbox': pred['boxes'],
                'label': pred['labels'],
                'score': pred['scores']
            })

    return boxes
```

## ROS 2 Integration

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        # Process image
        detections = self.process_image(cv_image)
        self.publish_detections(detections)
```

## Summary

This chapter covered:
- Camera models and calibration
- Image processing fundamentals
- Feature detection techniques
- Object detection with deep learning
- ROS 2 integration for vision systems

## Further Reading

- Szeliski, R. "Computer Vision: Algorithms and Applications"
- OpenCV Documentation
- PyTorch Vision Library
