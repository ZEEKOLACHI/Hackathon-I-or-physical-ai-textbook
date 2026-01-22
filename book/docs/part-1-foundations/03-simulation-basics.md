---
id: ch-1-03
title: Simulation Basics
sidebar_position: 3
difficulty: beginner
estimated_time: 30
prerequisites: [ch-1-01, ch-1-02]
---

# Simulation Basics

Simulation is the cornerstone of modern robotics development. This chapter introduces the key simulation tools and practices that enable safe, fast, and reproducible robot development.

## Why Simulate?

Simulation provides several critical advantages:

1. **Safety**: Test algorithms without risking expensive hardware or human safety
2. **Speed**: Run experiments faster than real-time, iterate rapidly
3. **Scale**: Execute thousands of trials in parallel
4. **Reproducibility**: Create consistent test conditions
5. **Accessibility**: Develop without physical robot hardware

## Simulation Tools Overview

### Gazebo

Gazebo is the most widely used open-source robot simulator in the ROS ecosystem.

**Key Features:**
- Physics engines (ODE, Bullet, DART, Simbody)
- Sensor simulation (cameras, LiDAR, IMU)
- ROS 2 integration via ros_gz bridge
- Plugin architecture for customization

```bash
# Install Gazebo with ROS 2 Humble
sudo apt install ros-humble-ros-gz

# Launch Gazebo
ros2 launch ros_gz_sim gz_sim.launch.py
```

### NVIDIA Isaac Sim

Isaac Sim provides GPU-accelerated, photorealistic simulation.

**Key Features:**
- RTX-enabled ray tracing for visual realism
- Domain randomization for ML training
- ROS 2 bridge for communication
- Synthetic data generation

### MuJoCo

MuJoCo (Multi-Joint dynamics with Contact) excels at contact-rich manipulation.

**Key Features:**
- Fast, accurate physics simulation
- Differentiable dynamics
- Python bindings (mujoco-py)
- Ideal for learning-based approaches

## Physics Simulation Fundamentals

### Rigid Body Dynamics

```
F = ma                    # Newton's second law
τ = Iα                    # Rotational dynamics
M(q)q̈ + C(q,q̇)q̇ + G(q) = τ   # Manipulator equation
```

Where:
- `M(q)`: Mass matrix
- `C(q,q̇)`: Coriolis and centrifugal terms
- `G(q)`: Gravity vector
- `τ`: Applied torques

### Contact Dynamics

Contact simulation is crucial for manipulation and locomotion:

```python
# Simplified contact model
def compute_contact_force(penetration, velocity):
    """
    Spring-damper contact model
    """
    k = 10000  # Spring stiffness
    d = 100    # Damping coefficient

    if penetration > 0:
        normal_force = k * penetration - d * velocity
        return max(0, normal_force)
    return 0
```

### Time Stepping

Simulation advances in discrete time steps:

```python
def simulation_step(state, dt):
    """
    Basic Euler integration step
    """
    # Compute forces
    forces = compute_forces(state)

    # Update velocities
    acceleration = forces / state.mass
    state.velocity += acceleration * dt

    # Update positions
    state.position += state.velocity * dt

    return state
```

## Setting Up a Gazebo Simulation

### World File

```xml
<?xml version="1.0"?>
<sdf version="1.8">
  <world name="robot_world">
    <!-- Physics configuration -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
    </physics>

    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Lighting -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Robot model -->
    <include>
      <uri>model://my_robot</uri>
      <pose>0 0 0.5 0 0 0</pose>
    </include>
  </world>
</sdf>
```

### Robot Description (URDF)

```xml
<?xml version="1.0"?>
<robot name="simple_robot">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.1"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.3 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10"/>
      <inertia ixx="0.1" ixy="0" ixz="0"
               iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Wheel link -->
  <link name="wheel_link">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1"/>
      <inertia ixx="0.01" ixy="0" ixz="0"
               iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Joint -->
  <joint name="wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_link"/>
    <origin xyz="0.2 0.2 0" rpy="1.5708 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>
</robot>
```

## Sensor Simulation

### Camera

```xml
<gazebo reference="camera_link">
  <sensor name="camera" type="camera">
    <update_rate>30</update_rate>
    <camera>
      <horizontal_fov>1.047</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>100</far>
      </clip>
    </camera>
    <plugin name="camera_plugin" filename="libgazebo_ros_camera.so">
      <ros>
        <namespace>/my_robot</namespace>
        <remapping>image_raw:=image</remapping>
      </ros>
    </plugin>
  </sensor>
</gazebo>
```

### LiDAR

```xml
<gazebo reference="lidar_link">
  <sensor name="lidar" type="ray">
    <update_rate>10</update_rate>
    <ray>
      <scan>
        <horizontal>
          <samples>360</samples>
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal>
      </scan>
      <range>
        <min>0.1</min>
        <max>30.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="lidar_plugin" filename="libgazebo_ros_ray_sensor.so">
      <ros>
        <namespace>/my_robot</namespace>
        <remapping>~/out:=scan</remapping>
      </ros>
      <output_type>sensor_msgs/LaserScan</output_type>
    </plugin>
  </sensor>
</gazebo>
```

## ROS 2 Integration

### Launch File

```python
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_dir = get_package_share_directory('my_robot_sim')

    # Start Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('ros_gz_sim'),
                        'launch', 'gz_sim.launch.py')
        ]),
        launch_arguments={'gz_args': '-r empty.sdf'}.items()
    )

    # Spawn robot
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'my_robot',
            '-topic', 'robot_description',
            '-x', '0', '-y', '0', '-z', '0.5'
        ]
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': open('urdf/robot.urdf').read()}]
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_robot,
    ])
```

### Bridge Configuration

```yaml
# gz_bridge.yaml
- ros_topic_name: "/cmd_vel"
  gz_topic_name: "/model/my_robot/cmd_vel"
  ros_type_name: "geometry_msgs/msg/Twist"
  gz_type_name: "gz.msgs.Twist"
  direction: ROS_TO_GZ

- ros_topic_name: "/scan"
  gz_topic_name: "/lidar"
  ros_type_name: "sensor_msgs/msg/LaserScan"
  gz_type_name: "gz.msgs.LaserScan"
  direction: GZ_TO_ROS
```

## Simulation Best Practices

### 1. Start Simple

Begin with basic scenarios and incrementally add complexity:

```python
# Progressive complexity
def test_robot():
    # Level 1: Static environment, no obstacles
    test_basic_motion()

    # Level 2: Static obstacles
    test_obstacle_avoidance()

    # Level 3: Dynamic obstacles
    test_dynamic_navigation()

    # Level 4: Full scenario
    test_complete_mission()
```

### 2. Validate Against Reality

Use sim-to-real transfer techniques:

- **Domain randomization**: Vary physics parameters
- **System identification**: Tune simulation to match real robot
- **Reality gap analysis**: Quantify differences

### 3. Reproducibility

```python
# Set seeds for reproducibility
import random
import numpy as np

def setup_simulation(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    # Also set simulator-specific seeds
```

### 4. Performance Optimization

```python
# Parallel simulation
from multiprocessing import Pool

def run_simulation(params):
    # Single simulation run
    return simulate(params)

# Run multiple simulations in parallel
with Pool(processes=8) as pool:
    results = pool.map(run_simulation, parameter_sets)
```

## Practical Exercise

Set up a simulation environment with:
1. A simple mobile robot
2. Camera and LiDAR sensors
3. ROS 2 topic bridge
4. Teleoperation capability

:::info Simulation Files
The exercise files include pre-configured Gazebo worlds and robot models. Follow the setup guide in the repository.
:::

## Summary

This chapter covered:
- Why simulation is essential for robotics development
- Overview of major simulation tools (Gazebo, Isaac Sim, MuJoCo)
- Physics simulation fundamentals
- Setting up robot models and sensors
- ROS 2 integration with Gazebo
- Best practices for simulation-based development

The next part of the textbook will focus on perception—teaching robots to understand their environment through sensors.

## Further Reading

- [Gazebo Documentation](https://gazebosim.org/docs)
- [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim)
- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [Open Dynamics Engine](https://ode.org/)
