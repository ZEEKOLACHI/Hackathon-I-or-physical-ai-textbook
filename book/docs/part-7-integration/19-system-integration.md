---
id: ch-7-19
title: System Integration
sidebar_position: 1
difficulty: advanced
estimated_time: 45
prerequisites: [ch-6-16, ch-6-17, ch-6-18]
---

# System Integration

Bringing together perception, planning, control, and learning into a cohesive robotic system is the ultimate challenge in Physical AI.

## Architecture Patterns

### Hierarchical Control Architecture

```python
class HierarchicalController:
    """
    Multi-level control architecture for humanoid robots.

    Levels:
    - Strategic: Task planning (seconds to minutes)
    - Tactical: Motion planning (100ms to seconds)
    - Operational: Real-time control (1-10ms)
    """

    def __init__(self):
        self.task_planner = TaskPlanner()
        self.motion_planner = MotionPlanner()
        self.controller = WholeBodyController()

    def execute_task(self, task_goal):
        # Strategic level: decompose task
        subtasks = self.task_planner.plan(task_goal)

        for subtask in subtasks:
            # Tactical level: plan motion
            trajectory = self.motion_planner.plan(subtask)

            # Operational level: execute with feedback
            for waypoint in trajectory:
                self.controller.track(waypoint)

    def run(self, dt=0.001):
        """Main control loop at 1kHz."""
        while self.running:
            state = self.get_state()
            command = self.controller.compute(state)
            self.send_command(command)
            time.sleep(dt)
```

### Behavior-Based Architecture

```python
class BehaviorArchitecture:
    """
    Subsumption-style behavior layering.

    Higher priority behaviors can suppress lower ones.
    """

    def __init__(self):
        self.behaviors = []

    def add_behavior(self, behavior, priority):
        self.behaviors.append((priority, behavior))
        self.behaviors.sort(key=lambda x: x[0], reverse=True)

    def compute_action(self, state):
        for priority, behavior in self.behaviors:
            if behavior.is_active(state):
                action = behavior.compute(state)
                if action is not None:
                    return action
        return self.default_action()


class SafetyBehavior:
    """Highest priority: emergency stops and collision avoidance."""

    def is_active(self, state):
        return state.obstacle_distance < 0.5 or state.joint_limit_margin < 0.1

    def compute(self, state):
        if state.obstacle_distance < 0.2:
            return EmergencyStop()
        return AvoidanceAction(state.obstacle_direction)
```

## ROS 2 System Integration

### Node Composition

```python
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception')
        self.camera_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.image_callback, 10)
        self.detection_pub = self.create_publisher(
            DetectionArray, '/detections', 10)

    def image_callback(self, msg):
        detections = self.detector.detect(msg)
        self.detection_pub.publish(detections)


class PlanningNode(Node):
    def __init__(self):
        super().__init__('planning')
        self.detection_sub = self.create_subscription(
            DetectionArray, '/detections', self.detection_callback, 10)
        self.trajectory_pub = self.create_publisher(
            JointTrajectory, '/trajectory', 10)

    def detection_callback(self, msg):
        if self.should_replan(msg):
            trajectory = self.planner.plan(msg)
            self.trajectory_pub.publish(trajectory)


class ControlNode(Node):
    def __init__(self):
        super().__init__('control')
        self.trajectory_sub = self.create_subscription(
            JointTrajectory, '/trajectory', self.trajectory_callback, 10)
        self.timer = self.create_timer(0.001, self.control_loop)

    def control_loop(self):
        command = self.controller.compute(self.current_state)
        self.command_pub.publish(command)


def main():
    rclpy.init()
    executor = MultiThreadedExecutor()

    executor.add_node(PerceptionNode())
    executor.add_node(PlanningNode())
    executor.add_node(ControlNode())

    executor.spin()
```

### Launch System

```python
# launch/robot_bringup.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    return LaunchDescription([
        # Perception container
        ComposableNodeContainer(
            name='perception_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container',
            composable_node_descriptions=[
                ComposableNode(
                    package='perception',
                    plugin='perception::CameraNode',
                    name='camera',
                ),
                ComposableNode(
                    package='perception',
                    plugin='perception::DetectorNode',
                    name='detector',
                ),
            ],
        ),

        # Planning node
        Node(
            package='planning',
            executable='motion_planner',
            name='motion_planner',
            parameters=[{'planning_time': 1.0}],
        ),

        # Control node with real-time priority
        Node(
            package='control',
            executable='whole_body_controller',
            name='controller',
            parameters=[{'control_rate': 1000.0}],
        ),
    ])
```

## State Machine Integration

```python
from enum import Enum, auto

class RobotState(Enum):
    IDLE = auto()
    NAVIGATING = auto()
    MANIPULATING = auto()
    RECOVERING = auto()
    EMERGENCY = auto()


class RobotStateMachine:
    def __init__(self):
        self.state = RobotState.IDLE
        self.transitions = {
            RobotState.IDLE: [RobotState.NAVIGATING, RobotState.MANIPULATING],
            RobotState.NAVIGATING: [RobotState.IDLE, RobotState.MANIPULATING, RobotState.RECOVERING],
            RobotState.MANIPULATING: [RobotState.IDLE, RobotState.NAVIGATING, RobotState.RECOVERING],
            RobotState.RECOVERING: [RobotState.IDLE, RobotState.EMERGENCY],
            RobotState.EMERGENCY: [RobotState.IDLE],
        }

    def transition(self, new_state):
        if new_state in self.transitions[self.state]:
            self.on_exit(self.state)
            self.state = new_state
            self.on_enter(new_state)
            return True
        return False

    def on_enter(self, state):
        if state == RobotState.EMERGENCY:
            self.emergency_stop()
        elif state == RobotState.RECOVERING:
            self.start_recovery()

    def on_exit(self, state):
        if state == RobotState.MANIPULATING:
            self.release_objects()
```

## Communication Patterns

### Real-Time Data Flow

```python
import asyncio
from dataclasses import dataclass
from typing import Optional

@dataclass
class SensorData:
    timestamp: float
    joint_positions: list
    joint_velocities: list
    imu_orientation: list
    force_torque: list


class DataBus:
    """Lock-free data sharing for real-time systems."""

    def __init__(self):
        self._latest_data: Optional[SensorData] = None

    def write(self, data: SensorData):
        """Non-blocking write (atomic pointer swap)."""
        self._latest_data = data

    def read(self) -> Optional[SensorData]:
        """Non-blocking read."""
        return self._latest_data


class RealtimePipeline:
    def __init__(self):
        self.sensor_bus = DataBus()
        self.command_bus = DataBus()

    async def sensor_loop(self):
        """High-frequency sensor reading."""
        while True:
            data = self.read_sensors()
            self.sensor_bus.write(data)
            await asyncio.sleep(0.0001)  # 10kHz

    async def control_loop(self):
        """Real-time control at 1kHz."""
        while True:
            sensor_data = self.sensor_bus.read()
            if sensor_data:
                command = self.controller.compute(sensor_data)
                self.command_bus.write(command)
            await asyncio.sleep(0.001)
```

## Testing Integrated Systems

```python
import pytest
from unittest.mock import Mock, patch

class TestSystemIntegration:
    def test_perception_to_planning_pipeline(self):
        """Test data flows from perception to planning."""
        perception = PerceptionNode()
        planning = PlanningNode()

        # Simulate camera input
        test_image = create_test_image_with_object()

        # Process through perception
        detections = perception.process(test_image)

        # Verify planning receives and processes
        trajectory = planning.plan_from_detections(detections)

        assert trajectory is not None
        assert len(trajectory.points) > 0

    def test_emergency_stop_propagation(self):
        """Test emergency stop reaches all subsystems."""
        system = IntegratedSystem()

        # Trigger emergency
        system.emergency_stop()

        # Verify all components stopped
        assert system.controller.is_stopped()
        assert system.planner.is_stopped()
        assert system.state == RobotState.EMERGENCY
```

## Summary

- Hierarchical architectures separate concerns by timescale
- Behavior-based systems enable reactive responses
- ROS 2 provides robust middleware for integration
- State machines manage system modes safely
- Lock-free patterns enable real-time performance
- Integration testing validates end-to-end behavior

## Further Reading

- Kortenkamp, D. "Building Intelligent Robots"
- Quigley, M. "Programming Robots with ROS"
- ROS 2 Design Documentation: https://design.ros2.org
