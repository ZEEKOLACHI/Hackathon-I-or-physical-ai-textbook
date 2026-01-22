---
id: ch-1-02
title: ROS 2 Fundamentals
sidebar_position: 2
difficulty: beginner
estimated_time: 35
prerequisites: [ch-1-01]
---

# ROS 2 Fundamentals

ROS 2 (Robot Operating System 2) is the foundation of modern robotic software development. This chapter introduces the core concepts and practical skills you need to build robot applications with ROS 2.

## What is ROS 2?

ROS 2 is not an operating system in the traditional sense. It's a middleware framework that provides:

- **Communication infrastructure**: Publish/subscribe messaging, services, actions
- **Hardware abstraction**: Standard interfaces for sensors and actuators
- **Development tools**: Visualization, debugging, simulation integration
- **Package ecosystem**: Thousands of reusable components

### Why ROS 2 over ROS 1?

ROS 2 was designed from the ground up to address ROS 1 limitations:

| Feature | ROS 1 | ROS 2 |
|---------|-------|-------|
| Real-time support | Limited | DDS-based, real-time capable |
| Security | No built-in security | DDS security extensions |
| Multi-robot | Single master limitation | Decentralized, scalable |
| Platforms | Linux only (primarily) | Linux, Windows, macOS |
| Lifecycle | None | Managed node lifecycle |

## Core Concepts

### Nodes

Nodes are the basic building blocks of ROS 2 applications. Each node is a process that performs a specific task.

```python
import rclpy
from rclpy.node import Node

class MinimalNode(Node):
    def __init__(self):
        super().__init__('minimal_node')
        self.get_logger().info('Hello from ROS 2!')

def main(args=None):
    rclpy.init(args=args)
    node = MinimalNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Topics

Topics enable asynchronous, many-to-many communication using a publish/subscribe pattern.

```python
from std_msgs.msg import String

class Publisher(Node):
    def __init__(self):
        super().__init__('publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        self.timer = self.create_timer(0.5, self.timer_callback)
        self.count = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Message {self.count}'
        self.publisher_.publish(msg)
        self.count += 1

class Subscriber(Node):
    def __init__(self):
        super().__init__('subscriber')
        self.subscription = self.create_subscription(
            String, 'topic', self.listener_callback, 10)

    def listener_callback(self, msg):
        self.get_logger().info(f'Received: {msg.data}')
```

### Services

Services provide synchronous, request/response communication.

```python
from example_interfaces.srv import AddTwoInts

class ServiceServer(Node):
    def __init__(self):
        super().__init__('add_two_ints_server')
        self.srv = self.create_service(
            AddTwoInts, 'add_two_ints', self.add_callback)

    def add_callback(self, request, response):
        response.sum = request.a + request.b
        return response

class ServiceClient(Node):
    def __init__(self):
        super().__init__('add_two_ints_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for service...')

    async def call_service(self, a, b):
        request = AddTwoInts.Request()
        request.a = a
        request.b = b
        future = self.cli.call_async(request)
        return await future
```

### Actions

Actions are for long-running tasks with feedback and cancellation support.

```python
from action_tutorials_interfaces.action import Fibonacci
from rclpy.action import ActionServer

class FibonacciActionServer(Node):
    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            self.execute_callback)

    async def execute_callback(self, goal_handle):
        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1])
            goal_handle.publish_feedback(feedback_msg)

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        return result
```

## ROS 2 Architecture

```
┌─────────────────────────────────────────────────┐
│                  Application                     │
│   (Your robot code, algorithms, behaviors)       │
├─────────────────────────────────────────────────┤
│               rclpy / rclcpp                     │
│   (ROS 2 Client Libraries)                       │
├─────────────────────────────────────────────────┤
│                    rcl                           │
│   (ROS Client Library - C implementation)        │
├─────────────────────────────────────────────────┤
│                    rmw                           │
│   (ROS Middleware Interface)                     │
├─────────────────────────────────────────────────┤
│            DDS Implementation                    │
│   (FastDDS, CycloneDDS, Connext)                │
├─────────────────────────────────────────────────┤
│              Operating System                    │
│   (Linux, Windows, macOS)                        │
└─────────────────────────────────────────────────┘
```

## Package Structure

A typical ROS 2 Python package:

```
my_package/
├── my_package/
│   ├── __init__.py
│   ├── node_one.py
│   └── node_two.py
├── resource/
│   └── my_package
├── test/
│   └── test_node.py
├── package.xml
├── setup.py
└── setup.cfg
```

### package.xml

```xml
<?xml version="1.0"?>
<package format="3">
  <name>my_package</name>
  <version>0.0.1</version>
  <description>My ROS 2 package</description>
  <maintainer email="user@example.com">Your Name</maintainer>
  <license>Apache-2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

### setup.py

```python
from setuptools import setup

package_name = 'my_package'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    entry_points={
        'console_scripts': [
            'node_one = my_package.node_one:main',
            'node_two = my_package.node_two:main',
        ],
    },
)
```

## Common Commands

```bash
# Build a workspace
colcon build

# Source the workspace
source install/setup.bash

# Run a node
ros2 run my_package node_one

# List topics
ros2 topic list

# Echo topic data
ros2 topic echo /topic_name

# Call a service
ros2 service call /add_two_ints example_interfaces/srv/AddTwoInts "{a: 1, b: 2}"

# View node graph
ros2 run rqt_graph rqt_graph

# Record data
ros2 bag record -a

# Play back data
ros2 bag play my_bag_file
```

## Quality of Service (QoS)

ROS 2 uses DDS QoS profiles to configure communication reliability:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# Reliable delivery (like TCP)
reliable_qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10
)

# Best effort delivery (like UDP)
best_effort_qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1
)

# Sensor data profile (commonly used)
from rclpy.qos import qos_profile_sensor_data
```

## Lifecycle Nodes

Managed lifecycle for deterministic startup/shutdown:

```python
from rclpy.lifecycle import LifecycleNode, State, TransitionCallbackReturn

class ManagedNode(LifecycleNode):
    def __init__(self, node_name):
        super().__init__(node_name)

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Configuring...')
        # Initialize resources
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Activating...')
        # Start processing
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Deactivating...')
        # Stop processing
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Cleaning up...')
        # Release resources
        return TransitionCallbackReturn.SUCCESS
```

## Practical Exercise

Create a simple robot controller that:
1. Subscribes to sensor data (simulated)
2. Publishes velocity commands
3. Provides a service to change speed

:::info Exercise Files
All exercise code is available in the companion repository. Follow the setup instructions in the README to get started with simulation.
:::

## Summary

This chapter covered the essential concepts of ROS 2:
- Nodes as the basic computation units
- Topics for asynchronous pub/sub communication
- Services for synchronous request/response
- Actions for long-running tasks with feedback
- Package structure and build system
- QoS profiles for communication configuration
- Lifecycle nodes for managed state

In the next chapter, we'll set up simulation environments to test our ROS 2 applications.

## Further Reading

- [ROS 2 Documentation](https://docs.ros.org/en/humble/)
- [ROS 2 Tutorials](https://docs.ros.org/en/humble/Tutorials.html)
- [ROS 2 Design](https://design.ros2.org/)
