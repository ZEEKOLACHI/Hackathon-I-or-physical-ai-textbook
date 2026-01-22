# Add Code Example

Add a well-documented code example to a chapter.

## Usage

```
/add-code-example <chapter-id> <topic>
```

## Process

1. Read the target chapter
2. Identify the best location for the code example
3. Generate code with:
   - Clear comments explaining each step
   - Type hints (for Python)
   - Error handling where appropriate
   - ROS 2 best practices if applicable
4. Add explanatory text before and after the code
5. Update the chapter file

## Code Style Guidelines

- Use Python 3.11+ syntax
- Include docstrings for functions/classes
- Follow PEP 8 style guide
- Use meaningful variable names
- Add inline comments for complex logic

## Example Output

```python
"""
Example: Simple ROS 2 Publisher Node

This example demonstrates how to create a basic publisher
that sends messages at a fixed rate.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MinimalPublisher(Node):
    """A minimal ROS 2 publisher node."""

    def __init__(self):
        super().__init__('minimal_publisher')
        # Create publisher with queue size of 10
        self.publisher_ = self.create_publisher(
            String,
            'topic',
            10
        )
        # Timer callback every 0.5 seconds
        self.timer = self.create_timer(0.5, self.timer_callback)
        self.count = 0

    def timer_callback(self):
        """Publish a message with incrementing count."""
        msg = String()
        msg.data = f'Hello World: {self.count}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.count += 1


def main(args=None):
    """Entry point for the node."""
    rclpy.init(args=args)
    node = MinimalPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```
