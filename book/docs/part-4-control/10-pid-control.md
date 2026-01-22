---
id: ch-4-10
title: PID Control
sidebar_position: 1
difficulty: beginner
estimated_time: 30
prerequisites: [ch-1-03]
---

# PID Control

PID (Proportional-Integral-Derivative) control is the most widely used control strategy in robotics.

## The PID Controller

```python
class PIDController:
    def __init__(self, kp, ki, kd, dt=0.01):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.dt = dt

        self.integral = 0
        self.prev_error = 0

    def compute(self, setpoint, measured):
        error = setpoint - measured

        # Proportional term
        p_term = self.kp * error

        # Integral term
        self.integral += error * self.dt
        i_term = self.ki * self.integral

        # Derivative term
        derivative = (error - self.prev_error) / self.dt
        d_term = self.kd * derivative

        self.prev_error = error

        return p_term + i_term + d_term
```

## Tuning Methods

### Ziegler-Nichols

```python
def ziegler_nichols_tuning(ku, tu):
    """
    Calculate PID gains using Ziegler-Nichols method.

    Args:
        ku: Ultimate gain (gain at oscillation)
        tu: Ultimate period (oscillation period)
    """
    kp = 0.6 * ku
    ki = 2 * kp / tu
    kd = kp * tu / 8
    return kp, ki, kd
```

## Anti-Windup

```python
class PIDWithAntiWindup(PIDController):
    def __init__(self, kp, ki, kd, dt=0.01, integral_limit=100):
        super().__init__(kp, ki, kd, dt)
        self.integral_limit = integral_limit

    def compute(self, setpoint, measured):
        error = setpoint - measured

        p_term = self.kp * error

        self.integral += error * self.dt
        self.integral = np.clip(
            self.integral,
            -self.integral_limit,
            self.integral_limit
        )
        i_term = self.ki * self.integral

        derivative = (error - self.prev_error) / self.dt
        d_term = self.kd * derivative

        self.prev_error = error

        return p_term + i_term + d_term
```

## ROS 2 Integration

```python
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

class VelocityController(Node):
    def __init__(self):
        super().__init__('velocity_controller')

        self.pid = PIDController(kp=1.0, ki=0.1, kd=0.05)
        self.target_velocity = 0.0

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)

    def odom_callback(self, msg):
        current_vel = msg.twist.twist.linear.x
        control = self.pid.compute(self.target_velocity, current_vel)

        cmd = Twist()
        cmd.linear.x = control
        self.cmd_pub.publish(cmd)
```

## Summary

- PID combines proportional, integral, and derivative control
- Tuning balances response speed and stability
- Anti-windup prevents integral term overflow
- PID is fundamental but has limitations for complex systems

## Further Reading

- Åström, K.J. & Murray, R.M. "Feedback Systems"
