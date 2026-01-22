---
id: ch-6-17
title: Bipedal Locomotion
sidebar_position: 2
difficulty: advanced
estimated_time: 45
prerequisites: [ch-6-16]
---

# Bipedal Locomotion

Walking is one of the most challenging problems in humanoid robotics.

## Zero Moment Point (ZMP)

```python
import numpy as np

def compute_zmp(com, com_ddot, contact_points, gravity=9.81):
    """
    Compute Zero Moment Point for balance.
    """
    # ZMP equation: p_zmp = com - (com_z / g) * com_ddot_xy
    zmp = np.zeros(2)
    zmp[0] = com[0] - (com[2] / gravity) * com_ddot[0]
    zmp[1] = com[1] - (com[2] / gravity) * com_ddot[1]
    return zmp

def is_balanced(zmp, support_polygon):
    """Check if ZMP is within support polygon."""
    return point_in_polygon(zmp, support_polygon)
```

## Linear Inverted Pendulum Model

```python
class LIPMWalker:
    """
    Linear Inverted Pendulum Model for walking.
    """
    def __init__(self, com_height, gravity=9.81):
        self.z_c = com_height
        self.g = gravity
        self.omega = np.sqrt(gravity / com_height)

    def plan_footsteps(self, start, goal, step_length=0.3):
        """Plan sequence of footsteps."""
        footsteps = []
        current = start

        while np.linalg.norm(current - goal) > step_length:
            direction = (goal - current) / np.linalg.norm(goal - current)
            next_step = current + step_length * direction
            footsteps.append(next_step)
            current = next_step

        footsteps.append(goal)
        return footsteps

    def generate_com_trajectory(self, footsteps, step_time=0.8):
        """Generate CoM trajectory for walking."""
        trajectories = []

        for i in range(len(footsteps) - 1):
            # Single support phase
            traj = self.single_support_trajectory(
                footsteps[i], footsteps[i+1], step_time
            )
            trajectories.append(traj)

        return trajectories
```

## Capture Point

```python
def compute_capture_point(com, com_dot, omega):
    """
    Capture point: where to step to stop falling.
    """
    capture_point = com[:2] + com_dot[:2] / omega
    return capture_point
```

## Summary

- ZMP criterion ensures dynamic balance
- LIPM simplifies walking analysis
- Capture point guides step placement
- Real walking requires online adaptation

## Further Reading

- Kajita, S. "Introduction to Humanoid Robotics"
