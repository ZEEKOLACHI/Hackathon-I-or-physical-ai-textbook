---
id: ch-6-16
title: Humanoid Kinematics
sidebar_position: 1
difficulty: advanced
estimated_time: 40
prerequisites: [ch-4-12]
---

# Humanoid Kinematics

Humanoid robots have complex kinematic structures that require specialized analysis.

## Forward Kinematics

```python
import numpy as np

def dh_transform(theta, d, a, alpha):
    """
    Denavit-Hartenberg transformation matrix.
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)

    return np.array([
        [ct, -st*ca, st*sa, a*ct],
        [st, ct*ca, -ct*sa, a*st],
        [0, sa, ca, d],
        [0, 0, 0, 1]
    ])

def forward_kinematics(joint_angles, dh_params):
    """Compute end-effector pose from joint angles."""
    T = np.eye(4)
    for i, (theta_offset, d, a, alpha) in enumerate(dh_params):
        theta = joint_angles[i] + theta_offset
        T = T @ dh_transform(theta, d, a, alpha)
    return T
```

## Inverse Kinematics

```python
def inverse_kinematics_numerical(target_pose, robot, q_init, max_iter=100):
    """
    Numerical IK using Jacobian pseudoinverse.
    """
    q = q_init.copy()

    for _ in range(max_iter):
        current_pose = robot.forward_kinematics(q)
        error = pose_error(target_pose, current_pose)

        if np.linalg.norm(error) < 1e-6:
            break

        J = robot.jacobian(q)
        J_pinv = np.linalg.pinv(J)
        dq = J_pinv @ error
        q += 0.1 * dq

    return q
```

## Center of Mass

```python
class HumanoidCoM:
    def __init__(self, robot):
        self.robot = robot

    def compute_com(self, joint_angles):
        """Compute whole-body center of mass."""
        total_mass = 0
        com = np.zeros(3)

        for link in self.robot.links:
            link_pose = self.robot.get_link_pose(link, joint_angles)
            link_com = link_pose[:3, 3] + link_pose[:3, :3] @ link.local_com

            com += link.mass * link_com
            total_mass += link.mass

        return com / total_mass
```

## Summary

- DH convention provides systematic joint parameterization
- Numerical IK handles complex kinematic chains
- CoM tracking is essential for balance
- Redundancy enables multiple solutions

## Further Reading

- Siciliano, B. "Robotics: Modelling, Planning and Control"
