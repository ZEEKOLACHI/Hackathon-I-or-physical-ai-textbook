---
id: ch-4-12
title: Whole-Body Control
sidebar_position: 3
difficulty: advanced
estimated_time: 40
prerequisites: [ch-4-10, ch-4-11]
---

# Whole-Body Control

Whole-body control coordinates all degrees of freedom for complex robots like humanoids.

## Task-Space Control

```python
import numpy as np

class TaskSpaceController:
    """
    Control robot end-effector in Cartesian space.
    """
    def __init__(self, robot):
        self.robot = robot

    def compute_torques(self, x_des, x_dot_des, x, x_dot):
        # Get Jacobian
        J = self.robot.jacobian()

        # Task-space PD control
        kp = 100
        kd = 20

        x_ddot_des = kp * (x_des - x) + kd * (x_dot_des - x_dot)

        # Operational space dynamics
        Lambda = np.linalg.inv(J @ np.linalg.inv(self.robot.mass_matrix()) @ J.T)

        # Compute torques
        tau = J.T @ Lambda @ x_ddot_des

        return tau
```

## Null-Space Control

```python
class NullSpaceController:
    """
    Use redundancy to achieve secondary objectives.
    """
    def compute_torques(self, robot, primary_task, secondary_task):
        J = robot.jacobian()
        J_pinv = np.linalg.pinv(J)

        # Primary task torques
        tau_primary = J.T @ primary_task

        # Null-space projector
        N = np.eye(robot.n_joints) - J_pinv @ J

        # Secondary task (e.g., joint limit avoidance)
        tau_secondary = N @ secondary_task

        return tau_primary + tau_secondary
```

## Quadratic Programming

```python
from scipy.optimize import minimize

class QPController:
    """
    Whole-body control via quadratic programming.
    """
    def solve(self, robot, tasks, constraints):
        n_vars = robot.n_joints + 6  # joints + contact forces

        def objective(x):
            tau = x[:robot.n_joints]
            f = x[robot.n_joints:]

            cost = 0
            for task in tasks:
                error = task.compute_error(robot, tau, f)
                cost += task.weight * error.T @ error
            return cost

        # Constraints: dynamics, friction cones, joint limits
        cons = []
        for constraint in constraints:
            cons.append({
                'type': 'ineq',
                'fun': lambda x: constraint.evaluate(x)
            })

        result = minimize(objective, np.zeros(n_vars), constraints=cons)
        return result.x[:robot.n_joints]
```

## Summary

- Task-space control operates in Cartesian coordinates
- Null-space control uses redundancy for secondary objectives
- QP formulation handles multiple tasks and constraints
- Essential for humanoid and mobile manipulators

## Further Reading

- Sentis, L. "Synthesis and Control of Whole-Body Behaviors"
