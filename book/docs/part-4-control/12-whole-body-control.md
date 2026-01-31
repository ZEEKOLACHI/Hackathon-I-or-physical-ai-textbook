---
id: ch-4-12
title: Whole-Body Control
sidebar_position: 3
difficulty: advanced
estimated_time: 55
prerequisites: [ch-4-10, ch-4-11]
---

# Whole-Body Control

> "A humanoid robot is not a collection of independent limbs—it is one unified system that must move as a whole."
> — Luis Sentis, University of Texas at Austin

When a human reaches for a cup while maintaining balance, they don't control their arm independently of their legs. Every muscle, every joint works together in a coordinated symphony. Whole-body control brings this same holistic approach to robotics, coordinating all degrees of freedom simultaneously to achieve complex, dynamic behaviors.

## The Whole-Body Control Challenge

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    THE WHOLE-BODY CONTROL PROBLEM                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Humanoid Robot: 30+ Degrees of Freedom                                    │
│                                                                             │
│                    ┌───┐                                                    │
│                    │ O │  Head (3 DOF)                                      │
│                   /│ │ │\                                                   │
│                  / │ │ │ \                                                  │
│           Arm   /  │ │ │  \   Arm                                          │
│         (7 DOF)│   │ │ │   │(7 DOF)                                        │
│                │   ├───┤   │                                                │
│                │   │   │   │  Torso (3 DOF)                                 │
│                │   │   │   │                                                │
│                │   ├───┤   │                                                │
│                    │   │                                                    │
│              Leg  /│   │\  Leg                                             │
│           (6 DOF)/  │   │ \(6 DOF)                                         │
│                 /   │   │  \                                               │
│                ▼    ▼   ▼   ▼                                              │
│            ════════════════════  Ground                                    │
│                                                                             │
│   Challenge: Coordinate ALL joints to achieve MULTIPLE tasks:              │
│                                                                             │
│   • Primary: Reach target position with hand                               │
│   • Secondary: Maintain balance (CoM over support polygon)                 │
│   • Tertiary: Avoid joint limits                                           │
│   • Quaternary: Minimize energy consumption                                │
│                                                                             │
│   With: Contact constraints, joint limits, torque limits, dynamics         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Operational Space Formulation

The foundation of whole-body control is Oussama Khatib's Operational Space framework, which allows us to control the robot in task space rather than joint space.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    OPERATIONAL SPACE CONTROL                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Joint Space (q)                    Task Space (x)                         │
│   ┌───────────────┐                  ┌───────────────┐                     │
│   │ θ₁, θ₂, ... θₙ│  ◀───────────▶  │ x, y, z,      │                     │
│   │               │    Jacobian J    │ roll, pitch,  │                     │
│   │ n joints      │    J: q → x      │ yaw           │                     │
│   └───────────────┘                  └───────────────┘                     │
│                                                                             │
│   Dynamics in Joint Space:                                                  │
│   M(q)q̈ + C(q,q̇)q̇ + g(q) = τ + Jᵀfₑₓₜ                                    │
│                                                                             │
│   Dynamics in Task Space:                                                   │
│   Λ(x)ẍ + μ(x,ẋ) + p(x) = F + fₑₓₜ                                        │
│                                                                             │
│   Where:                                                                    │
│   Λ = (JM⁻¹Jᵀ)⁻¹           ← Task-space inertia                           │
│   μ = Λ(JM⁻¹C - J̇)q̇       ← Task-space Coriolis                          │
│   p = ΛJM⁻¹g               ← Task-space gravity                            │
│   F = Λẍ_des + μ + p       ← Task-space control force                      │
│   τ = JᵀF                   ← Joint torques to achieve F                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Task-Space Controller Implementation

```python
"""
Operational Space Control (Task-Space Control)

Enables control of robot end-effector directly in Cartesian space while
accounting for the full robot dynamics.

Key insight: Instead of thinking in joint angles, think in task coordinates
(position, velocity of end-effector). The mathematics handles the mapping.

Reference: Khatib, O. (1987) "A Unified Approach for Motion and Force Control
           of Robot Manipulators"
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
from abc import ABC, abstractmethod


@dataclass
class RobotState:
    """Complete state of a robot."""
    q: np.ndarray         # Joint positions
    q_dot: np.ndarray     # Joint velocities
    x: np.ndarray         # End-effector pose (6D or more)
    x_dot: np.ndarray     # End-effector velocity


class RobotDynamics(ABC):
    """Abstract interface for robot dynamics model."""

    @abstractmethod
    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """Return n×n mass matrix M(q)."""
        pass

    @abstractmethod
    def coriolis_vector(self, q: np.ndarray, q_dot: np.ndarray) -> np.ndarray:
        """Return n×1 Coriolis/centrifugal vector C(q,q̇)q̇."""
        pass

    @abstractmethod
    def gravity_vector(self, q: np.ndarray) -> np.ndarray:
        """Return n×1 gravity vector g(q)."""
        pass

    @abstractmethod
    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """Return m×n Jacobian J(q)."""
        pass

    @abstractmethod
    def jacobian_derivative(self, q: np.ndarray, q_dot: np.ndarray) -> np.ndarray:
        """Return m×n Jacobian time derivative J̇(q,q̇)."""
        pass


class TaskSpaceController:
    """
    Operational Space (Task-Space) Controller.

    Controls robot end-effector directly in Cartesian space using
    the full dynamics model.

    Block Diagram:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                                                                         │
    │   x_des ──┬──▶ ┌──────────┐                                            │
    │   ẋ_des ──┼──▶ │  Task    │     ┌───────────┐                          │
    │   ẍ_des ──┘    │  Space   │────▶│ Jacobian  │────▶ τ ────▶ Robot      │
    │                │ Dynamics │     │ Transpose │                          │
    │   x ─────────▶ │          │     └───────────┘                          │
    │   ẋ ─────────▶ └──────────┘                                            │
    │                                                                         │
    │   Key equations:                                                        │
    │   F = Λ·ẍ_des + μ + p    (Task-space dynamics)                        │
    │   τ = Jᵀ·F               (Map to joint torques)                        │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, robot: RobotDynamics, n_task_dims: int = 6):
        """
        Initialize task-space controller.

        Args:
            robot: Robot dynamics model
            n_task_dims: Number of task-space dimensions (typically 6)
        """
        self.robot = robot
        self.m = n_task_dims

        # Control gains
        self.Kp = 100.0 * np.eye(n_task_dims)  # Position gain
        self.Kd = 20.0 * np.eye(n_task_dims)   # Velocity gain

    def set_gains(self, kp: float, kd: float):
        """Set control gains (scalar applied to all dimensions)."""
        self.Kp = kp * np.eye(self.m)
        self.Kd = kd * np.eye(self.m)

    def set_gains_matrix(self, Kp: np.ndarray, Kd: np.ndarray):
        """Set control gains as matrices."""
        self.Kp = Kp
        self.Kd = Kd

    def compute_task_space_dynamics(
        self,
        q: np.ndarray,
        q_dot: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute task-space dynamics matrices.

        Args:
            q: Joint positions
            q_dot: Joint velocities

        Returns:
            Tuple of (Lambda, mu, p):
                Lambda: Task-space inertia matrix
                mu: Task-space Coriolis vector
                p: Task-space gravity vector
        """
        # Get joint-space dynamics
        M = self.robot.mass_matrix(q)
        C_q_dot = self.robot.coriolis_vector(q, q_dot)
        g = self.robot.gravity_vector(q)
        J = self.robot.jacobian(q)
        J_dot = self.robot.jacobian_derivative(q, q_dot)

        # Compute task-space inertia (operational space inertia)
        M_inv = np.linalg.inv(M)
        Lambda_inv = J @ M_inv @ J.T

        # Handle singularities
        try:
            Lambda = np.linalg.inv(Lambda_inv)
        except np.linalg.LinAlgError:
            # Near singularity - use damped least squares
            Lambda = np.linalg.inv(Lambda_inv + 0.01 * np.eye(self.m))

        # Compute task-space Coriolis
        # μ = Λ(JM⁻¹C - J̇)q̇
        mu = Lambda @ (J @ M_inv @ C_q_dot - J_dot @ q_dot)

        # Compute task-space gravity
        # p = ΛJM⁻¹g
        p = Lambda @ J @ M_inv @ g

        return Lambda, mu, p

    def compute_torques(
        self,
        x_des: np.ndarray,
        x_dot_des: np.ndarray,
        x: np.ndarray,
        x_dot: np.ndarray,
        q: np.ndarray,
        q_dot: np.ndarray,
        x_ddot_ff: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute joint torques for task-space control.

        Args:
            x_des: Desired task-space position
            x_dot_des: Desired task-space velocity
            x: Current task-space position
            x_dot: Current task-space velocity
            q: Current joint positions
            q_dot: Current joint velocities
            x_ddot_ff: Feedforward acceleration (optional)

        Returns:
            Joint torques
        """
        # Compute task-space dynamics
        Lambda, mu, p = self.compute_task_space_dynamics(q, q_dot)

        # Task-space PD control with feedforward
        x_ddot_des = (
            self.Kp @ (x_des - x) +
            self.Kd @ (x_dot_des - x_dot)
        )

        if x_ddot_ff is not None:
            x_ddot_des += x_ddot_ff

        # Task-space control force
        # F = Λẍ_des + μ + p (model-based control)
        F = Lambda @ x_ddot_des + mu + p

        # Map to joint torques
        J = self.robot.jacobian(q)
        tau = J.T @ F

        return tau


# Example usage with simulated 2-DOF planar robot
if __name__ == "__main__":
    print("Task-Space Control Demo")
    print("=" * 50)

    # Simple 2-link planar robot dynamics (for illustration)
    class PlanarRobot(RobotDynamics):
        def __init__(self, l1=1.0, l2=1.0, m1=1.0, m2=1.0):
            self.l1, self.l2 = l1, l2
            self.m1, self.m2 = m1, m2

        def mass_matrix(self, q):
            # Simplified - actual would include all inertia terms
            return np.array([
                [self.m1 + self.m2, 0],
                [0, self.m2]
            ])

        def coriolis_vector(self, q, q_dot):
            return np.zeros(2)

        def gravity_vector(self, q):
            g = 9.81
            return np.array([
                (self.m1 + self.m2) * g * self.l1 * np.sin(q[0]),
                self.m2 * g * self.l2 * np.sin(q[0] + q[1])
            ])

        def jacobian(self, q):
            s1 = np.sin(q[0])
            c1 = np.cos(q[0])
            s12 = np.sin(q[0] + q[1])
            c12 = np.cos(q[0] + q[1])

            return np.array([
                [-self.l1*s1 - self.l2*s12, -self.l2*s12],
                [self.l1*c1 + self.l2*c12, self.l2*c12]
            ])

        def jacobian_derivative(self, q, q_dot):
            return np.zeros((2, 2))

    robot = PlanarRobot()
    controller = TaskSpaceController(robot, n_task_dims=2)

    # Test at a configuration
    q = np.array([np.pi/4, np.pi/4])
    q_dot = np.zeros(2)
    x = np.array([1.0, 1.0])  # Current end-effector position
    x_dot = np.zeros(2)
    x_des = np.array([1.2, 0.8])  # Target position

    tau = controller.compute_torques(x_des, np.zeros(2), x, x_dot, q, q_dot)
    print(f"Joint configuration: {q}")
    print(f"Target position: {x_des}")
    print(f"Computed torques: {tau}")
```

## Null-Space Control

For redundant robots (more DOFs than task dimensions), the null space provides freedom to achieve secondary objectives without affecting the primary task.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       NULL-SPACE CONTROL                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Concept: Exploit redundancy for secondary objectives                      │
│                                                                             │
│   For a 7-DOF arm controlling 6D pose:                                      │
│   • Task requires 6 DOF                                                     │
│   • Robot has 7 DOF                                                         │
│   • 1 "extra" DOF = Null space (motions that don't affect end-effector)   │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   Joint Space (7D)          Task Space (6D)                         │  │
│   │   ┌─────────────┐           ┌─────────────┐                        │  │
│   │   │             │───────────│             │                        │  │
│   │   │   q (7×1)   │     J     │   x (6×1)   │                        │  │
│   │   │             │───────────│             │                        │  │
│   │   └─────────────┘           └─────────────┘                        │  │
│   │         │                                                          │  │
│   │         │ Null-space                                               │  │
│   │         ▼ (1D for this example)                                    │  │
│   │   ┌─────────────┐                                                  │  │
│   │   │ q_null (1×1)│  "Self-motion" - doesn't affect x                │  │
│   │   └─────────────┘                                                  │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Null-space projector:  N = I - J⁺J                                       │
│                                                                             │
│   Control law:  τ = τ_primary + N·τ_secondary                              │
│                                                                             │
│   Secondary objectives:                                                     │
│   • Joint limit avoidance                                                  │
│   • Obstacle avoidance                                                     │
│   • Singularity avoidance                                                  │
│   • Posture optimization                                                   │
│   • Energy minimization                                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Null-Space Controller Implementation

```python
"""
Null-Space Control for Redundant Robots

Enables simultaneous achievement of primary and secondary tasks by
exploiting kinematic redundancy.

For a robot with n joints controlling an m-dimensional task (n > m),
there are (n - m) degrees of redundancy that can be used for
secondary objectives.

Key mathematical tools:
- Pseudoinverse: J⁺ = Jᵀ(JJᵀ)⁻¹
- Null-space projector: N = I - J⁺J
- Projected secondary task: τ₂ = N·τ_secondary
"""

import numpy as np
from typing import List, Callable, Optional
from dataclasses import dataclass


@dataclass
class Task:
    """
    Represents a control task with priority.

    Higher priority tasks are satisfied first; lower priority tasks
    use remaining DOFs through null-space projection.
    """
    jacobian: np.ndarray       # Task Jacobian
    error: np.ndarray          # Task error (desired - current)
    priority: int              # Lower number = higher priority
    gain: float = 10.0         # Control gain


class NullSpaceController:
    """
    Null-space controller using task prioritization.

    Implements the classic null-space projection approach:
    τ = J₁ᵀF₁ + N₁(J₂ᵀF₂ + N₂(J₃ᵀF₃ + ...))

    Where N_i is the null-space projector for task i.
    """

    def __init__(self, n_joints: int):
        """
        Initialize null-space controller.

        Args:
            n_joints: Number of robot joints
        """
        self.n = n_joints
        self.damping = 0.01  # Damping for pseudoinverse

    def pseudoinverse(self, J: np.ndarray) -> np.ndarray:
        """
        Compute damped pseudoinverse.

        Uses damped least squares to handle singularities:
        J⁺ = Jᵀ(JJᵀ + λ²I)⁻¹

        Args:
            J: Jacobian matrix (m × n)

        Returns:
            Pseudoinverse (n × m)
        """
        m = J.shape[0]
        JJT = J @ J.T
        damped = JJT + self.damping**2 * np.eye(m)
        return J.T @ np.linalg.inv(damped)

    def null_space_projector(self, J: np.ndarray) -> np.ndarray:
        """
        Compute null-space projector.

        N = I - J⁺J

        Any vector multiplied by N will be projected into the
        null space of J (i.e., will produce no motion in task space).

        Args:
            J: Jacobian matrix

        Returns:
            Null-space projector (n × n)
        """
        J_pinv = self.pseudoinverse(J)
        return np.eye(self.n) - J_pinv @ J

    def compute_torques(
        self,
        primary_task_J: np.ndarray,
        primary_task_error: np.ndarray,
        secondary_task_torque: np.ndarray,
        primary_gain: float = 100.0
    ) -> np.ndarray:
        """
        Compute joint torques with null-space secondary task.

        τ = J₁ᵀΛ₁(Kp·e₁) + N₁·τ_secondary

        Args:
            primary_task_J: Primary task Jacobian
            primary_task_error: Primary task error
            secondary_task_torque: Torque for secondary objective
            primary_gain: Gain for primary task

        Returns:
            Joint torques
        """
        # Primary task
        J = primary_task_J
        J_pinv = self.pseudoinverse(J)

        # Task-space control
        x_ddot_des = primary_gain * primary_task_error
        tau_primary = J.T @ x_ddot_des

        # Null-space projector
        N = self.null_space_projector(J)

        # Project secondary task into null space
        tau_secondary = N @ secondary_task_torque

        return tau_primary + tau_secondary


class HierarchicalNullSpaceController:
    """
    Hierarchical null-space controller for multiple prioritized tasks.

    Implements strict task priority: higher priority tasks are
    guaranteed to be satisfied before lower priority tasks.

    Reference: Siciliano & Slotine (1991) "A General Framework for
               Managing Multiple Tasks in Highly Redundant Robotic Systems"
    """

    def __init__(self, n_joints: int, damping: float = 0.01):
        """
        Initialize hierarchical controller.

        Args:
            n_joints: Number of robot joints
            damping: Damping for pseudoinverse computation
        """
        self.n = n_joints
        self.damping = damping

    def compute_torques(self, tasks: List[Task]) -> np.ndarray:
        """
        Compute joint torques for hierarchical task execution.

        Tasks are sorted by priority and executed using cascaded
        null-space projection.

        Args:
            tasks: List of tasks with priorities

        Returns:
            Joint torques
        """
        # Sort tasks by priority (lowest number = highest priority)
        sorted_tasks = sorted(tasks, key=lambda t: t.priority)

        # Initialize
        tau = np.zeros(self.n)
        N = np.eye(self.n)  # Current null-space projector

        for task in sorted_tasks:
            J = task.jacobian
            e = task.error

            # Compute task contribution in current null space
            J_bar = J @ N  # Jacobian in null space of higher priority tasks

            # Damped pseudoinverse
            m = J_bar.shape[0]
            J_bar_pinv = J_bar.T @ np.linalg.inv(
                J_bar @ J_bar.T + self.damping**2 * np.eye(m)
            )

            # Task-space control
            x_ddot_des = task.gain * e
            tau_task = N @ J_bar_pinv @ x_ddot_des

            # Accumulate
            tau += tau_task

            # Update null-space projector for next (lower priority) task
            N = N @ (np.eye(self.n) - J_bar_pinv @ J_bar)

        return tau


class SecondaryObjectives:
    """
    Common secondary objectives for null-space control.
    """

    @staticmethod
    def joint_limit_avoidance(
        q: np.ndarray,
        q_min: np.ndarray,
        q_max: np.ndarray,
        gain: float = 10.0
    ) -> np.ndarray:
        """
        Compute torque to avoid joint limits.

        Uses a potential field that increases as joints approach limits.

        Args:
            q: Current joint positions
            q_min: Minimum joint limits
            q_max: Maximum joint limits
            gain: Repulsive gain

        Returns:
            Joint torques pushing away from limits
        """
        n = len(q)
        tau = np.zeros(n)

        for i in range(n):
            q_range = q_max[i] - q_min[i]
            q_mid = (q_max[i] + q_min[i]) / 2

            # Normalized position in range [-1, 1]
            q_norm = 2 * (q[i] - q_mid) / q_range

            # Repulsive potential gradient
            # Stronger as we approach limits
            tau[i] = -gain * q_norm / (1 - q_norm**2 + 0.01)

        return tau

    @staticmethod
    def manipulability_maximization(
        J: np.ndarray,
        gain: float = 1.0
    ) -> np.ndarray:
        """
        Compute torque to maximize manipulability.

        Manipulability w = sqrt(det(JJᵀ)) measures how far
        the robot is from singularities.

        Args:
            J: Current Jacobian
            gain: Optimization gain

        Returns:
            Joint velocities to increase manipulability

        Note: This returns velocity-level commands, needs
              integration for torque control.
        """
        # Compute manipulability Jacobian (gradient of w w.r.t. q)
        # This is approximate - full computation requires
        # derivatives of Jacobian w.r.t. joint positions

        JJT = J @ J.T
        w = np.sqrt(max(np.linalg.det(JJT), 1e-10))

        # Approximate gradient using numerical differentiation
        # In practice, analytical gradients are preferred
        n = J.shape[1]
        grad = np.zeros(n)

        # Simple heuristic: move toward configuration that increases w
        # Full implementation would use ∂w/∂q

        return gain * grad

    @staticmethod
    def preferred_posture(
        q: np.ndarray,
        q_preferred: np.ndarray,
        gain: float = 5.0
    ) -> np.ndarray:
        """
        Pull toward a preferred robot posture.

        Useful for maintaining a "home" or "comfortable" configuration.

        Args:
            q: Current joint positions
            q_preferred: Preferred joint positions
            gain: Attraction gain

        Returns:
            Joint torques toward preferred posture
        """
        return gain * (q_preferred - q)


# Example usage
if __name__ == "__main__":
    print("Null-Space Control Demo")
    print("=" * 50)

    # Simulated 7-DOF robot
    n_joints = 7
    controller = NullSpaceController(n_joints)

    # Primary task: 6D end-effector control
    J_primary = np.random.randn(6, 7)  # Task Jacobian
    error_primary = np.array([0.1, 0.05, 0.02, 0, 0, 0])  # Position/orientation error

    # Secondary task: Joint limit avoidance
    q = np.array([0, 0.5, -0.3, 0.8, 0, -0.2, 0])
    q_min = np.full(7, -2.0)
    q_max = np.full(7, 2.0)
    tau_secondary = SecondaryObjectives.joint_limit_avoidance(q, q_min, q_max)

    # Compute control
    tau = controller.compute_torques(
        primary_task_J=J_primary,
        primary_task_error=error_primary,
        secondary_task_torque=tau_secondary
    )

    print(f"Primary task error: {error_primary[:3]}")
    print(f"Secondary (joint limit) torques: {tau_secondary[:3]}...")
    print(f"Combined torques: {tau[:3]}...")
```

## Quadratic Programming for Whole-Body Control

Modern whole-body control often uses Quadratic Programming (QP) to optimally balance multiple objectives and constraints.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    QP-BASED WHOLE-BODY CONTROL                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Optimization Problem:                                                     │
│                                                                             │
│   minimize     ½ xᵀHx + fᵀx                                                │
│       x                                                                     │
│                                                                             │
│   subject to:  Ax = b        (Equality constraints: dynamics, contacts)    │
│                Gx ≤ h        (Inequality: joint/torque limits, friction)   │
│                                                                             │
│   Decision Variables x = [q̈, τ, λ]                                         │
│   • q̈: Joint accelerations                                                 │
│   • τ: Joint torques                                                        │
│   • λ: Contact forces                                                       │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                        QP Structure                                 │  │
│   │                                                                     │  │
│   │   Cost: Σ wᵢ||Jᵢq̈ - aᵢ_des||²   (Weighted task errors)            │  │
│   │         + wτ||τ||²               (Torque regularization)           │  │
│   │                                                                     │  │
│   │   Equality:                                                         │  │
│   │     M(q)q̈ + h(q,q̇) = Sτ + Jcᵀλ    (Dynamics)                      │  │
│   │     Jc q̈ + J̇c q̇ = 0               (Contact acceleration)          │  │
│   │                                                                     │  │
│   │   Inequality:                                                       │  │
│   │     τ_min ≤ τ ≤ τ_max             (Torque limits)                  │  │
│   │     q̈_min ≤ q̈ ≤ q̈_max             (Acceleration limits)           │  │
│   │     λ ∈ Friction Cone             (Contact feasibility)            │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### QP Controller Implementation

```python
"""
Quadratic Programming Based Whole-Body Controller

Uses optimization to find joint accelerations and torques that:
1. Track desired task accelerations
2. Satisfy dynamics equations
3. Respect joint/torque limits
4. Maintain contact constraints (friction cones)

This is the state-of-the-art approach used in modern humanoid robots.

Dependencies:
- quadprog or cvxpy for QP solving
- numpy for matrix operations
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ContactType(Enum):
    """Type of contact."""
    POINT = "point"      # Single point (e.g., fingertip)
    SURFACE = "surface"  # Planar surface (e.g., foot)


@dataclass
class Contact:
    """Contact information."""
    jacobian: np.ndarray     # Contact Jacobian
    position: np.ndarray     # Contact position
    normal: np.ndarray       # Surface normal
    mu: float                # Friction coefficient
    contact_type: ContactType


@dataclass
class TaskSpec:
    """Task specification for QP."""
    jacobian: np.ndarray     # Task Jacobian J
    desired_accel: np.ndarray  # Desired task acceleration ẍ_des
    weight: float            # Task weight in cost function


class QPWholeBodyController:
    """
    Quadratic Programming based Whole-Body Controller.

    Solves the following optimization at each control cycle:

    min   Σ wᵢ||Jᵢq̈ - ẍᵢ_des||² + wτ||τ||² + wλ||λ||²
    q̈,τ,λ

    s.t.  Mq̈ + h = Sτ + Jcᵀλ     (dynamics)
          Jc q̈ = -J̇c q̇           (contact constraint)
          τ_min ≤ τ ≤ τ_max       (torque limits)
          Aλ ≤ b                  (friction cone)
    """

    def __init__(
        self,
        n_joints: int,
        floating_base: bool = True,
        dt: float = 0.001
    ):
        """
        Initialize QP controller.

        Args:
            n_joints: Number of actuated joints
            floating_base: Whether robot has floating base (humanoid)
            dt: Control timestep
        """
        self.n_actuated = n_joints
        self.floating_base = floating_base
        self.dt = dt

        # State dimension: 6 (floating base) + n_actuated
        self.n_v = 6 + n_joints if floating_base else n_joints

        # Default limits
        self.tau_min = -100 * np.ones(n_joints)
        self.tau_max = 100 * np.ones(n_joints)
        self.q_ddot_max = 50 * np.ones(self.n_v)

        # Regularization weights
        self.w_tau = 0.001   # Torque regularization
        self.w_lambda = 0.001  # Force regularization

    def set_torque_limits(self, tau_min: np.ndarray, tau_max: np.ndarray):
        """Set joint torque limits."""
        self.tau_min = tau_min
        self.tau_max = tau_max

    def _build_friction_cone_constraints(
        self,
        contact: Contact,
        lambda_start_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build linear approximation of friction cone constraints.

        For a point contact with friction coefficient μ:
        |f_t| ≤ μ * f_n  (tangential force bounded by normal * friction)

        This is linearized using a polyhedral approximation.

        Returns:
            Tuple of (A, b) for constraint Aλ ≤ b
        """
        mu = contact.mu

        if contact.contact_type == ContactType.POINT:
            # 3D contact force: [fx, fy, fz] with fz being normal
            # Linearized friction cone with 4 faces
            # fx ≤ μ*fz, -fx ≤ μ*fz, fy ≤ μ*fz, -fy ≤ μ*fz

            A = np.array([
                [1, 0, -mu],   # fx - μ*fz ≤ 0
                [-1, 0, -mu],  # -fx - μ*fz ≤ 0
                [0, 1, -mu],   # fy - μ*fz ≤ 0
                [0, -1, -mu],  # -fy - μ*fz ≤ 0
                [0, 0, -1],    # -fz ≤ 0 (normal force positive)
            ])
            b = np.zeros(5)

        else:  # Surface contact
            # For a planar contact, we have force and torque
            # 6D wrench with ZMP constraints
            A = np.zeros((8, 6))  # Simplified
            b = np.zeros(8)

        return A, b

    def solve(
        self,
        M: np.ndarray,
        h: np.ndarray,
        tasks: List[TaskSpec],
        contacts: List[Contact],
        q_dot: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve the QP to get optimal accelerations and torques.

        Args:
            M: Mass matrix (n_v × n_v)
            h: Coriolis + gravity vector (n_v)
            tasks: List of task specifications
            contacts: List of active contacts
            q_dot: Current joint velocities

        Returns:
            Tuple of (q_ddot, tau, lambda):
                q_ddot: Joint accelerations
                tau: Joint torques
                lambda: Contact forces
        """
        # Count contact force dimensions
        n_lambda = sum(3 if c.contact_type == ContactType.POINT else 6
                       for c in contacts)

        # Decision variable: x = [q̈; τ; λ]
        n_x = self.n_v + self.n_actuated + n_lambda

        # Build cost function: ½xᵀHx + fᵀx
        H = np.zeros((n_x, n_x))
        f = np.zeros(n_x)

        # Add task costs
        for task in tasks:
            J = task.jacobian
            a_des = task.desired_accel
            w = task.weight

            # ||Jq̈ - a_des||² = q̈ᵀJᵀJq̈ - 2a_desᵀJq̈ + const
            H[:self.n_v, :self.n_v] += w * J.T @ J
            f[:self.n_v] -= w * J.T @ a_des

        # Add regularization
        # Torque regularization
        tau_start = self.n_v
        tau_end = self.n_v + self.n_actuated
        H[tau_start:tau_end, tau_start:tau_end] += self.w_tau * np.eye(self.n_actuated)

        # Force regularization
        lambda_start = tau_end
        if n_lambda > 0:
            H[lambda_start:, lambda_start:] += self.w_lambda * np.eye(n_lambda)

        # Build equality constraints: Ax = b
        n_eq = self.n_v  # Dynamics equation

        # Add contact acceleration constraints
        for contact in contacts:
            n_eq += contact.jacobian.shape[0]

        A_eq = np.zeros((n_eq, n_x))
        b_eq = np.zeros(n_eq)

        # Dynamics: Mq̈ - Sτ - Jcᵀλ = -h
        A_eq[:self.n_v, :self.n_v] = M

        # Selection matrix S (maps actuated torques to full dimension)
        if self.floating_base:
            S = np.zeros((self.n_v, self.n_actuated))
            S[6:, :] = np.eye(self.n_actuated)
        else:
            S = np.eye(self.n_actuated)

        A_eq[:self.n_v, tau_start:tau_end] = -S

        # Contact Jacobians
        lambda_idx = lambda_start
        for contact in contacts:
            Jc = contact.jacobian
            n_f = 3 if contact.contact_type == ContactType.POINT else 6
            A_eq[:self.n_v, lambda_idx:lambda_idx+n_f] = -Jc.T
            lambda_idx += n_f

        b_eq[:self.n_v] = -h

        # Contact acceleration constraints: Jc q̈ = -J̇c q̇
        row_idx = self.n_v
        for contact in contacts:
            Jc = contact.jacobian
            m = Jc.shape[0]
            A_eq[row_idx:row_idx+m, :self.n_v] = Jc
            # Note: J̇c q̇ should come from robot model
            b_eq[row_idx:row_idx+m] = 0  # Simplified
            row_idx += m

        # Build inequality constraints: Gx ≤ h
        # Count inequality constraints
        n_ineq = 2 * self.n_actuated  # Torque limits (upper and lower)
        n_ineq += 2 * self.n_v  # Acceleration limits

        # Friction cone constraints
        for contact in contacts:
            if contact.contact_type == ContactType.POINT:
                n_ineq += 5  # 4 friction cone faces + normal force positive
            else:
                n_ineq += 8  # Surface contact constraints

        G = np.zeros((n_ineq, n_x))
        h_ineq = np.zeros(n_ineq)

        row = 0

        # Torque limits: τ_min ≤ τ ≤ τ_max
        # τ ≤ τ_max  →  τ ≤ τ_max
        G[row:row+self.n_actuated, tau_start:tau_end] = np.eye(self.n_actuated)
        h_ineq[row:row+self.n_actuated] = self.tau_max
        row += self.n_actuated

        # -τ ≤ -τ_min  →  τ ≥ τ_min
        G[row:row+self.n_actuated, tau_start:tau_end] = -np.eye(self.n_actuated)
        h_ineq[row:row+self.n_actuated] = -self.tau_min
        row += self.n_actuated

        # Acceleration limits
        G[row:row+self.n_v, :self.n_v] = np.eye(self.n_v)
        h_ineq[row:row+self.n_v] = self.q_ddot_max
        row += self.n_v

        G[row:row+self.n_v, :self.n_v] = -np.eye(self.n_v)
        h_ineq[row:row+self.n_v] = self.q_ddot_max
        row += self.n_v

        # Friction cone constraints
        lambda_idx = lambda_start
        for contact in contacts:
            A_cone, b_cone = self._build_friction_cone_constraints(contact, lambda_idx)
            n_cone = A_cone.shape[0]
            n_f = A_cone.shape[1]

            G[row:row+n_cone, lambda_idx:lambda_idx+n_f] = A_cone
            h_ineq[row:row+n_cone] = b_cone
            row += n_cone
            lambda_idx += n_f

        # Solve QP
        # Using simplified solver (in practice, use quadprog, OSQP, or qpOASES)
        try:
            x_opt = self._solve_qp(H, f, A_eq, b_eq, G, h_ineq)
        except Exception as e:
            print(f"QP solve failed: {e}")
            x_opt = np.zeros(n_x)

        # Extract solution
        q_ddot = x_opt[:self.n_v]
        tau = x_opt[tau_start:tau_end]
        lambda_forces = x_opt[lambda_start:] if n_lambda > 0 else np.array([])

        return q_ddot, tau, lambda_forces

    def _solve_qp(
        self,
        H: np.ndarray,
        f: np.ndarray,
        A_eq: np.ndarray,
        b_eq: np.ndarray,
        G: np.ndarray,
        h: np.ndarray
    ) -> np.ndarray:
        """
        Solve QP using available solver.

        In production, use a fast QP solver like:
        - qpOASES (C++)
        - OSQP (open source)
        - quadprog (Python)
        - Gurobi (commercial)

        This is a simplified placeholder.
        """
        # Placeholder: solve unconstrained then project
        # Real implementation should use proper QP solver

        try:
            from scipy.optimize import minimize

            def objective(x):
                return 0.5 * x @ H @ x + f @ x

            def eq_constraint(x):
                return A_eq @ x - b_eq

            def ineq_constraint(x):
                return h - G @ x

            constraints = [
                {'type': 'eq', 'fun': eq_constraint},
                {'type': 'ineq', 'fun': ineq_constraint}
            ]

            n = H.shape[0]
            x0 = np.zeros(n)
            result = minimize(objective, x0, method='SLSQP', constraints=constraints)
            return result.x

        except Exception:
            # Fallback: solve only equality constrained problem
            n = H.shape[0]

            # Solve with equality constraints using KKT
            n_eq = A_eq.shape[0]
            KKT = np.block([
                [H, A_eq.T],
                [A_eq, np.zeros((n_eq, n_eq))]
            ])
            rhs = np.concatenate([-f, b_eq])

            try:
                sol = np.linalg.solve(KKT, rhs)
                return sol[:n]
            except np.linalg.LinAlgError:
                return np.zeros(n)


# Example: Humanoid balancing while reaching
if __name__ == "__main__":
    print("QP Whole-Body Control Demo")
    print("=" * 50)

    # Simplified humanoid with 12 actuated joints
    n_joints = 12
    controller = QPWholeBodyController(n_joints, floating_base=True)

    # Mock dynamics
    n_v = 6 + n_joints  # 6 floating base + joints
    M = np.eye(n_v) * 10  # Mass matrix
    h = np.zeros(n_v)     # Coriolis + gravity

    # Tasks
    # 1. Right hand tracking
    J_hand = np.random.randn(6, n_v)
    hand_task = TaskSpec(
        jacobian=J_hand,
        desired_accel=np.array([1.0, 0, 0, 0, 0, 0]),  # Reach forward
        weight=10.0
    )

    # 2. Center of mass tracking (for balance)
    J_com = np.random.randn(3, n_v)
    com_task = TaskSpec(
        jacobian=J_com,
        desired_accel=np.zeros(3),  # Keep CoM stationary
        weight=100.0  # High priority for balance
    )

    tasks = [com_task, hand_task]

    # Contacts (both feet on ground)
    left_foot = Contact(
        jacobian=np.random.randn(6, n_v),
        position=np.array([0, 0.1, 0]),
        normal=np.array([0, 0, 1]),
        mu=0.8,
        contact_type=ContactType.SURFACE
    )
    right_foot = Contact(
        jacobian=np.random.randn(6, n_v),
        position=np.array([0, -0.1, 0]),
        normal=np.array([0, 0, 1]),
        mu=0.8,
        contact_type=ContactType.SURFACE
    )

    contacts = [left_foot, right_foot]

    # Solve
    q_dot = np.zeros(n_v)
    q_ddot, tau, forces = controller.solve(M, h, tasks, contacts, q_dot)

    print(f"Base acceleration: {q_ddot[:6]}")
    print(f"Joint accelerations: {q_ddot[6:9]}...")
    print(f"Joint torques: {tau[:3]}...")
    print(f"Contact forces: {forces[:3]}..." if len(forces) > 0 else "No forces")
```

## Contact-Consistent Control

For robots in contact with the environment, control must respect contact constraints:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CONTACT-CONSISTENT CONTROL                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Contact Constraints:                                                      │
│                                                                             │
│   1. NO SLIP (friction cone):                                              │
│      |f_tangent| ≤ μ · f_normal                                            │
│                                                                             │
│                    f_normal                                                 │
│                       ▲                                                     │
│                       │                                                     │
│                      /│\      Friction Cone                                │
│                     / │ \     (μ = tan(θ))                                 │
│                    /  │  \                                                 │
│                   /   │   \                                                │
│                  /    │    \                                               │
│                 ──────┼──────                                              │
│              f_t      │      f_t                                           │
│                       ▼                                                     │
│                                                                             │
│   2. NO PENETRATION:                                                        │
│      ẍ_contact · n ≥ 0  (acceleration into surface not allowed)           │
│                                                                             │
│   3. COMPLEMENTARITY:                                                       │
│      f_normal · gap = 0   (force only when in contact)                     │
│      f_normal ≥ 0, gap ≥ 0                                                 │
│                                                                             │
│   4. UNILATERAL (for point contacts):                                       │
│      f_normal ≥ 0   (can only push, not pull)                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

```python
"""
Contact-Consistent Whole-Body Control

Ensures that computed motions and forces are physically realizable
given the contact state.

Key concepts:
- Friction cone: Limits on tangential forces
- Support polygon: Region where CoM can be for static balance
- ZMP: Zero Moment Point for dynamic balance
"""

import numpy as np
from typing import List, Tuple


class ContactConsistentController:
    """
    Controller that maintains contact consistency.

    Ensures:
    1. Contact forces stay within friction cones
    2. CoM stays over support region (for balance)
    3. No unintended contact breaking
    """

    def __init__(self, friction_coefficient: float = 0.7):
        self.mu = friction_coefficient

    def compute_support_polygon(
        self,
        contact_positions: List[np.ndarray]
    ) -> np.ndarray:
        """
        Compute convex hull of contact points projected to ground.

        The support polygon is the region where the CoM projection
        must lie for static stability.

        Args:
            contact_positions: List of contact point positions

        Returns:
            Vertices of support polygon (N × 2)
        """
        from scipy.spatial import ConvexHull

        # Project to XY plane
        points_2d = np.array([[p[0], p[1]] for p in contact_positions])

        if len(points_2d) < 3:
            return points_2d

        try:
            hull = ConvexHull(points_2d)
            return points_2d[hull.vertices]
        except Exception:
            return points_2d

    def point_in_polygon(
        self,
        point: np.ndarray,
        polygon: np.ndarray,
        margin: float = 0.0
    ) -> bool:
        """
        Check if point is inside polygon (with optional margin).

        Args:
            point: 2D point to check
            polygon: Polygon vertices (N × 2)
            margin: Safety margin (shrinks polygon)

        Returns:
            True if point is inside polygon
        """
        n = len(polygon)
        if n < 3:
            return False

        for i in range(n):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % n]

            # Edge normal (pointing inward)
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            normal = normal / (np.linalg.norm(normal) + 1e-10)

            # Distance from point to edge
            dist = np.dot(point - p1, normal)

            if dist < -margin:
                return False

        return True

    def compute_zmp(
        self,
        com_pos: np.ndarray,
        com_accel: np.ndarray,
        gravity: float = 9.81
    ) -> np.ndarray:
        """
        Compute Zero Moment Point (ZMP).

        ZMP is where the total ground reaction force acts to balance
        the robot's motion. For dynamic balance, ZMP must be inside
        the support polygon.

        Args:
            com_pos: Center of mass position [x, y, z]
            com_accel: Center of mass acceleration
            gravity: Gravitational acceleration

        Returns:
            ZMP position [x, y]

        ZMP formula:
            p_zmp = p_com - (z_com / (g + z_ddot)) * [x_ddot, y_ddot]
        """
        z = com_pos[2]
        z_ddot = com_accel[2]

        denominator = gravity + z_ddot
        if abs(denominator) < 0.01:
            denominator = 0.01 * np.sign(denominator)

        zmp_x = com_pos[0] - z / denominator * com_accel[0]
        zmp_y = com_pos[1] - z / denominator * com_accel[1]

        return np.array([zmp_x, zmp_y])

    def check_friction_cone(
        self,
        force: np.ndarray,
        normal: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Check if force is within friction cone.

        Args:
            force: Contact force [fx, fy, fz]
            normal: Surface normal (pointing up)

        Returns:
            Tuple of (is_valid, margin):
                is_valid: True if force is in friction cone
                margin: How much margin to cone boundary (negative if outside)
        """
        # Normal component
        f_n = np.dot(force, normal)

        # Tangential component
        f_t = force - f_n * normal
        f_t_mag = np.linalg.norm(f_t)

        # Friction constraint: |f_t| ≤ μ * f_n
        if f_n <= 0:
            return False, -1.0

        margin = self.mu * f_n - f_t_mag
        return margin >= 0, margin

    def project_to_friction_cone(
        self,
        force: np.ndarray,
        normal: np.ndarray
    ) -> np.ndarray:
        """
        Project force to nearest point on friction cone.

        Useful for ensuring contact forces are physically realizable.

        Args:
            force: Desired contact force
            normal: Surface normal

        Returns:
            Projected force (on cone surface if original was outside)
        """
        # Normal component
        f_n = np.dot(force, normal)

        # Handle negative normal force
        if f_n <= 0:
            return np.zeros_like(force)

        # Tangential component
        f_t = force - f_n * normal
        f_t_mag = np.linalg.norm(f_t)

        # Check if inside cone
        if f_t_mag <= self.mu * f_n:
            return force  # Already valid

        # Project to cone surface
        if f_t_mag > 1e-10:
            f_t_unit = f_t / f_t_mag
        else:
            f_t_unit = np.zeros_like(f_t)

        # On cone: f_t = μ * f_n
        # Also need to find the point on cone closest to original force
        # This requires some geometry

        # Simplified: scale tangential component to cone boundary
        projected = f_n * normal + self.mu * f_n * f_t_unit

        return projected


# Demonstration
if __name__ == "__main__":
    print("Contact-Consistent Control Demo")
    print("=" * 50)

    controller = ContactConsistentController(friction_coefficient=0.7)

    # Define contacts (standing on two feet)
    contacts = [
        np.array([0.0, 0.1, 0.0]),   # Left foot
        np.array([0.2, 0.1, 0.0]),   # Left foot front
        np.array([0.0, -0.1, 0.0]),  # Right foot
        np.array([0.2, -0.1, 0.0])   # Right foot front
    ]

    # Compute support polygon
    support_polygon = controller.compute_support_polygon(contacts)
    print(f"Support polygon vertices:\n{support_polygon}")

    # Check CoM position
    com_pos = np.array([0.1, 0.0, 0.9])  # CoM at center, 0.9m high
    com_in_support = controller.point_in_polygon(
        com_pos[:2], support_polygon, margin=0.02
    )
    print(f"CoM at {com_pos[:2]}: {'Inside' if com_in_support else 'Outside'} support")

    # Compute ZMP for given CoM acceleration
    com_accel = np.array([0.5, 0.0, 0.0])  # Accelerating forward
    zmp = controller.compute_zmp(com_pos, com_accel)
    print(f"ZMP position: {zmp}")

    zmp_in_support = controller.point_in_polygon(zmp, support_polygon, margin=0.01)
    print(f"ZMP: {'Stable' if zmp_in_support else 'Unstable'}")

    # Check friction cone
    force = np.array([5.0, 3.0, 50.0])  # Force with tangential component
    normal = np.array([0, 0, 1])
    valid, margin = controller.check_friction_cone(force, normal)
    print(f"Force {force}: {'Valid' if valid else 'Invalid'}, margin={margin:.2f}N")
```

## Industry Perspective: Whole-Body Control in Practice

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                WHOLE-BODY CONTROL IN INDUSTRY                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   BOSTON DYNAMICS ATLAS                                                     │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │ • QP-based whole-body control at 1kHz                           │      │
│   │ • Model Predictive Control for walking                          │      │
│   │ • Hybrid system: continuous dynamics + discrete contacts        │      │
│   │ • Real-time contact estimation and replanning                   │      │
│   │ • 28 hydraulic actuators, 6D force/torque sensing              │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│   TESLA OPTIMUS                                                             │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │ • Custom actuators with torque sensing                          │      │
│   │ • Hierarchical control: balance → locomotion → manipulation    │      │
│   │ • Learning-based motion adaptation                              │      │
│   │ • Focus on efficiency and cost-effective design                 │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│   FIGURE AI                                                                 │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │ • Operational space control framework                           │      │
│   │ • Real-time collision avoidance in null space                  │      │
│   │ • Compliance control for safe human interaction                 │      │
│   │ • Integration with LLM-based task planning                      │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│   Key Research Groups:                                                      │
│   • Stanford (Khatib): Operational space formulation                       │
│   • MIT (Russ Tedrake): Optimization-based control                         │
│   • ETH Zurich: Legged locomotion                                          │
│   • UT Austin (Sentis): Whole-body prioritized control                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Summary

### Key Takeaways

1. **Operational Space Control**: Control robots in task space (Cartesian) while respecting full dynamics

2. **Null-Space Control**: Use redundancy for secondary objectives without affecting primary task

3. **QP-Based Control**: Modern approach that optimally handles multiple tasks and constraints simultaneously

4. **Contact Consistency**: All motions and forces must be physically realizable given contact constraints

5. **Hierarchical Priorities**: Critical for complex robots—balance comes before manipulation

### Control Strategy Comparison

| Approach | Pros | Cons | Best For |
|----------|------|------|----------|
| Task-Space | Intuitive, Cartesian goals | Single task focus | Simple manipulation |
| Null-Space | Multiple objectives | Priority tuning needed | Redundant arms |
| QP-Based | Optimal, handles constraints | Computationally heavy | Full humanoids |
| Hierarchical | Strict priorities | May sacrifice low-priority | Safety-critical |

### Design Checklist

- [ ] Identified all tasks and their priorities
- [ ] Modeled contact state (which links are in contact)
- [ ] Set appropriate friction coefficients
- [ ] Defined joint and torque limits
- [ ] Selected QP solver appropriate for timing requirements
- [ ] Implemented balance maintenance as high priority
- [ ] Tested singularity handling
- [ ] Validated contact force feasibility

## Further Reading

### Foundational Works
- Khatib, O. (1987). "A Unified Approach for Motion and Force Control of Robot Manipulators"
- Sentis, L. & Khatib, O. (2005). "Synthesis of Whole-Body Behaviors through Hierarchical Control"

### Modern Approaches
- Wensing, P. & Orin, D. (2013). "High-speed humanoid running through control with a 3D-SLIP model"
- Bellicoso, C.D. et al. (2016). "Perception-less terrain adaptation through whole body control"

### Software
- Drake (MIT): Optimization-based robotics toolbox
- Pinocchio: Fast rigid body dynamics
- qpOASES: Real-time QP solver
- OSQP: Operator Splitting QP solver

---

*"The art of whole-body control is orchestrating dozens of joints to move as one, like a symphony where every instrument matters."*
