---
id: ch-6-16
title: Humanoid Kinematics
sidebar_position: 1
difficulty: advanced
estimated_time: 60
prerequisites: [ch-4-12]
---

# Humanoid Kinematics

> "The human body is the best picture of the human soul."
> — Ludwig Wittgenstein

Humanoid robots present unique kinematic challenges due to their high degree-of-freedom structures, multiple kinematic chains, and the need for whole-body coordination. This chapter develops the mathematical framework for analyzing humanoid motion, from individual joint transformations to full-body center of mass computation.

## The Humanoid Kinematic Structure

Unlike industrial manipulators with a single kinematic chain, humanoids feature **tree-structured** kinematics with multiple branches sharing a common base (the torso or pelvis).

```
                    Humanoid Kinematic Tree

                         ┌─────────┐
                         │  Head   │ ← 2-3 DOF (pan, tilt, roll)
                         └────┬────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
    ┌────┴────┐         ┌─────┴─────┐        ┌────┴────┐
    │  L.Arm  │         │   Torso   │        │  R.Arm  │
    │  7 DOF  │         │   3 DOF   │        │  7 DOF  │
    └────┬────┘         │  (waist)  │        └────┬────┘
         │              └─────┬─────┘              │
    ┌────┴────┐               │              ┌────┴────┐
    │ L.Hand  │         ┌─────┴─────┐        │ R.Hand  │
    │  5 DOF  │         │   Pelvis  │        │  5 DOF  │
    └─────────┘         │  (base)   │        └─────────┘
                        └─────┬─────┘
              ┌───────────────┼───────────────┐
              │                               │
         ┌────┴────┐                     ┌────┴────┐
         │  L.Leg  │                     │  R.Leg  │
         │  6 DOF  │                     │  6 DOF  │
         └────┬────┘                     └────┬────┘
              │                               │
         ┌────┴────┐                     ┌────┴────┐
         │ L.Foot  │                     │ R.Foot  │
         └─────────┘                     └─────────┘

    Total: ~30-50+ DOF depending on hand complexity
```

### Degrees of Freedom Analysis

```python
"""
Humanoid DOF Configuration Module

This module defines standard humanoid robot configurations
and their degree-of-freedom distributions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import numpy as np


class JointType(Enum):
    """Types of joints found in humanoid robots."""
    REVOLUTE = "revolute"      # Single-axis rotation
    PRISMATIC = "prismatic"    # Linear translation
    SPHERICAL = "spherical"    # Ball-and-socket (3 DOF)
    UNIVERSAL = "universal"    # 2-DOF rotation
    FIXED = "fixed"            # No motion


@dataclass
class JointSpec:
    """Specification for a single joint."""
    name: str
    joint_type: JointType
    dof: int
    limits_lower: np.ndarray  # Joint angle/position lower limits
    limits_upper: np.ndarray  # Joint angle/position upper limits
    max_velocity: np.ndarray  # Maximum joint velocities
    max_torque: np.ndarray    # Maximum joint torques

    def __post_init__(self):
        """Validate joint specification."""
        assert len(self.limits_lower) == self.dof
        assert len(self.limits_upper) == self.dof
        assert np.all(self.limits_lower <= self.limits_upper)


@dataclass
class LimbConfig:
    """Configuration for a robot limb."""
    name: str
    joints: List[JointSpec]
    total_dof: int = field(init=False)

    def __post_init__(self):
        self.total_dof = sum(j.dof for j in self.joints)


@dataclass
class HumanoidConfig:
    """Complete humanoid robot configuration."""
    name: str
    limbs: Dict[str, LimbConfig]
    total_dof: int = field(init=False)

    def __post_init__(self):
        self.total_dof = sum(limb.total_dof for limb in self.limbs.values())

    def get_joint_limits(self) -> tuple:
        """Return concatenated joint limits for all joints."""
        lower = []
        upper = []
        for limb in self.limbs.values():
            for joint in limb.joints:
                lower.extend(joint.limits_lower)
                upper.extend(joint.limits_upper)
        return np.array(lower), np.array(upper)


def create_standard_humanoid() -> HumanoidConfig:
    """
    Create a standard humanoid robot configuration.

    Based on typical research humanoids like NAO, Atlas, or similar.

    Returns:
        HumanoidConfig: Complete humanoid specification
    """
    # Head configuration (2 DOF: pan + tilt)
    head = LimbConfig(
        name="head",
        joints=[
            JointSpec("head_pan", JointType.REVOLUTE, 1,
                     np.array([-2.0]), np.array([2.0]),
                     np.array([3.0]), np.array([10.0])),
            JointSpec("head_tilt", JointType.REVOLUTE, 1,
                     np.array([-0.5]), np.array([0.5]),
                     np.array([3.0]), np.array([10.0]))
        ]
    )

    # Torso configuration (3 DOF: yaw, pitch, roll)
    torso = LimbConfig(
        name="torso",
        joints=[
            JointSpec("torso_yaw", JointType.REVOLUTE, 1,
                     np.array([-1.5]), np.array([1.5]),
                     np.array([2.0]), np.array([200.0])),
            JointSpec("torso_pitch", JointType.REVOLUTE, 1,
                     np.array([-0.5]), np.array([1.0]),
                     np.array([2.0]), np.array([200.0])),
            JointSpec("torso_roll", JointType.REVOLUTE, 1,
                     np.array([-0.5]), np.array([0.5]),
                     np.array([2.0]), np.array([200.0]))
        ]
    )

    # Arm configuration (7 DOF each)
    def create_arm(side: str) -> LimbConfig:
        prefix = f"{side}_arm"
        return LimbConfig(
            name=prefix,
            joints=[
                JointSpec(f"{prefix}_shoulder_pitch", JointType.REVOLUTE, 1,
                         np.array([-2.0]), np.array([2.0]),
                         np.array([3.0]), np.array([80.0])),
                JointSpec(f"{prefix}_shoulder_roll", JointType.REVOLUTE, 1,
                         np.array([-1.5]), np.array([1.5]),
                         np.array([3.0]), np.array([80.0])),
                JointSpec(f"{prefix}_shoulder_yaw", JointType.REVOLUTE, 1,
                         np.array([-2.0]), np.array([2.0]),
                         np.array([3.0]), np.array([40.0])),
                JointSpec(f"{prefix}_elbow", JointType.REVOLUTE, 1,
                         np.array([0.0]), np.array([2.5]),
                         np.array([3.0]), np.array([40.0])),
                JointSpec(f"{prefix}_wrist_yaw", JointType.REVOLUTE, 1,
                         np.array([-2.0]), np.array([2.0]),
                         np.array([4.0]), np.array([10.0])),
                JointSpec(f"{prefix}_wrist_pitch", JointType.REVOLUTE, 1,
                         np.array([-1.0]), np.array([1.0]),
                         np.array([4.0]), np.array([10.0])),
                JointSpec(f"{prefix}_wrist_roll", JointType.REVOLUTE, 1,
                         np.array([-2.0]), np.array([2.0]),
                         np.array([4.0]), np.array([10.0]))
            ]
        )

    # Leg configuration (6 DOF each)
    def create_leg(side: str) -> LimbConfig:
        prefix = f"{side}_leg"
        return LimbConfig(
            name=prefix,
            joints=[
                JointSpec(f"{prefix}_hip_yaw", JointType.REVOLUTE, 1,
                         np.array([-0.8]), np.array([0.8]),
                         np.array([2.5]), np.array([150.0])),
                JointSpec(f"{prefix}_hip_roll", JointType.REVOLUTE, 1,
                         np.array([-0.5]), np.array([0.5]),
                         np.array([2.5]), np.array([150.0])),
                JointSpec(f"{prefix}_hip_pitch", JointType.REVOLUTE, 1,
                         np.array([-1.5]), np.array([1.5]),
                         np.array([2.5]), np.array([200.0])),
                JointSpec(f"{prefix}_knee", JointType.REVOLUTE, 1,
                         np.array([0.0]), np.array([2.5]),
                         np.array([2.5]), np.array([200.0])),
                JointSpec(f"{prefix}_ankle_pitch", JointType.REVOLUTE, 1,
                         np.array([-1.0]), np.array([1.0]),
                         np.array([3.0]), np.array([100.0])),
                JointSpec(f"{prefix}_ankle_roll", JointType.REVOLUTE, 1,
                         np.array([-0.5]), np.array([0.5]),
                         np.array([3.0]), np.array([100.0]))
            ]
        )

    return HumanoidConfig(
        name="standard_humanoid",
        limbs={
            "head": head,
            "torso": torso,
            "left_arm": create_arm("left"),
            "right_arm": create_arm("right"),
            "left_leg": create_leg("left"),
            "right_leg": create_leg("right")
        }
    )
```

## Denavit-Hartenberg Convention

The **Denavit-Hartenberg (DH) convention** provides a systematic method for assigning coordinate frames to each link of a kinematic chain.

```
            DH Parameter Definition

    Link i-1                           Link i
    ═══════╗                         ╔═══════
           ║     aᵢ (link length)    ║
           ║◄────────────────────────║
           ║         Zᵢ₋₁            ║  Zᵢ
           ║          ↑              ║   ↑
           ║    θᵢ    │    αᵢ       ║   │
           ║    ↺     │    ↺        ║   │
           ╚══════════╪══════════════╝   │
                      │                   │
              ────────┼──► Xᵢ₋₁          │
                  dᵢ  │                   │
                      ▼              ─────┼──► Xᵢ

    Four DH Parameters:
    ───────────────────
    θᵢ : Joint angle (rotation about Zᵢ₋₁)
    dᵢ : Link offset (translation along Zᵢ₋₁)
    aᵢ : Link length (translation along Xᵢ)
    αᵢ : Link twist (rotation about Xᵢ)

    For revolute joints: θ is variable, d is constant
    For prismatic joints: d is variable, θ is constant
```

### DH Transformation Implementation

```python
"""
Denavit-Hartenberg Transformation Module

Implements DH transformations and forward kinematics
for humanoid robot limbs.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DHParameter:
    """
    Denavit-Hartenberg parameters for a single joint.

    Attributes:
        theta_offset: Fixed offset added to joint variable (for revolute)
        d: Link offset along previous z-axis
        a: Link length along rotated x-axis
        alpha: Link twist about rotated x-axis
        joint_type: 'revolute' or 'prismatic'
    """
    theta_offset: float
    d: float
    a: float
    alpha: float
    joint_type: str = 'revolute'


class DHKinematics:
    """
    Forward kinematics using Denavit-Hartenberg convention.

    This class handles transformation computations for serial
    kinematic chains using standard DH parameters.
    """

    def __init__(self, dh_params: List[DHParameter]):
        """
        Initialize DH kinematics.

        Args:
            dh_params: List of DH parameters for each joint
        """
        self.dh_params = dh_params
        self.n_joints = len(dh_params)

    @staticmethod
    def dh_transform(theta: float, d: float,
                     a: float, alpha: float) -> np.ndarray:
        """
        Compute homogeneous transformation from DH parameters.

        The transformation is computed as:
        T = Rz(θ) · Tz(d) · Tx(a) · Rx(α)

        Args:
            theta: Joint angle (rotation about z-axis)
            d: Link offset (translation along z-axis)
            a: Link length (translation along x-axis)
            alpha: Link twist (rotation about x-axis)

        Returns:
            4x4 homogeneous transformation matrix
        """
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)

        return np.array([
            [ct, -st * ca,  st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [0,   sa,       ca,      d     ],
            [0,   0,        0,       1     ]
        ])

    def forward_kinematics(self, joint_values: np.ndarray,
                          up_to_joint: Optional[int] = None) -> np.ndarray:
        """
        Compute forward kinematics for the chain.

        Args:
            joint_values: Array of joint values (angles for revolute,
                         displacements for prismatic)
            up_to_joint: Compute FK only up to this joint (inclusive).
                        If None, compute full chain.

        Returns:
            4x4 homogeneous transformation from base to end-effector
        """
        if up_to_joint is None:
            up_to_joint = self.n_joints - 1

        T = np.eye(4)

        for i in range(up_to_joint + 1):
            param = self.dh_params[i]

            if param.joint_type == 'revolute':
                theta = joint_values[i] + param.theta_offset
                d = param.d
            else:  # prismatic
                theta = param.theta_offset
                d = joint_values[i] + param.d

            T = T @ self.dh_transform(theta, d, param.a, param.alpha)

        return T

    def get_all_transforms(self, joint_values: np.ndarray) -> List[np.ndarray]:
        """
        Get transformation matrices for all links.

        Useful for visualization and collision checking.

        Args:
            joint_values: Array of joint values

        Returns:
            List of 4x4 transforms, one for each link frame
        """
        transforms = []
        T = np.eye(4)

        for i, param in enumerate(self.dh_params):
            if param.joint_type == 'revolute':
                theta = joint_values[i] + param.theta_offset
                d = param.d
            else:
                theta = param.theta_offset
                d = joint_values[i] + param.d

            T = T @ self.dh_transform(theta, d, param.a, param.alpha)
            transforms.append(T.copy())

        return transforms


def create_7dof_arm_dh() -> List[DHParameter]:
    """
    Create DH parameters for a 7-DOF humanoid arm.

    Configuration: shoulder (3 DOF) + elbow (1 DOF) + wrist (3 DOF)

    Returns:
        List of DH parameters
    """
    # Typical measurements in meters
    d1 = 0.0      # Shoulder to shoulder roll
    d3 = 0.30     # Upper arm length
    d5 = 0.25     # Forearm length
    d7 = 0.10     # Wrist to end-effector

    return [
        DHParameter(0, 0, 0, -np.pi/2),        # Shoulder pitch
        DHParameter(-np.pi/2, 0, 0, np.pi/2),  # Shoulder roll
        DHParameter(0, d3, 0, -np.pi/2),       # Shoulder yaw
        DHParameter(0, 0, 0, np.pi/2),         # Elbow
        DHParameter(0, d5, 0, -np.pi/2),       # Wrist yaw
        DHParameter(0, 0, 0, np.pi/2),         # Wrist pitch
        DHParameter(0, d7, 0, 0)               # Wrist roll
    ]


def create_6dof_leg_dh() -> List[DHParameter]:
    """
    Create DH parameters for a 6-DOF humanoid leg.

    Configuration: hip (3 DOF) + knee (1 DOF) + ankle (2 DOF)

    Returns:
        List of DH parameters
    """
    # Typical measurements in meters
    hip_offset = 0.10    # Lateral hip offset
    thigh_length = 0.40  # Upper leg length
    shank_length = 0.40  # Lower leg length
    foot_height = 0.05   # Ankle to ground

    return [
        DHParameter(0, 0, hip_offset, -np.pi/2),      # Hip yaw
        DHParameter(-np.pi/2, 0, 0, np.pi/2),         # Hip roll
        DHParameter(0, 0, 0, 0),                       # Hip pitch
        DHParameter(0, -thigh_length, 0, 0),          # Knee
        DHParameter(0, -shank_length, 0, 0),          # Ankle pitch
        DHParameter(-np.pi/2, 0, 0, np.pi/2)          # Ankle roll
    ]
```

## Jacobian Analysis

The **Jacobian matrix** maps joint velocities to end-effector velocities, crucial for control and motion planning.

```
             Jacobian Matrix Structure

    ┌                                         ┐   ┌     ┐
    │  ∂x/∂q₁  ∂x/∂q₂  ...  ∂x/∂qₙ          │   │ q̇₁ │
    │  ∂y/∂q₁  ∂y/∂q₂  ...  ∂y/∂qₙ          │   │ q̇₂ │
    │  ∂z/∂q₁  ∂z/∂q₂  ...  ∂z/∂qₙ          │ × │ .  │ = ẋ
    │  ∂ωx/∂q₁ ∂ωx/∂q₂ ... ∂ωx/∂qₙ          │   │ .  │
    │  ∂ωy/∂q₁ ∂ωy/∂q₂ ... ∂ωy/∂qₙ          │   │ .  │
    │  ∂ωz/∂q₁ ∂ωz/∂q₂ ... ∂ωz/∂qₙ          │   │ q̇ₙ │
    └                                         ┘   └     ┘

             J (6×n)                    q̇        ẋ (6×1)

    Linear velocity:  v = Jᵥ · q̇   (upper 3 rows)
    Angular velocity: ω = Jω · q̇   (lower 3 rows)
```

### Jacobian Implementation

```python
"""
Jacobian Computation Module

Implements geometric and analytical Jacobian computation
for humanoid robot limbs.
"""

import numpy as np
from typing import List, Optional


class JacobianComputer:
    """
    Computes Jacobian matrices for serial kinematic chains.

    Supports both geometric and numerical Jacobian computation
    with utilities for singularity analysis.
    """

    def __init__(self, kinematics: DHKinematics):
        """
        Initialize Jacobian computer.

        Args:
            kinematics: DHKinematics object for the chain
        """
        self.kin = kinematics
        self.n_joints = kinematics.n_joints

    def geometric_jacobian(self, joint_values: np.ndarray) -> np.ndarray:
        """
        Compute the geometric Jacobian matrix.

        For a revolute joint i:
            Jᵥᵢ = zᵢ₋₁ × (pₑ - pᵢ₋₁)
            Jωᵢ = zᵢ₋₁

        For a prismatic joint i:
            Jᵥᵢ = zᵢ₋₁
            Jωᵢ = 0

        Args:
            joint_values: Current joint configuration

        Returns:
            6×n Jacobian matrix
        """
        transforms = self.kin.get_all_transforms(joint_values)

        # End-effector position
        p_ee = transforms[-1][:3, 3]

        J = np.zeros((6, self.n_joints))

        # Base frame z-axis and origin
        z_prev = np.array([0, 0, 1])
        p_prev = np.array([0, 0, 0])

        for i in range(self.n_joints):
            param = self.kin.dh_params[i]

            if param.joint_type == 'revolute':
                # Linear velocity component: z × (p_ee - p)
                J[:3, i] = np.cross(z_prev, p_ee - p_prev)
                # Angular velocity component: z
                J[3:, i] = z_prev
            else:  # prismatic
                # Linear velocity component: z
                J[:3, i] = z_prev
                # Angular velocity component: 0
                J[3:, i] = 0

            # Update for next iteration
            if i < self.n_joints - 1:
                z_prev = transforms[i][:3, 2]  # z-axis of frame i
                p_prev = transforms[i][:3, 3]  # origin of frame i

        return J

    def numerical_jacobian(self, joint_values: np.ndarray,
                          epsilon: float = 1e-6) -> np.ndarray:
        """
        Compute Jacobian numerically using finite differences.

        Useful for validation and complex kinematic structures.

        Args:
            joint_values: Current joint configuration
            epsilon: Perturbation size for finite differences

        Returns:
            6×n Jacobian matrix
        """
        J = np.zeros((6, self.n_joints))

        T0 = self.kin.forward_kinematics(joint_values)
        p0 = T0[:3, 3]
        R0 = T0[:3, :3]

        for i in range(self.n_joints):
            q_plus = joint_values.copy()
            q_plus[i] += epsilon

            T_plus = self.kin.forward_kinematics(q_plus)
            p_plus = T_plus[:3, 3]
            R_plus = T_plus[:3, :3]

            # Linear velocity approximation
            J[:3, i] = (p_plus - p0) / epsilon

            # Angular velocity approximation using rotation matrix
            dR = (R_plus - R0) / epsilon
            # Skew-symmetric extraction: ω = [R32-R23, R13-R31, R21-R12] / 2
            omega_skew = dR @ R0.T
            J[3, i] = omega_skew[2, 1]
            J[4, i] = omega_skew[0, 2]
            J[5, i] = omega_skew[1, 0]

        return J

    def manipulability(self, joint_values: np.ndarray) -> float:
        """
        Compute Yoshikawa's manipulability measure.

        w = √(det(J·Jᵀ))

        High values indicate the robot is far from singularities.

        Args:
            joint_values: Current joint configuration

        Returns:
            Manipulability measure (scalar)
        """
        J = self.geometric_jacobian(joint_values)
        return np.sqrt(max(0, np.linalg.det(J @ J.T)))

    def condition_number(self, joint_values: np.ndarray) -> float:
        """
        Compute condition number of the Jacobian.

        Low values indicate good conditioning.
        Infinite value indicates singularity.

        Args:
            joint_values: Current joint configuration

        Returns:
            Condition number
        """
        J = self.geometric_jacobian(joint_values)
        return np.linalg.cond(J)

    def singular_values(self, joint_values: np.ndarray) -> np.ndarray:
        """
        Compute singular values of the Jacobian.

        Useful for understanding velocity transmission
        in different directions.

        Args:
            joint_values: Current joint configuration

        Returns:
            Array of singular values (sorted descending)
        """
        J = self.geometric_jacobian(joint_values)
        return np.linalg.svd(J, compute_uv=False)

    def is_singular(self, joint_values: np.ndarray,
                   threshold: float = 1e-3) -> bool:
        """
        Check if configuration is near a singularity.

        Args:
            joint_values: Current joint configuration
            threshold: Minimum acceptable singular value

        Returns:
            True if near singularity
        """
        sv = self.singular_values(joint_values)
        return sv[-1] < threshold
```

## Inverse Kinematics

Inverse kinematics (IK) finds joint configurations that achieve desired end-effector poses. Humanoids present challenges due to redundancy (more joints than task DOF) and multiple limbs.

```
            Inverse Kinematics Approaches

    ┌─────────────────────────────────────────────────────────┐
    │                   Desired End-Effector                  │
    │                        Pose                             │
    └────────────────────────┬────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                              ▼
    ┌─────────────────────┐      ┌─────────────────────────┐
    │    Analytical IK    │      │     Numerical IK        │
    │  ─────────────────  │      │  ─────────────────────  │
    │  • Closed-form      │      │  • Jacobian-based       │
    │  • Fast (O(1))      │      │  • Iterative            │
    │  • Limited to       │      │  • General purpose      │
    │    specific robots  │      │  • Handles redundancy   │
    └─────────┬───────────┘      └───────────┬─────────────┘
              │                              │
              │         ┌────────────────────┼─────────────┐
              │         ▼                    ▼             ▼
              │   ┌───────────┐      ┌────────────┐  ┌──────────┐
              │   │ Jacobian  │      │ Damped LS  │  │ Task     │
              │   │ Transpose │      │ (DLS)      │  │ Priority │
              │   └───────────┘      └────────────┘  └──────────┘
              │
              ▼
    ┌─────────────────────┐
    │   All Solutions     │
    │  (up to 16 for 6R)  │
    └─────────────────────┘
```

### Comprehensive IK Implementation

```python
"""
Inverse Kinematics Module

Implements various IK solvers for humanoid robots including
numerical iterative methods and redundancy resolution.
"""

import numpy as np
from typing import Optional, Callable, Tuple, List
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod


class IKStatus(Enum):
    """Status codes for IK solutions."""
    SUCCESS = "success"
    MAX_ITERATIONS = "max_iterations"
    SINGULARITY = "singularity"
    JOINT_LIMITS = "joint_limits"
    NO_SOLUTION = "no_solution"


@dataclass
class IKResult:
    """Result of inverse kinematics computation."""
    joint_values: np.ndarray
    status: IKStatus
    iterations: int
    final_error: float

    @property
    def success(self) -> bool:
        return self.status == IKStatus.SUCCESS


class IKSolver(ABC):
    """Abstract base class for IK solvers."""

    @abstractmethod
    def solve(self, target_pose: np.ndarray,
              q_init: np.ndarray) -> IKResult:
        """Solve IK for target pose."""
        pass


class JacobianIK(IKSolver):
    """
    Jacobian-based iterative IK solver.

    Supports multiple methods:
    - Jacobian transpose
    - Pseudoinverse
    - Damped least squares (Levenberg-Marquardt)
    """

    def __init__(self, kinematics: DHKinematics,
                 jacobian_computer: JacobianComputer,
                 method: str = 'damped_ls',
                 damping: float = 0.01,
                 position_tolerance: float = 1e-4,
                 orientation_tolerance: float = 1e-3,
                 max_iterations: int = 100,
                 step_size: float = 0.5,
                 joint_limits: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """
        Initialize Jacobian IK solver.

        Args:
            kinematics: Forward kinematics object
            jacobian_computer: Jacobian computation object
            method: 'transpose', 'pseudoinverse', or 'damped_ls'
            damping: Damping factor for damped LS method
            position_tolerance: Convergence tolerance for position
            orientation_tolerance: Convergence tolerance for orientation
            max_iterations: Maximum iterations
            step_size: Step size multiplier
            joint_limits: Tuple of (lower, upper) joint limits
        """
        self.kin = kinematics
        self.jac = jacobian_computer
        self.method = method
        self.damping = damping
        self.pos_tol = position_tolerance
        self.ori_tol = orientation_tolerance
        self.max_iter = max_iterations
        self.step_size = step_size
        self.joint_limits = joint_limits

    def pose_error(self, target: np.ndarray,
                   current: np.ndarray) -> np.ndarray:
        """
        Compute 6D pose error (position + orientation).

        Uses axis-angle for orientation error.

        Args:
            target: Target 4x4 transformation
            current: Current 4x4 transformation

        Returns:
            6D error vector [position_error; orientation_error]
        """
        # Position error
        pos_error = target[:3, 3] - current[:3, 3]

        # Orientation error using axis-angle
        R_error = target[:3, :3] @ current[:3, :3].T

        # Extract axis-angle from rotation matrix
        angle = np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1, 1))

        if angle < 1e-6:
            ori_error = np.zeros(3)
        else:
            axis = np.array([
                R_error[2, 1] - R_error[1, 2],
                R_error[0, 2] - R_error[2, 0],
                R_error[1, 0] - R_error[0, 1]
            ]) / (2 * np.sin(angle))
            ori_error = angle * axis

        return np.concatenate([pos_error, ori_error])

    def _compute_update(self, J: np.ndarray,
                        error: np.ndarray) -> np.ndarray:
        """Compute joint update based on method."""
        if self.method == 'transpose':
            return self.step_size * J.T @ error

        elif self.method == 'pseudoinverse':
            J_pinv = np.linalg.pinv(J)
            return self.step_size * J_pinv @ error

        else:  # damped_ls
            n = J.shape[1]
            JJT = J @ J.T
            damped = JJT + self.damping**2 * np.eye(JJT.shape[0])
            J_dls = J.T @ np.linalg.inv(damped)
            return self.step_size * J_dls @ error

    def _clamp_to_limits(self, q: np.ndarray) -> np.ndarray:
        """Clamp joint values to limits."""
        if self.joint_limits is not None:
            lower, upper = self.joint_limits
            return np.clip(q, lower, upper)
        return q

    def solve(self, target_pose: np.ndarray,
              q_init: np.ndarray) -> IKResult:
        """
        Solve inverse kinematics.

        Args:
            target_pose: 4x4 target transformation matrix
            q_init: Initial joint configuration guess

        Returns:
            IKResult with solution and status
        """
        q = q_init.copy()

        for iteration in range(self.max_iter):
            # Forward kinematics
            current_pose = self.kin.forward_kinematics(q)

            # Compute error
            error = self.pose_error(target_pose, current_pose)
            pos_error_norm = np.linalg.norm(error[:3])
            ori_error_norm = np.linalg.norm(error[3:])

            # Check convergence
            if pos_error_norm < self.pos_tol and ori_error_norm < self.ori_tol:
                return IKResult(
                    joint_values=q,
                    status=IKStatus.SUCCESS,
                    iterations=iteration + 1,
                    final_error=np.linalg.norm(error)
                )

            # Compute Jacobian
            J = self.jac.geometric_jacobian(q)

            # Check for singularity
            if self.jac.is_singular(q):
                return IKResult(
                    joint_values=q,
                    status=IKStatus.SINGULARITY,
                    iterations=iteration + 1,
                    final_error=np.linalg.norm(error)
                )

            # Compute and apply update
            dq = self._compute_update(J, error)
            q = self._clamp_to_limits(q + dq)

        return IKResult(
            joint_values=q,
            status=IKStatus.MAX_ITERATIONS,
            iterations=self.max_iter,
            final_error=np.linalg.norm(error)
        )


class RedundancyResolutionIK(IKSolver):
    """
    IK solver with null-space optimization for redundant robots.

    Minimizes a secondary objective while achieving the primary
    end-effector task.
    """

    def __init__(self, kinematics: DHKinematics,
                 jacobian_computer: JacobianComputer,
                 secondary_objective: Optional[Callable] = None,
                 **kwargs):
        """
        Initialize redundancy resolution IK.

        Args:
            kinematics: Forward kinematics object
            jacobian_computer: Jacobian computation object
            secondary_objective: Function that returns gradient of
                               secondary objective to minimize
            **kwargs: Additional arguments for base solver
        """
        self.kin = kinematics
        self.jac = jacobian_computer
        self.base_solver = JacobianIK(kinematics, jacobian_computer, **kwargs)

        if secondary_objective is None:
            self.secondary_obj = self._default_secondary_objective
        else:
            self.secondary_obj = secondary_objective

    def _default_secondary_objective(self, q: np.ndarray) -> np.ndarray:
        """
        Default secondary objective: stay away from joint limits.

        Gradient points toward joint midpoints.
        """
        if self.base_solver.joint_limits is None:
            return np.zeros_like(q)

        lower, upper = self.base_solver.joint_limits
        mid = (lower + upper) / 2
        return -0.1 * (q - mid)  # Gradient toward midpoint

    def solve(self, target_pose: np.ndarray,
              q_init: np.ndarray) -> IKResult:
        """
        Solve IK with null-space optimization.

        Uses: q̇ = J⁺·ẋ + (I - J⁺·J)·q̇₀

        where q̇₀ is the secondary objective gradient.
        """
        q = q_init.copy()

        for iteration in range(self.base_solver.max_iter):
            current_pose = self.kin.forward_kinematics(q)
            error = self.base_solver.pose_error(target_pose, current_pose)

            pos_error = np.linalg.norm(error[:3])
            ori_error = np.linalg.norm(error[3:])

            if pos_error < self.base_solver.pos_tol and \
               ori_error < self.base_solver.ori_tol:
                return IKResult(
                    joint_values=q,
                    status=IKStatus.SUCCESS,
                    iterations=iteration + 1,
                    final_error=np.linalg.norm(error)
                )

            J = self.jac.geometric_jacobian(q)

            # Damped pseudoinverse
            damping = self.base_solver.damping
            JJT = J @ J.T
            J_pinv = J.T @ np.linalg.inv(JJT + damping**2 * np.eye(6))

            # Primary task
            dq_primary = J_pinv @ error

            # Null-space projection
            null_proj = np.eye(len(q)) - J_pinv @ J
            dq_secondary = self.secondary_obj(q)
            dq_null = null_proj @ dq_secondary

            # Combined update
            dq = self.base_solver.step_size * (dq_primary + dq_null)
            q = self.base_solver._clamp_to_limits(q + dq)

        return IKResult(
            joint_values=q,
            status=IKStatus.MAX_ITERATIONS,
            iterations=self.base_solver.max_iter,
            final_error=np.linalg.norm(error)
        )


class TaskPriorityIK(IKSolver):
    """
    Task-priority IK for multiple simultaneous objectives.

    Higher priority tasks are satisfied first, with lower
    priority tasks executed in the remaining null space.
    """

    def __init__(self, kinematics: DHKinematics,
                 jacobian_computer: JacobianComputer,
                 max_iterations: int = 100,
                 tolerance: float = 1e-4,
                 damping: float = 0.01):
        """
        Initialize task-priority IK solver.

        Args:
            kinematics: Forward kinematics object
            jacobian_computer: Jacobian computation object
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            damping: Damping factor
        """
        self.kin = kinematics
        self.jac = jacobian_computer
        self.max_iter = max_iterations
        self.tol = tolerance
        self.damping = damping

    def solve_multi_task(self, tasks: List[Tuple[np.ndarray, np.ndarray]],
                         q_init: np.ndarray) -> IKResult:
        """
        Solve IK for multiple prioritized tasks.

        Args:
            tasks: List of (Jacobian, error) tuples in priority order
            q_init: Initial joint configuration

        Returns:
            IKResult with solution
        """
        q = q_init.copy()
        n = len(q)

        for iteration in range(self.max_iter):
            dq = np.zeros(n)
            null_proj = np.eye(n)

            total_error = 0

            for J, compute_error in tasks:
                error = compute_error(q)
                total_error += np.linalg.norm(error)

                # Project Jacobian onto remaining null space
                J_proj = J @ null_proj

                # Damped pseudoinverse of projected Jacobian
                JJT = J_proj @ J_proj.T
                J_pinv = J_proj.T @ np.linalg.inv(
                    JJT + self.damping**2 * np.eye(JJT.shape[0])
                )

                # Accumulate joint update
                dq += J_pinv @ error

                # Update null space projector
                null_proj = null_proj @ (np.eye(n) - J_pinv @ J_proj)

            # Check convergence
            if total_error < self.tol:
                return IKResult(
                    joint_values=q,
                    status=IKStatus.SUCCESS,
                    iterations=iteration + 1,
                    final_error=total_error
                )

            q = q + 0.5 * dq

        return IKResult(
            joint_values=q,
            status=IKStatus.MAX_ITERATIONS,
            iterations=self.max_iter,
            final_error=total_error
        )

    def solve(self, target_pose: np.ndarray,
              q_init: np.ndarray) -> IKResult:
        """Single task solve (wrapper for compatibility)."""
        J_func = lambda q: self.jac.geometric_jacobian(q)

        base_solver = JacobianIK(self.kin, self.jac, damping=self.damping)

        def error_func(q):
            current = self.kin.forward_kinematics(q)
            return base_solver.pose_error(target_pose, current)

        return self.solve_multi_task(
            [(J_func(q_init), error_func)],
            q_init
        )
```

## Center of Mass Computation

For bipedal balance, accurate Center of Mass (CoM) computation is essential.

```
         Center of Mass for Balance

                  CoM
                   ●────────────────► CoM velocity
                  /│\
                 / │ \
    ┌───────────/──│──\───────────┐
    │          /   │   \          │   Support Polygon
    │         /    │    \         │   (convex hull of
    │        /     │     \        │    foot contacts)
    │       ▼      ▼      ▼       │
    │      ○      ○      ○        │
    └─────────────────────────────┘

    Static balance:  CoM projection inside support polygon
    Dynamic balance: ZMP (Zero Moment Point) inside support polygon


              CoM Computation

         CoM = Σ(mᵢ · pᵢ) / Σmᵢ

    where:
        mᵢ = mass of link i
        pᵢ = position of link i's CoM in world frame
```

### Center of Mass Implementation

```python
"""
Center of Mass Computation Module

Implements whole-body CoM computation and related dynamics
for humanoid balance analysis.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class LinkProperties:
    """Physical properties of a robot link."""
    name: str
    mass: float                    # kg
    local_com: np.ndarray         # CoM in link frame [m]
    inertia: np.ndarray           # 3x3 inertia tensor [kg·m²]
    parent_joint_idx: int         # Index of parent joint

    def __post_init__(self):
        self.local_com = np.asarray(self.local_com)
        self.inertia = np.asarray(self.inertia)


class HumanoidCoMComputer:
    """
    Computes whole-body center of mass and related quantities.

    Essential for balance control in humanoid robots.
    """

    def __init__(self, kinematics: DHKinematics,
                 link_properties: List[LinkProperties]):
        """
        Initialize CoM computer.

        Args:
            kinematics: Forward kinematics object
            link_properties: Physical properties for each link
        """
        self.kin = kinematics
        self.links = link_properties
        self.total_mass = sum(link.mass for link in link_properties)

    def compute_com(self, joint_values: np.ndarray) -> np.ndarray:
        """
        Compute whole-body center of mass position.

        Args:
            joint_values: Current joint configuration

        Returns:
            3D CoM position in world frame
        """
        transforms = self.kin.get_all_transforms(joint_values)

        com = np.zeros(3)

        for link in self.links:
            if link.parent_joint_idx >= 0:
                T = transforms[link.parent_joint_idx]
            else:
                T = np.eye(4)

            # Transform local CoM to world frame
            link_com_world = T[:3, :3] @ link.local_com + T[:3, 3]
            com += link.mass * link_com_world

        return com / self.total_mass

    def compute_com_jacobian(self, joint_values: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian of center of mass.

        Maps joint velocities to CoM velocity.

        Args:
            joint_values: Current joint configuration

        Returns:
            3×n CoM Jacobian matrix
        """
        n = len(joint_values)
        J_com = np.zeros((3, n))

        transforms = self.kin.get_all_transforms(joint_values)

        for link in self.links:
            # Get link transform
            if link.parent_joint_idx >= 0:
                T = transforms[link.parent_joint_idx]
            else:
                T = np.eye(4)

            # Link CoM in world frame
            p_com = T[:3, :3] @ link.local_com + T[:3, 3]

            # Compute contribution from each joint
            z_prev = np.array([0, 0, 1])
            p_prev = np.array([0, 0, 0])

            for j in range(min(link.parent_joint_idx + 1, n)):
                param = self.kin.dh_params[j]

                if param.joint_type == 'revolute':
                    # Contribution: m * (z × (p_com - p_joint)) / M_total
                    J_com[:, j] += link.mass * np.cross(z_prev, p_com - p_prev)
                else:
                    # Prismatic: m * z / M_total
                    J_com[:, j] += link.mass * z_prev

                # Update for next joint
                if j < len(transforms):
                    z_prev = transforms[j][:3, 2]
                    p_prev = transforms[j][:3, 3]

        return J_com / self.total_mass

    def compute_zmp(self, joint_values: np.ndarray,
                    joint_velocities: np.ndarray,
                    joint_accelerations: np.ndarray,
                    gravity: float = 9.81) -> np.ndarray:
        """
        Compute Zero Moment Point (ZMP).

        ZMP is where the horizontal moment is zero. For stable
        walking, ZMP must stay within the support polygon.

        Args:
            joint_values: Joint positions
            joint_velocities: Joint velocities
            joint_accelerations: Joint accelerations
            gravity: Gravitational acceleration

        Returns:
            2D ZMP position (x, y) on ground plane
        """
        # Compute CoM and its derivatives
        com = self.compute_com(joint_values)

        # Numerical differentiation for CoM velocity and acceleration
        epsilon = 1e-6
        q_plus = joint_values + epsilon * joint_velocities
        com_plus = self.compute_com(q_plus)
        com_vel = (com_plus - com) / epsilon

        q_plus2 = q_plus + epsilon * joint_velocities
        com_plus2 = self.compute_com(q_plus2)
        com_vel_plus = (com_plus2 - com_plus) / epsilon
        com_acc = (com_vel_plus - com_vel) / epsilon

        # Add effect of joint accelerations (simplified)
        J_com = self.compute_com_jacobian(joint_values)
        com_acc += J_com @ joint_accelerations

        # ZMP formula:
        # x_zmp = x_com - z_com * (x_ddot_com / (z_ddot_com + g))
        # y_zmp = y_com - z_com * (y_ddot_com / (z_ddot_com + g))

        denom = com_acc[2] + gravity
        if abs(denom) < 1e-6:
            denom = 1e-6  # Avoid division by zero

        zmp = np.array([
            com[0] - com[2] * com_acc[0] / denom,
            com[1] - com[2] * com_acc[1] / denom
        ])

        return zmp

    def compute_angular_momentum(self, joint_values: np.ndarray,
                                  joint_velocities: np.ndarray) -> np.ndarray:
        """
        Compute whole-body angular momentum about CoM.

        Important for dynamic balance and motion planning.

        Args:
            joint_values: Joint positions
            joint_velocities: Joint velocities

        Returns:
            3D angular momentum vector
        """
        transforms = self.kin.get_all_transforms(joint_values)
        com = self.compute_com(joint_values)

        L = np.zeros(3)  # Total angular momentum

        for link in self.links:
            if link.parent_joint_idx >= 0:
                T = transforms[link.parent_joint_idx]
            else:
                T = np.eye(4)

            R = T[:3, :3]
            p = T[:3, 3]

            # Link CoM position relative to total CoM
            p_link_com = R @ link.local_com + p - com

            # Get link velocity (simplified - assumes all joints affect link)
            # In practice, need proper Jacobian for each link
            link_vel = np.zeros(3)
            link_omega = np.zeros(3)

            z_prev = np.array([0, 0, 1])
            p_prev = np.array([0, 0, 0])

            for j in range(min(link.parent_joint_idx + 1, len(joint_values))):
                param = self.kin.dh_params[j]

                if param.joint_type == 'revolute':
                    link_vel += np.cross(z_prev, p_link_com - (p_prev - com)) * \
                                joint_velocities[j]
                    link_omega += z_prev * joint_velocities[j]

                if j < len(transforms):
                    z_prev = transforms[j][:3, 2]
                    p_prev = transforms[j][:3, 3]

            # Linear momentum contribution: m * (r × v)
            L += link.mass * np.cross(p_link_com, link_vel)

            # Rotational contribution: I * ω (in world frame)
            I_world = R @ link.inertia @ R.T
            L += I_world @ link_omega

        return L


def create_humanoid_links() -> List[LinkProperties]:
    """
    Create link properties for a standard humanoid.

    Based on typical anthropometric data scaled to robot size.

    Returns:
        List of link properties
    """
    return [
        # Torso
        LinkProperties(
            name="torso",
            mass=15.0,
            local_com=np.array([0, 0, 0.15]),
            inertia=np.diag([0.5, 0.4, 0.3]),
            parent_joint_idx=2  # After waist joints
        ),
        # Head
        LinkProperties(
            name="head",
            mass=3.0,
            local_com=np.array([0, 0, 0.05]),
            inertia=np.diag([0.02, 0.02, 0.01]),
            parent_joint_idx=4  # After neck joints
        ),
        # Left upper arm
        LinkProperties(
            name="left_upper_arm",
            mass=2.0,
            local_com=np.array([0, 0, -0.15]),
            inertia=np.diag([0.01, 0.01, 0.002]),
            parent_joint_idx=7  # After shoulder joints
        ),
        # Left forearm
        LinkProperties(
            name="left_forearm",
            mass=1.5,
            local_com=np.array([0, 0, -0.12]),
            inertia=np.diag([0.005, 0.005, 0.001]),
            parent_joint_idx=8  # After elbow
        ),
        # Right upper arm
        LinkProperties(
            name="right_upper_arm",
            mass=2.0,
            local_com=np.array([0, 0, -0.15]),
            inertia=np.diag([0.01, 0.01, 0.002]),
            parent_joint_idx=14
        ),
        # Right forearm
        LinkProperties(
            name="right_forearm",
            mass=1.5,
            local_com=np.array([0, 0, -0.12]),
            inertia=np.diag([0.005, 0.005, 0.001]),
            parent_joint_idx=15
        ),
        # Left thigh
        LinkProperties(
            name="left_thigh",
            mass=5.0,
            local_com=np.array([0, 0, -0.2]),
            inertia=np.diag([0.08, 0.08, 0.02]),
            parent_joint_idx=23  # After hip joints
        ),
        # Left shank
        LinkProperties(
            name="left_shank",
            mass=3.0,
            local_com=np.array([0, 0, -0.2]),
            inertia=np.diag([0.04, 0.04, 0.01]),
            parent_joint_idx=24  # After knee
        ),
        # Right thigh
        LinkProperties(
            name="right_thigh",
            mass=5.0,
            local_com=np.array([0, 0, -0.2]),
            inertia=np.diag([0.08, 0.08, 0.02]),
            parent_joint_idx=29
        ),
        # Right shank
        LinkProperties(
            name="right_shank",
            mass=3.0,
            local_com=np.array([0, 0, -0.2]),
            inertia=np.diag([0.04, 0.04, 0.01]),
            parent_joint_idx=30
        )
    ]
```

## Whole-Body Kinematics

For coordinated humanoid motion, we need to consider the robot as a single kinematic system with multiple task frames.

```python
"""
Whole-Body Kinematics Module

Handles multi-chain kinematics for humanoid robots,
coordinating multiple limbs relative to different base frames.
"""

import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class TaskFrame:
    """Definition of a task-space frame."""
    name: str
    chain_name: str           # Which kinematic chain
    position_dof: List[int]   # Which position DOF to control [x, y, z]
    orientation_dof: List[int]  # Which orientation DOF [rx, ry, rz]
    priority: int = 0         # Higher = more important


class WholeBodyKinematics:
    """
    Manages whole-body kinematics for humanoid robots.

    Coordinates multiple kinematic chains sharing a common
    floating base, handling the tree structure of humanoid
    kinematics.
    """

    def __init__(self, chains: Dict[str, DHKinematics],
                 base_to_chain: Dict[str, np.ndarray]):
        """
        Initialize whole-body kinematics.

        Args:
            chains: Dictionary of kinematic chains by name
            base_to_chain: Fixed transforms from floating base
                          to each chain's root
        """
        self.chains = chains
        self.base_to_chain = base_to_chain

        # Count total DOF
        self.n_floating = 6  # Floating base: x, y, z, roll, pitch, yaw
        self.chain_dof = {name: chain.n_joints
                         for name, chain in chains.items()}
        self.total_dof = self.n_floating + sum(self.chain_dof.values())

        # Build joint index mapping
        self._build_joint_mapping()

    def _build_joint_mapping(self):
        """Create mapping from chain joints to whole-body indices."""
        self.joint_indices = {}
        idx = self.n_floating

        for name in self.chains:
            n_joints = self.chain_dof[name]
            self.joint_indices[name] = list(range(idx, idx + n_joints))
            idx += n_joints

    def floating_base_transform(self,
                                base_state: np.ndarray) -> np.ndarray:
        """
        Compute floating base transformation.

        Args:
            base_state: [x, y, z, roll, pitch, yaw]

        Returns:
            4x4 transformation from world to floating base
        """
        x, y, z, roll, pitch, yaw = base_state

        # Rotation from RPY (ZYX Euler convention)
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr           ]
        ])

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]

        return T

    def forward_kinematics(self, full_state: np.ndarray,
                          chain_name: str) -> np.ndarray:
        """
        Compute end-effector pose for a specific chain.

        Args:
            full_state: Full robot state [base_state, chain_joints...]
            chain_name: Name of the kinematic chain

        Returns:
            4x4 transformation of chain end-effector in world frame
        """
        base_state = full_state[:6]
        chain_joints = full_state[self.joint_indices[chain_name]]

        # World to floating base
        T_world_base = self.floating_base_transform(base_state)

        # Floating base to chain root
        T_base_chain = self.base_to_chain[chain_name]

        # Chain root to end-effector
        T_chain_ee = self.chains[chain_name].forward_kinematics(chain_joints)

        return T_world_base @ T_base_chain @ T_chain_ee

    def whole_body_jacobian(self, full_state: np.ndarray,
                           chain_name: str) -> np.ndarray:
        """
        Compute whole-body Jacobian for a chain's end-effector.

        Includes contributions from floating base motion.

        Args:
            full_state: Full robot state
            chain_name: Name of the kinematic chain

        Returns:
            6 × total_dof Jacobian matrix
        """
        J = np.zeros((6, self.total_dof))

        base_state = full_state[:6]
        chain_joints = full_state[self.joint_indices[chain_name]]

        # Get end-effector position in world frame
        T_ee = self.forward_kinematics(full_state, chain_name)
        p_ee = T_ee[:3, 3]

        # Floating base Jacobian
        T_base = self.floating_base_transform(base_state)
        p_base = T_base[:3, 3]

        # Translation DOF
        J[:3, :3] = np.eye(3)

        # Rotation DOF (using skew-symmetric form of r × ω)
        r = p_ee - p_base
        J[:3, 3:6] = -self._skew(r)
        J[3:6, 3:6] = np.eye(3)

        # Chain Jacobian (transformed to world frame)
        chain = self.chains[chain_name]
        jac_computer = JacobianComputer(chain)
        J_chain_local = jac_computer.geometric_jacobian(chain_joints)

        # Transform chain Jacobian to world frame
        R_world_chain = (T_base @ self.base_to_chain[chain_name])[:3, :3]
        J_chain_world = np.zeros_like(J_chain_local)
        J_chain_world[:3, :] = R_world_chain @ J_chain_local[:3, :]
        J_chain_world[3:, :] = R_world_chain @ J_chain_local[3:, :]

        # Insert chain Jacobian at correct indices
        chain_idx = self.joint_indices[chain_name]
        J[:, chain_idx] = J_chain_world

        return J

    @staticmethod
    def _skew(v: np.ndarray) -> np.ndarray:
        """Create skew-symmetric matrix from vector."""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    def multi_task_ik(self, tasks: List[Tuple[str, np.ndarray]],
                      current_state: np.ndarray,
                      priorities: Optional[List[int]] = None,
                      max_iterations: int = 100,
                      tolerance: float = 1e-4) -> np.ndarray:
        """
        Solve IK for multiple simultaneous tasks.

        Args:
            tasks: List of (chain_name, target_pose) tuples
            current_state: Current full state
            priorities: Priority for each task (higher = more important)
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance

        Returns:
            Solution state
        """
        if priorities is None:
            priorities = list(range(len(tasks)))

        # Sort tasks by priority (descending)
        sorted_tasks = sorted(zip(priorities, tasks),
                             key=lambda x: -x[0])

        state = current_state.copy()

        for _ in range(max_iterations):
            total_error = 0
            null_proj = np.eye(self.total_dof)
            dq = np.zeros(self.total_dof)

            for _, (chain_name, target) in sorted_tasks:
                # Current pose and error
                current = self.forward_kinematics(state, chain_name)
                error = self._pose_error(target, current)
                total_error += np.linalg.norm(error)

                # Jacobian projected onto null space
                J = self.whole_body_jacobian(state, chain_name)
                J_proj = J @ null_proj

                # Damped pseudoinverse
                damping = 0.01
                JJT = J_proj @ J_proj.T
                J_pinv = J_proj.T @ np.linalg.inv(
                    JJT + damping**2 * np.eye(6)
                )

                # Accumulate update
                dq += J_pinv @ error

                # Update null space
                null_proj = null_proj @ (np.eye(self.total_dof) - J_pinv @ J_proj)

            if total_error < tolerance * len(tasks):
                break

            state = state + 0.3 * dq

        return state

    @staticmethod
    def _pose_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
        """Compute 6D pose error."""
        pos_error = target[:3, 3] - current[:3, 3]

        R_error = target[:3, :3] @ current[:3, :3].T
        angle = np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1, 1))

        if angle < 1e-6:
            ori_error = np.zeros(3)
        else:
            axis = np.array([
                R_error[2, 1] - R_error[1, 2],
                R_error[0, 2] - R_error[2, 0],
                R_error[1, 0] - R_error[0, 1]
            ]) / (2 * np.sin(angle))
            ori_error = angle * axis

        return np.concatenate([pos_error, ori_error])
```

## Industry Perspective: Humanoid Platforms

Modern humanoid robots showcase different kinematic design philosophies:

| Platform | DOF | Notable Features |
|----------|-----|------------------|
| **Boston Dynamics Atlas** | 28 | Hydraulic, highly dynamic, focused on whole-body motion |
| **Tesla Optimus** | 28+ | Electric actuators, designed for industrial tasks |
| **Figure 01/02** | 40+ | Dexterous hands (16 DOF each), VLA integration |
| **Unitree H1** | 19 | Cost-effective, educational platform |
| **Agility Digit** | 16 | Simplified legs, focus on logistics |
| **1X NEO** | 30+ | Soft actuators, human-safe interaction |

The trend is toward **higher DOF hands** (12-20 DOF per hand) for manipulation dexterity, while **leg designs remain relatively standard** (6 DOF per leg) since locomotion requirements are well understood.

**Key challenges in modern humanoid kinematics:**
1. **Real-time performance**: Computing IK for 30+ DOF at 1kHz
2. **Singularity handling**: Robust operation near kinematic limits
3. **Multi-contact scenarios**: Both feet plus both hands in contact
4. **Self-collision avoidance**: Many DOF means many potential collisions

## Summary

```
┌────────────────────────────────────────────────────────────────────┐
│                    Humanoid Kinematics Recap                       │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Kinematic Structure                                               │
│  ───────────────────                                               │
│  • Tree topology with multiple branches                            │
│  • 30-50+ DOF typical for full humanoids                          │
│  • Floating base adds 6 DOF for world positioning                 │
│                                                                    │
│  Forward Kinematics                                                │
│  ──────────────────                                                │
│  • DH convention: systematic frame assignment                      │
│  • Chain composition for multi-limb robots                        │
│  • Per-link transforms enable CoM computation                     │
│                                                                    │
│  Inverse Kinematics                                                │
│  ──────────────────                                                │
│  • Damped least squares for robust convergence                    │
│  • Null-space optimization for redundant systems                  │
│  • Task-priority for multiple simultaneous goals                  │
│                                                                    │
│  Jacobian Analysis                                                 │
│  ─────────────────                                                 │
│  • Maps joint velocities to task-space velocities                 │
│  • Manipulability quantifies distance from singularity            │
│  • CoM Jacobian for balance control                               │
│                                                                    │
│  Balance Metrics                                                   │
│  ───────────────                                                   │
│  • Center of Mass (CoM) for static balance                        │
│  • Zero Moment Point (ZMP) for dynamic balance                    │
│  • Angular momentum for predictive control                        │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### Key Equations Reference

| Concept | Equation |
|---------|----------|
| DH Transform | T = Rz(θ) · Tz(d) · Tx(a) · Rx(α) |
| Jacobian | J = [Jv; Jω], v = Jv·q̇, ω = Jω·q̇ |
| Damped LS | q̇ = Jᵀ(JJᵀ + λ²I)⁻¹ẋ |
| Null-space | q̇ = J⁺ẋ + (I - J⁺J)q̇₀ |
| CoM | pcom = Σ(mᵢpᵢ) / Σmᵢ |
| ZMP | xzmp = xcom - zcom·ẍcom/(z̈com + g) |

### Implementation Checklist

- [ ] Model robot using DH parameters for each chain
- [ ] Implement forward kinematics with transform caching
- [ ] Use damped least squares for robust IK
- [ ] Add null-space optimization for redundancy
- [ ] Compute CoM Jacobian for balance control
- [ ] Handle singularities gracefully
- [ ] Validate with known poses before deployment
- [ ] Profile performance for real-time operation

## Further Reading

- Siciliano, B. et al. "Robotics: Modelling, Planning and Control" (2009)
- Lynch, K. & Park, F. "Modern Robotics" (2017) - Free online
- Kajita, S. "Introduction to Humanoid Robotics" (2014)
- Sentis, L. "Synthesis and Control of Whole-Body Behaviors" (PhD Thesis)
- Khatib, O. "A Unified Approach for Motion and Force Control of Robot Manipulators" (1987)
