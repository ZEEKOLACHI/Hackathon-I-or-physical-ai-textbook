---
id: ch-6-17
title: Bipedal Locomotion
sidebar_position: 2
difficulty: advanced
estimated_time: 60
prerequisites: [ch-6-16]
---

# Bipedal Locomotion

> "A journey of a thousand miles begins with a single step."
> — Lao Tzu

Walking on two legs is one of the most challenging problems in robotics. Humans make it look effortless, yet bipedal locomotion requires precise coordination of dozens of joints, continuous balance maintenance, and rapid adaptation to terrain changes. This chapter explores the fundamental principles and algorithms that enable humanoid robots to walk.

## The Challenge of Bipedal Walking

```
                     Why Walking is Hard

    Static Standing          Dynamic Walking
    ───────────────          ───────────────
         ┌───┐                    ┌───┐
         │   │                    │   │  ← CoM outside
         │   │                    │   │    support!
         └─┬─┘                    └─┬─┘
          /│\                      /│\
         / │ \                    / │ \
        /  │  \                  /  │  \
       /   ▼   \                /   │   \
    ━━━●━━━━━━━●━━━          ━━━●━━━│━━━●━━━
      CoM inside               ┌────▼────┐
      support                  │ Falling │
                               └─────────┘

    Walking = Controlled Falling + Recovery

    Key Insight: During walking, the center of mass is
    OUTSIDE the support polygon most of the time!
```

### Gait Cycle Phases

```python
"""
Bipedal Gait Cycle Module

Defines the phases and timing of humanoid walking.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Tuple, Optional
import numpy as np


class GaitPhase(Enum):
    """Phases of the bipedal gait cycle."""
    LEFT_STANCE = auto()      # Left foot on ground, right swinging
    RIGHT_STANCE = auto()     # Right foot on ground, left swinging
    DOUBLE_SUPPORT = auto()   # Both feet on ground (transition)
    FLIGHT = auto()           # Both feet in air (running only)


class FootState(Enum):
    """State of individual foot."""
    STANCE = "stance"         # Foot in contact with ground
    SWING = "swing"           # Foot moving through air
    HEEL_STRIKE = "heel_strike"  # Initial contact
    TOE_OFF = "toe_off"       # Leaving ground


@dataclass
class GaitParameters:
    """
    Parameters defining a walking gait.

    Attributes:
        step_length: Distance between consecutive foot placements [m]
        step_width: Lateral distance between feet [m]
        step_height: Maximum foot lift height during swing [m]
        step_time: Duration of one step [s]
        double_support_ratio: Fraction of gait with both feet down
        com_height: Height of center of mass [m]
    """
    step_length: float = 0.30
    step_width: float = 0.10
    step_height: float = 0.05
    step_time: float = 0.80
    double_support_ratio: float = 0.20
    com_height: float = 0.85

    def __post_init__(self):
        """Validate gait parameters."""
        assert self.step_length > 0
        assert self.step_width > 0
        assert self.step_height > 0
        assert self.step_time > 0
        assert 0 < self.double_support_ratio < 0.5

    @property
    def single_support_time(self) -> float:
        """Duration of single support phase."""
        return self.step_time * (1 - self.double_support_ratio)

    @property
    def double_support_time(self) -> float:
        """Duration of double support phase."""
        return self.step_time * self.double_support_ratio

    @property
    def walking_speed(self) -> float:
        """Average forward walking velocity [m/s]."""
        return self.step_length / self.step_time


@dataclass
class GaitState:
    """Current state in the gait cycle."""
    phase: GaitPhase
    phase_time: float         # Time within current phase
    left_foot: FootState
    right_foot: FootState
    support_foot: str         # 'left', 'right', or 'both'


class GaitClock:
    """
    Manages timing throughout the gait cycle.

    Tracks phase transitions and provides normalized
    phase time for trajectory generation.
    """

    def __init__(self, params: GaitParameters):
        """
        Initialize gait clock.

        Args:
            params: Gait timing parameters
        """
        self.params = params
        self.time = 0.0
        self.cycle_count = 0

    def update(self, dt: float) -> GaitState:
        """
        Advance time and compute current gait state.

        Args:
            dt: Time step [s]

        Returns:
            Current gait state
        """
        self.time += dt

        # Compute position within gait cycle
        cycle_time = 2 * self.params.step_time  # Full left-right cycle
        cycle_phase = (self.time % cycle_time) / cycle_time

        # Determine current phase
        ds_ratio = self.params.double_support_ratio / 2

        if cycle_phase < ds_ratio:
            # First double support (left to right transition)
            phase = GaitPhase.DOUBLE_SUPPORT
            phase_time = cycle_phase / ds_ratio
            support_foot = 'both'
            left_foot = FootState.STANCE
            right_foot = FootState.HEEL_STRIKE

        elif cycle_phase < 0.5 - ds_ratio:
            # Right stance, left swing
            phase = GaitPhase.RIGHT_STANCE
            phase_time = (cycle_phase - ds_ratio) / (0.5 - 2 * ds_ratio)
            support_foot = 'right'
            left_foot = FootState.SWING
            right_foot = FootState.STANCE

        elif cycle_phase < 0.5 + ds_ratio:
            # Second double support (right to left transition)
            phase = GaitPhase.DOUBLE_SUPPORT
            phase_time = (cycle_phase - 0.5 + ds_ratio) / (2 * ds_ratio)
            support_foot = 'both'
            left_foot = FootState.HEEL_STRIKE
            right_foot = FootState.STANCE

        else:
            # Left stance, right swing
            phase = GaitPhase.LEFT_STANCE
            phase_time = (cycle_phase - 0.5 - ds_ratio) / (0.5 - 2 * ds_ratio)
            support_foot = 'left'
            left_foot = FootState.STANCE
            right_foot = FootState.SWING

        return GaitState(
            phase=phase,
            phase_time=phase_time,
            left_foot=left_foot,
            right_foot=right_foot,
            support_foot=support_foot
        )

    def reset(self):
        """Reset clock to beginning of cycle."""
        self.time = 0.0
        self.cycle_count = 0
```

## Zero Moment Point (ZMP)

The **Zero Moment Point** is where the horizontal component of the moment of ground reaction forces equals zero. For stable walking, ZMP must remain within the support polygon.

```
                    ZMP Concept

          Robot State                Ground Reaction
          ───────────                ───────────────
              ┌───┐
              │   │ ← CoM (mass m)
              │   │
              └─┬─┘                        ↑ Fz
               /│\                         │
              / │ \                        │
             /  │  \                  ─────┴───── Ground
            /   │   \                 ▲         ▲
        ━━━━━━━━━━━━━━━━━             │         │
            ▲       ▲                 │ τy      │
            │       │                 └────●────┘
            Fl      Fr                    ZMP

    ZMP Definition:
    ───────────────
    At the ZMP, the net horizontal moment is zero:

    Σ τhorizontal = 0

    ZMP_x = (Σ mᵢ(gzᵢ + z̈ᵢ)xᵢ - Σ mᵢẍᵢzᵢ) / (Σ mᵢ(g + z̈ᵢ))

    Stability Criterion:
    ────────────────────
    ZMP ∈ Support Polygon → Robot won't tip over
```

### ZMP Implementation

```python
"""
Zero Moment Point Computation Module

Implements ZMP calculation and stability analysis
for bipedal robots.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ContactPoint:
    """Contact point on the ground."""
    position: np.ndarray  # [x, y, z] position
    normal: np.ndarray    # Surface normal (usually [0, 0, 1])
    friction: float       # Friction coefficient


class SupportPolygon:
    """
    Represents the convex hull of foot contact points.

    The support polygon is the region where ZMP must
    stay for stable balance.
    """

    def __init__(self, contact_points: List[np.ndarray]):
        """
        Initialize support polygon.

        Args:
            contact_points: List of 2D or 3D contact positions
        """
        # Project to ground plane (x, y)
        self.points = np.array([p[:2] for p in contact_points])
        self._compute_convex_hull()

    def _compute_convex_hull(self):
        """Compute convex hull of contact points."""
        from scipy.spatial import ConvexHull

        if len(self.points) < 3:
            # Line or point - degenerate case
            self.hull = None
            self.vertices = self.points
        else:
            try:
                self.hull = ConvexHull(self.points)
                self.vertices = self.points[self.hull.vertices]
            except:
                self.hull = None
                self.vertices = self.points

    def contains(self, point: np.ndarray, margin: float = 0.0) -> bool:
        """
        Check if point is inside support polygon.

        Args:
            point: 2D point to check [x, y]
            margin: Safety margin (shrink polygon by this amount)

        Returns:
            True if point is inside (with margin)
        """
        if len(self.vertices) < 3:
            return False

        # Use winding number algorithm
        n = len(self.vertices)
        winding = 0

        for i in range(n):
            v1 = self.vertices[i] - point[:2]
            v2 = self.vertices[(i + 1) % n] - point[:2]

            if v1[1] <= 0:
                if v2[1] > 0:
                    if np.cross(v1, v2) > margin:
                        winding += 1
            else:
                if v2[1] <= 0:
                    if np.cross(v1, v2) < -margin:
                        winding -= 1

        return winding != 0

    def distance_to_edge(self, point: np.ndarray) -> float:
        """
        Compute signed distance to nearest edge.

        Positive = inside, negative = outside.

        Args:
            point: 2D point to check

        Returns:
            Signed distance to nearest edge
        """
        min_dist = float('inf')
        p = point[:2]

        n = len(self.vertices)
        for i in range(n):
            v1 = self.vertices[i]
            v2 = self.vertices[(i + 1) % n]

            # Edge vector and normal
            edge = v2 - v1
            edge_length = np.linalg.norm(edge)
            edge_dir = edge / edge_length
            normal = np.array([-edge_dir[1], edge_dir[0]])

            # Project point onto edge
            t = np.dot(p - v1, edge_dir)
            t = np.clip(t, 0, edge_length)

            closest = v1 + t * edge_dir
            dist = np.linalg.norm(p - closest)

            # Sign based on which side of edge
            sign = np.sign(np.dot(p - v1, normal))
            signed_dist = sign * dist

            if abs(dist) < abs(min_dist):
                min_dist = signed_dist

        return min_dist

    def get_center(self) -> np.ndarray:
        """Get centroid of support polygon."""
        return np.mean(self.vertices, axis=0)


class ZMPComputer:
    """
    Computes Zero Moment Point for bipedal balance.

    Handles both static and dynamic ZMP calculation.
    """

    def __init__(self, gravity: float = 9.81):
        """
        Initialize ZMP computer.

        Args:
            gravity: Gravitational acceleration [m/s²]
        """
        self.gravity = gravity

    def compute_zmp(self,
                    com_position: np.ndarray,
                    com_acceleration: np.ndarray,
                    ground_height: float = 0.0) -> np.ndarray:
        """
        Compute ZMP from CoM state.

        Uses the simplified cart-table model:
        ZMP_x = CoM_x - (CoM_z - ground) * CoM_accel_x / (g + CoM_accel_z)

        Args:
            com_position: 3D CoM position [x, y, z]
            com_acceleration: 3D CoM acceleration [ax, ay, az]
            ground_height: Height of ground plane

        Returns:
            2D ZMP position [x, y]
        """
        z_rel = com_position[2] - ground_height

        # Denominator: g + vertical acceleration
        denom = self.gravity + com_acceleration[2]
        if abs(denom) < 1e-6:
            denom = 1e-6  # Avoid division by zero

        zmp_x = com_position[0] - z_rel * com_acceleration[0] / denom
        zmp_y = com_position[1] - z_rel * com_acceleration[1] / denom

        return np.array([zmp_x, zmp_y])

    def compute_zmp_multi_body(self,
                               link_positions: List[np.ndarray],
                               link_masses: List[float],
                               link_accelerations: List[np.ndarray],
                               ground_height: float = 0.0) -> np.ndarray:
        """
        Compute ZMP for multi-body system.

        Full formulation considering all link contributions.

        Args:
            link_positions: List of 3D link CoM positions
            link_masses: List of link masses
            link_accelerations: List of 3D link accelerations
            ground_height: Height of ground plane

        Returns:
            2D ZMP position
        """
        total_mass = sum(link_masses)

        num_x = 0.0
        num_y = 0.0
        denom = 0.0

        for pos, mass, acc in zip(link_positions, link_masses,
                                   link_accelerations):
            z_rel = pos[2] - ground_height

            # Numerator contributions
            num_x += mass * (pos[0] * (self.gravity + acc[2]) - z_rel * acc[0])
            num_y += mass * (pos[1] * (self.gravity + acc[2]) - z_rel * acc[1])

            # Denominator contribution
            denom += mass * (self.gravity + acc[2])

        if abs(denom) < 1e-6:
            denom = 1e-6

        return np.array([num_x / denom, num_y / denom])

    def is_stable(self, zmp: np.ndarray,
                  support: SupportPolygon,
                  margin: float = 0.02) -> bool:
        """
        Check if ZMP is within support polygon.

        Args:
            zmp: 2D ZMP position
            support: Current support polygon
            margin: Safety margin [m]

        Returns:
            True if stable
        """
        return support.contains(zmp, margin)

    def compute_stability_margin(self, zmp: np.ndarray,
                                  support: SupportPolygon) -> float:
        """
        Compute distance from ZMP to support polygon edge.

        Positive = stable, negative = unstable.

        Args:
            zmp: 2D ZMP position
            support: Current support polygon

        Returns:
            Stability margin [m]
        """
        return support.distance_to_edge(zmp)
```

## Linear Inverted Pendulum Model (LIPM)

The **LIPM** simplifies walking dynamics by treating the robot as a point mass on a massless leg, with the CoM constrained to move on a horizontal plane.

```
                LIPM Simplification

    Full Robot                    LIPM Model
    ──────────                    ──────────
        ○ head                        ○ CoM
       /█\                           /
      / █ \                         / L (leg length)
     /  █  \                       /
    ─┼──█──┼─                     /
     │  █  │                     /
     │  █  │                    ● Foot (pivot)
     ├──█──┤               ─────────────────
     │     │                    ZMP
    ═╧═   ═╧═


    Equations of Motion:
    ────────────────────
    ẍ = ω² (x - p)

    where:
        x = CoM position
        p = ZMP position
        ω = √(g/zc) = natural frequency
        zc = constant CoM height


    Solution:
    ─────────
    x(t) = (x₀-p)cosh(ωt) + (ẋ₀/ω)sinh(ωt) + p
    ẋ(t) = (x₀-p)ω·sinh(ωt) + ẋ₀·cosh(ωt)
```

### LIPM Implementation

```python
"""
Linear Inverted Pendulum Model (LIPM) Module

Implements the LIPM dynamics for walking pattern generation.
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class LIPMState:
    """State of the Linear Inverted Pendulum."""
    position: np.ndarray   # 2D CoM position [x, y]
    velocity: np.ndarray   # 2D CoM velocity [vx, vy]
    zmp: np.ndarray        # 2D ZMP position


class LIPM:
    """
    Linear Inverted Pendulum Model.

    Simplified dynamics for bipedal walking that treats
    the robot as a point mass on a massless leg.
    """

    def __init__(self, com_height: float, gravity: float = 9.81):
        """
        Initialize LIPM.

        Args:
            com_height: Constant CoM height [m]
            gravity: Gravitational acceleration [m/s²]
        """
        self.z_c = com_height
        self.g = gravity
        self.omega = np.sqrt(gravity / com_height)

    def compute_trajectory(self,
                          x0: float,
                          v0: float,
                          zmp: float,
                          t: float) -> Tuple[float, float]:
        """
        Compute CoM position and velocity at time t.

        Analytical solution to the LIPM equations.

        Args:
            x0: Initial CoM position
            v0: Initial CoM velocity
            zmp: ZMP position (constant during step)
            t: Time

        Returns:
            Tuple of (position, velocity) at time t
        """
        omega = self.omega

        # Analytical solution
        x = (x0 - zmp) * np.cosh(omega * t) + \
            (v0 / omega) * np.sinh(omega * t) + zmp

        v = (x0 - zmp) * omega * np.sinh(omega * t) + \
            v0 * np.cosh(omega * t)

        return x, v

    def compute_trajectory_2d(self,
                              state: LIPMState,
                              t: float) -> LIPMState:
        """
        Compute 2D trajectory evolution.

        Args:
            state: Initial LIPM state
            t: Time

        Returns:
            State at time t
        """
        x, vx = self.compute_trajectory(
            state.position[0], state.velocity[0], state.zmp[0], t
        )
        y, vy = self.compute_trajectory(
            state.position[1], state.velocity[1], state.zmp[1], t
        )

        return LIPMState(
            position=np.array([x, y]),
            velocity=np.array([vx, vy]),
            zmp=state.zmp.copy()
        )

    def compute_final_state(self,
                            x0: float,
                            v0: float,
                            zmp: float,
                            T: float) -> Tuple[float, float]:
        """
        Compute state at end of step.

        Args:
            x0: Initial position
            v0: Initial velocity
            zmp: ZMP position
            T: Step duration

        Returns:
            Final (position, velocity)
        """
        return self.compute_trajectory(x0, v0, zmp, T)

    def compute_required_zmp(self,
                             x0: float,
                             v0: float,
                             xf: float,
                             vf: float,
                             T: float) -> float:
        """
        Compute ZMP that achieves desired final state.

        Inverts the LIPM equations to find required ZMP.

        Args:
            x0: Initial position
            v0: Initial velocity
            xf: Desired final position
            vf: Desired final velocity
            T: Step duration

        Returns:
            Required ZMP position
        """
        omega = self.omega

        cosh_T = np.cosh(omega * T)
        sinh_T = np.sinh(omega * T)

        # From final velocity equation:
        # vf = (x0 - p) * ω * sinh(ωT) + v0 * cosh(ωT)
        # Solving for p:
        numerator = omega * sinh_T * x0 + cosh_T * v0 - vf
        denominator = omega * sinh_T

        if abs(denominator) < 1e-10:
            # Degenerate case
            return x0

        return numerator / denominator


class LIPMWalker:
    """
    Walking pattern generator using LIPM.

    Generates CoM trajectories for bipedal walking
    based on planned footsteps.
    """

    def __init__(self,
                 com_height: float,
                 step_time: float,
                 double_support_ratio: float = 0.2,
                 gravity: float = 9.81):
        """
        Initialize LIPM walker.

        Args:
            com_height: Constant CoM height [m]
            step_time: Duration of one step [s]
            double_support_ratio: Fraction of step in double support
            gravity: Gravitational acceleration
        """
        self.lipm = LIPM(com_height, gravity)
        self.step_time = step_time
        self.ds_ratio = double_support_ratio
        self.ss_time = step_time * (1 - double_support_ratio)
        self.ds_time = step_time * double_support_ratio

    def plan_zmp_trajectory(self,
                            footsteps: List[np.ndarray]) -> List[np.ndarray]:
        """
        Plan ZMP trajectory through footsteps.

        ZMP moves from foot to foot with smooth transitions.

        Args:
            footsteps: List of 2D footstep positions

        Returns:
            List of ZMP positions for each phase
        """
        zmp_trajectory = []

        for i, foot in enumerate(footsteps):
            # During single support, ZMP at support foot
            zmp_trajectory.append(foot.copy())

            # During double support, ZMP transitions to next foot
            if i < len(footsteps) - 1:
                # Could add intermediate ZMP positions for smoother transition
                pass

        return zmp_trajectory

    def generate_com_trajectory(self,
                                footsteps: List[np.ndarray],
                                initial_velocity: np.ndarray = None,
                                dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate CoM trajectory for walking.

        Uses preview control or analytical LIPM solution.

        Args:
            footsteps: Planned footstep positions
            initial_velocity: Starting CoM velocity
            dt: Time resolution

        Returns:
            Tuple of (positions, velocities) arrays
        """
        if initial_velocity is None:
            initial_velocity = np.zeros(2)

        positions = []
        velocities = []

        # Initial state
        current_pos = footsteps[0].copy()
        current_vel = initial_velocity.copy()

        for i in range(len(footsteps) - 1):
            current_zmp = footsteps[i]
            next_zmp = footsteps[i + 1]

            # Single support phase
            t_steps = int(self.ss_time / dt)
            for t_idx in range(t_steps):
                t = t_idx * dt
                state = LIPMState(current_pos, current_vel, current_zmp)
                new_state = self.lipm.compute_trajectory_2d(state, dt)

                positions.append(new_state.position)
                velocities.append(new_state.velocity)

                current_pos = new_state.position
                current_vel = new_state.velocity

            # Double support phase (ZMP transition)
            t_steps = int(self.ds_time / dt)
            for t_idx in range(t_steps):
                alpha = t_idx / t_steps
                interp_zmp = (1 - alpha) * current_zmp + alpha * next_zmp

                state = LIPMState(current_pos, current_vel, interp_zmp)
                new_state = self.lipm.compute_trajectory_2d(state, dt)

                positions.append(new_state.position)
                velocities.append(new_state.velocity)

                current_pos = new_state.position
                current_vel = new_state.velocity

        return np.array(positions), np.array(velocities)

    def compute_capture_point(self,
                              com_pos: np.ndarray,
                              com_vel: np.ndarray) -> np.ndarray:
        """
        Compute Instantaneous Capture Point (ICP).

        The capture point indicates where to step to stop falling.

        Args:
            com_pos: Current 2D CoM position
            com_vel: Current 2D CoM velocity

        Returns:
            2D capture point position
        """
        return com_pos + com_vel / self.lipm.omega
```

## Capture Point and Divergent Component of Motion

The **Capture Point** (or **Instantaneous Capture Point**, ICP) represents where the robot must step to come to a complete stop. This is crucial for push recovery.

```
            Capture Point Dynamics

         Normal Walking              After Push
         ──────────────              ──────────

              ●──► v                    ●───────► v (increased)
             /                         /
            /                         /
           / CP                      /  CP (moved forward)
          /  ●                      /    ●
         /                         /
    ────●────────           ────●────────────────
      Foot                    Foot

    CP = CoM + v/ω            CP moved! Need to
    (ahead of CoM)            step to new CP location


         Capture Point Control

    ┌─────────────────────────────────────────┐
    │                                         │
    │   Target CP                             │
    │      ○─ ─ ─ ─ ─┐                       │
    │               │                         │
    │   Current CP  │                         │
    │      ●───────►│ Step to here!          │
    │     /         │                         │
    │    /          ▼                         │
    │   ●     [Planned Foot Position]        │
    │  CoM                                    │
    │                                         │
    └─────────────────────────────────────────┘

    To stop walking: Place foot AT capture point
    To continue:     Place foot BEHIND capture point
    To speed up:     Place foot FURTHER behind CP
```

### Capture Point Controller

```python
"""
Capture Point Control Module

Implements Divergent Component of Motion (DCM) / Capture Point
based walking control.
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class DCMState:
    """Divergent Component of Motion state."""
    position: np.ndarray      # 2D DCM/CP position
    velocity: np.ndarray      # 2D DCM velocity
    com_position: np.ndarray  # 2D CoM position
    com_velocity: np.ndarray  # 2D CoM velocity


class CapturePointController:
    """
    Controller based on Capture Point / DCM dynamics.

    The Divergent Component of Motion (DCM) provides a
    powerful framework for bipedal balance and walking control.
    """

    def __init__(self,
                 com_height: float,
                 gravity: float = 9.81,
                 control_gain: float = 2.0):
        """
        Initialize capture point controller.

        Args:
            com_height: Constant CoM height [m]
            gravity: Gravitational acceleration
            control_gain: DCM tracking gain
        """
        self.z_c = com_height
        self.g = gravity
        self.omega = np.sqrt(gravity / com_height)
        self.gain = control_gain

    def compute_capture_point(self,
                              com_pos: np.ndarray,
                              com_vel: np.ndarray) -> np.ndarray:
        """
        Compute Instantaneous Capture Point.

        CP = x + ẋ/ω

        Args:
            com_pos: 2D CoM position
            com_vel: 2D CoM velocity

        Returns:
            2D capture point position
        """
        return com_pos + com_vel / self.omega

    def compute_com_from_dcm(self,
                              dcm: np.ndarray,
                              dcm_vel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Recover CoM state from DCM.

        The CoM lies between the current position and DCM:
        x = ξ - ξ̇/ω

        Args:
            dcm: 2D DCM position
            dcm_vel: 2D DCM velocity

        Returns:
            Tuple of (com_position, com_velocity)
        """
        com_pos = dcm - dcm_vel / self.omega
        com_vel = dcm_vel  # CoM velocity equals DCM velocity

        return com_pos, com_vel

    def compute_dcm_trajectory(self,
                               dcm_end: np.ndarray,
                               foot_pos: np.ndarray,
                               t: float,
                               T: float) -> np.ndarray:
        """
        Compute DCM trajectory during single support.

        DCM follows exponential trajectory toward final DCM:
        ξ(t) = (ξf - p) * exp(ω(t-T)) + p

        Args:
            dcm_end: Final DCM position at end of step
            foot_pos: Current support foot position (= ZMP)
            t: Current time within step
            T: Total step duration

        Returns:
            DCM position at time t
        """
        omega = self.omega
        return (dcm_end - foot_pos) * np.exp(omega * (t - T)) + foot_pos

    def compute_required_foot_position(self,
                                        current_dcm: np.ndarray,
                                        desired_dcm: np.ndarray,
                                        step_time: float) -> np.ndarray:
        """
        Compute where to step to achieve desired DCM.

        Inverts the DCM dynamics to find foot position.

        Args:
            current_dcm: Current DCM position
            desired_dcm: Desired DCM at end of step
            step_time: Duration of the step

        Returns:
            Required foot placement position
        """
        omega = self.omega
        exp_term = np.exp(-omega * step_time)

        # p = (ξf - ξ0 * exp(-ωT)) / (1 - exp(-ωT))
        foot_pos = (desired_dcm - current_dcm * exp_term) / (1 - exp_term)

        return foot_pos

    def compute_vrp(self,
                    dcm: np.ndarray,
                    dcm_ref: np.ndarray) -> np.ndarray:
        """
        Compute Virtual Repellent Point (VRP) for control.

        VRP is the control input that drives DCM toward reference.

        Args:
            dcm: Current DCM position
            dcm_ref: Reference DCM position

        Returns:
            VRP position (use as ZMP reference)
        """
        # VRP = ξref + (1/ω) * K * (ξref - ξ)
        # Simplified: VRP ≈ ξ - (1/ω) * ξ̇_desired
        error = dcm_ref - dcm
        vrp = dcm - self.gain * error / self.omega

        return vrp


class DCMPlanner:
    """
    Plans DCM trajectory for walking.

    Uses backwards recursion from final stance to
    compute DCM waypoints through the gait.
    """

    def __init__(self, lipm: LIPM, step_time: float):
        """
        Initialize DCM planner.

        Args:
            lipm: LIPM model
            step_time: Duration of each step
        """
        self.omega = lipm.omega
        self.step_time = step_time

    def plan_dcm_waypoints(self,
                           footsteps: List[np.ndarray]) -> List[np.ndarray]:
        """
        Plan DCM waypoints for footstep sequence.

        Uses backwards recursion: final DCM at last foot,
        then propagate backwards using DCM dynamics.

        Args:
            footsteps: List of planned footstep positions

        Returns:
            List of DCM waypoints (one per step transition)
        """
        n_steps = len(footsteps)
        dcm_waypoints = [None] * n_steps

        # Final DCM at final foot position (stopped)
        dcm_waypoints[-1] = footsteps[-1].copy()

        # Backwards recursion
        exp_term = np.exp(self.omega * self.step_time)

        for i in range(n_steps - 2, -1, -1):
            foot = footsteps[i]
            dcm_next = dcm_waypoints[i + 1]

            # ξᵢ = (ξᵢ₊₁ - pᵢ) * exp(ωT) + pᵢ
            dcm_waypoints[i] = (dcm_next - foot) * exp_term + foot

        return dcm_waypoints

    def generate_dcm_trajectory(self,
                                footsteps: List[np.ndarray],
                                dt: float = 0.01) -> np.ndarray:
        """
        Generate continuous DCM trajectory.

        Args:
            footsteps: Planned footsteps
            dt: Time resolution

        Returns:
            Array of DCM positions over time
        """
        dcm_waypoints = self.plan_dcm_waypoints(footsteps)
        trajectory = []

        for i, (foot, dcm_end) in enumerate(zip(footsteps[:-1],
                                                 dcm_waypoints[1:])):
            # Generate trajectory during this step
            t_steps = int(self.step_time / dt)

            for t_idx in range(t_steps):
                t = t_idx * dt
                alpha = t / self.step_time

                # DCM trajectory: exponential from current to next waypoint
                dcm = (dcm_end - foot) * np.exp(self.omega * (t - self.step_time)) \
                      + foot
                trajectory.append(dcm)

        return np.array(trajectory)
```

## Footstep Planning

Footstep planning determines where to place feet to navigate to a goal while avoiding obstacles.

```python
"""
Footstep Planning Module

Plans sequences of footsteps for bipedal navigation.
"""

import numpy as np
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import heapq


class FootSide(Enum):
    """Which foot."""
    LEFT = "left"
    RIGHT = "right"


@dataclass
class Footstep:
    """A single footstep."""
    position: np.ndarray    # 2D position [x, y]
    orientation: float      # Yaw angle [rad]
    side: FootSide          # Which foot


@dataclass
class FootstepConstraints:
    """Physical constraints on footsteps."""
    max_step_length: float = 0.40
    min_step_length: float = 0.05
    max_step_width: float = 0.25
    min_step_width: float = 0.05
    max_step_rotation: float = 0.5   # rad
    nominal_width: float = 0.10       # Natural stance width


class FootstepPlanner:
    """
    Plans footstep sequences using A* search.

    Discretizes the footstep space and searches for
    collision-free paths to the goal.
    """

    def __init__(self, constraints: FootstepConstraints):
        """
        Initialize footstep planner.

        Args:
            constraints: Physical footstep constraints
        """
        self.constraints = constraints
        self.obstacle_map = None  # Set externally

    def set_obstacle_map(self, obstacles: List[Tuple[np.ndarray, float]]):
        """
        Set obstacle map for collision checking.

        Args:
            obstacles: List of (center, radius) tuples
        """
        self.obstacle_map = obstacles

    def is_collision_free(self, footstep: Footstep) -> bool:
        """
        Check if footstep is collision-free.

        Args:
            footstep: Footstep to check

        Returns:
            True if no collision
        """
        if self.obstacle_map is None:
            return True

        # Simple circular foot approximation
        foot_radius = 0.10

        for obs_center, obs_radius in self.obstacle_map:
            dist = np.linalg.norm(footstep.position - obs_center[:2])
            if dist < foot_radius + obs_radius:
                return False

        return True

    def get_reachable_footsteps(self,
                                 current: Footstep) -> List[Footstep]:
        """
        Get all valid footsteps from current position.

        Args:
            current: Current footstep

        Returns:
            List of reachable footsteps
        """
        footsteps = []
        c = self.constraints

        # Next foot is opposite side
        next_side = FootSide.RIGHT if current.side == FootSide.LEFT \
                    else FootSide.LEFT

        # Lateral offset direction
        lateral_dir = 1 if next_side == FootSide.LEFT else -1

        # Discretize the reachable space
        for length in np.linspace(c.min_step_length, c.max_step_length, 5):
            for width in np.linspace(c.min_step_width, c.max_step_width, 3):
                for rotation in np.linspace(-c.max_step_rotation,
                                            c.max_step_rotation, 5):
                    # Compute footstep position in current foot frame
                    local_x = length
                    local_y = lateral_dir * width

                    # Rotate to world frame
                    cos_th = np.cos(current.orientation)
                    sin_th = np.sin(current.orientation)

                    world_x = current.position[0] + cos_th * local_x \
                              - sin_th * local_y
                    world_y = current.position[1] + sin_th * local_x \
                              + cos_th * local_y
                    world_th = current.orientation + rotation

                    new_step = Footstep(
                        position=np.array([world_x, world_y]),
                        orientation=world_th,
                        side=next_side
                    )

                    if self.is_collision_free(new_step):
                        footsteps.append(new_step)

        return footsteps

    def heuristic(self, footstep: Footstep, goal: np.ndarray) -> float:
        """
        Heuristic estimate of cost to goal.

        Args:
            footstep: Current footstep
            goal: Goal position

        Returns:
            Estimated cost
        """
        dist = np.linalg.norm(footstep.position - goal)
        # Rough estimate: distance / step_length
        return dist / self.constraints.max_step_length

    def step_cost(self,
                  from_step: Footstep,
                  to_step: Footstep) -> float:
        """
        Cost of taking a step.

        Args:
            from_step: Current footstep
            to_step: Next footstep

        Returns:
            Step cost
        """
        # Distance cost
        dist = np.linalg.norm(to_step.position - from_step.position)

        # Rotation cost
        rot_cost = abs(to_step.orientation - from_step.orientation) * 0.5

        return 1.0 + dist * 2.0 + rot_cost

    def plan(self,
             start_left: Footstep,
             start_right: Footstep,
             goal: np.ndarray,
             start_foot: FootSide = FootSide.LEFT,
             max_steps: int = 50) -> Optional[List[Footstep]]:
        """
        Plan footstep sequence to goal using A*.

        Args:
            start_left: Initial left foot position
            start_right: Initial right foot position
            goal: Goal position [x, y]
            start_foot: Which foot to start with
            max_steps: Maximum number of steps

        Returns:
            List of footsteps, or None if no path found
        """
        # Initialize with starting foot
        start = start_left if start_foot == FootSide.LEFT else start_right

        # Priority queue: (f_score, counter, footstep, path)
        counter = 0
        open_set = [(self.heuristic(start, goal), counter, start, [start])]

        # Track visited states
        visited = set()

        while open_set and counter < max_steps * 100:
            f_score, _, current, path = heapq.heappop(open_set)

            # Check if at goal
            if np.linalg.norm(current.position - goal) < 0.15:
                return path

            # Create state key for visited check
            state_key = (round(current.position[0], 2),
                        round(current.position[1], 2),
                        round(current.orientation, 1),
                        current.side)

            if state_key in visited:
                continue
            visited.add(state_key)

            # Expand neighbors
            for next_step in self.get_reachable_footsteps(current):
                g_score = len(path) * self.step_cost(current, next_step)
                h_score = self.heuristic(next_step, goal)

                counter += 1
                heapq.heappush(
                    open_set,
                    (g_score + h_score, counter, next_step, path + [next_step])
                )

        return None  # No path found


def simple_footstep_plan(start: np.ndarray,
                         goal: np.ndarray,
                         step_length: float = 0.30,
                         step_width: float = 0.10) -> List[np.ndarray]:
    """
    Simple straight-line footstep planning.

    For basic scenarios without obstacles.

    Args:
        start: Start position [x, y]
        goal: Goal position [x, y]
        step_length: Length of each step
        step_width: Width between feet

    Returns:
        List of footstep positions
    """
    direction = goal - start
    distance = np.linalg.norm(direction)

    if distance < 0.01:
        return [start]

    direction = direction / distance

    # Perpendicular direction for foot width
    perp = np.array([-direction[1], direction[0]])

    footsteps = []
    current_pos = start.copy()
    is_left = True

    while np.linalg.norm(goal - current_pos) > step_length:
        # Advance position
        current_pos = current_pos + step_length * direction

        # Offset for left/right foot
        offset = step_width / 2 * perp if is_left else -step_width / 2 * perp
        footsteps.append(current_pos + offset)

        is_left = not is_left

    # Final step to goal
    final_offset = step_width / 2 * perp if is_left else -step_width / 2 * perp
    footsteps.append(goal + final_offset)

    return footsteps
```

## Push Recovery

When a humanoid is pushed, it must quickly adapt to prevent falling. Push recovery strategies include:

```
           Push Recovery Strategies

    1. Ankle Strategy           2. Hip Strategy
    ─────────────────           ───────────────
         ┌───┐                      ┌───┐
         │   │                      │ ↰ │ ← Hip rotation
         │   │                      └─┬─┘
         └─┬─┘                        │
          ─┴─                         │
         ↰ ↰ ← Ankle torque         ─┴─
    ━━━━━━━━━━━━━━              ━━━━━━━━━━━━━━

    Small pushes              Medium pushes
    Fast response             Larger CoM shift


    3. Stepping Strategy
    ────────────────────
              ●
             /│\  ← Push
            / │ \
           /  │  \
          /   │   ● ← Must step here!
    ━━━━━━━━━━━━━━━━━━━━━━━━
         Old  │   New
         foot │   foot

    Large pushes require stepping to new location
    Use capture point to determine where
```

### Push Recovery Controller

```python
"""
Push Recovery Control Module

Implements reactive push recovery strategies.
"""

import numpy as np
from typing import Optional, Tuple
from enum import Enum, auto


class RecoveryStrategy(Enum):
    """Push recovery strategies."""
    ANKLE = auto()      # Small corrections using ankle torque
    HIP = auto()        # Medium corrections using hip motion
    STEP = auto()       # Large corrections requiring stepping


class PushRecoveryController:
    """
    Reactive push recovery controller.

    Selects appropriate recovery strategy based on
    disturbance magnitude and current state.
    """

    def __init__(self,
                 com_height: float,
                 ankle_torque_limit: float = 100.0,
                 hip_angle_limit: float = 0.5,
                 gravity: float = 9.81):
        """
        Initialize push recovery controller.

        Args:
            com_height: CoM height [m]
            ankle_torque_limit: Maximum ankle torque [Nm]
            hip_angle_limit: Maximum hip angle deviation [rad]
            gravity: Gravitational acceleration
        """
        self.z_c = com_height
        self.omega = np.sqrt(gravity / com_height)
        self.tau_max = ankle_torque_limit
        self.theta_max = hip_angle_limit
        self.g = gravity

    def detect_push(self,
                    expected_velocity: np.ndarray,
                    actual_velocity: np.ndarray,
                    threshold: float = 0.1) -> Optional[np.ndarray]:
        """
        Detect if a push has occurred.

        Args:
            expected_velocity: Expected CoM velocity
            actual_velocity: Measured CoM velocity
            threshold: Detection threshold [m/s]

        Returns:
            Push direction and magnitude, or None
        """
        velocity_error = actual_velocity - expected_velocity
        magnitude = np.linalg.norm(velocity_error)

        if magnitude > threshold:
            return velocity_error
        return None

    def select_strategy(self,
                        capture_point: np.ndarray,
                        support_polygon: SupportPolygon) -> RecoveryStrategy:
        """
        Select appropriate recovery strategy.

        Based on where capture point is relative to support polygon.

        Args:
            capture_point: Current capture point position
            support_polygon: Current support polygon

        Returns:
            Selected recovery strategy
        """
        margin = support_polygon.distance_to_edge(capture_point)

        if margin > 0.05:
            # Capture point well inside support - ankle strategy
            return RecoveryStrategy.ANKLE

        elif margin > -0.05:
            # Capture point near edge - hip strategy
            return RecoveryStrategy.HIP

        else:
            # Capture point outside support - must step
            return RecoveryStrategy.STEP

    def ankle_recovery(self,
                       capture_point: np.ndarray,
                       support_center: np.ndarray,
                       robot_mass: float) -> np.ndarray:
        """
        Compute ankle torque for recovery.

        Uses proportional control to drive capture point
        toward support center.

        Args:
            capture_point: Current capture point
            support_center: Center of support polygon
            robot_mass: Total robot mass [kg]

        Returns:
            2D ankle torque command [τx, τy]
        """
        error = capture_point - support_center
        gain = 2.0 * robot_mass * self.g

        torque = -gain * error

        # Saturate to limits
        magnitude = np.linalg.norm(torque)
        if magnitude > self.tau_max:
            torque = torque / magnitude * self.tau_max

        return torque

    def hip_recovery(self,
                     capture_point: np.ndarray,
                     support_center: np.ndarray,
                     upper_body_mass: float) -> Tuple[float, float]:
        """
        Compute hip angles for recovery.

        Shifts upper body mass to adjust CoM.

        Args:
            capture_point: Current capture point
            support_center: Center of support polygon
            upper_body_mass: Mass of upper body [kg]

        Returns:
            Hip angles (pitch, roll) [rad]
        """
        error = capture_point - support_center

        # Gain to convert position error to angle
        gain = 0.5

        hip_pitch = -gain * error[0]  # Forward error -> lean back
        hip_roll = -gain * error[1]   # Lateral error -> lean opposite

        # Saturate
        hip_pitch = np.clip(hip_pitch, -self.theta_max, self.theta_max)
        hip_roll = np.clip(hip_roll, -self.theta_max, self.theta_max)

        return hip_pitch, hip_roll

    def step_recovery(self,
                      capture_point: np.ndarray,
                      current_foot: np.ndarray,
                      swing_foot: np.ndarray,
                      max_step_length: float = 0.5) -> np.ndarray:
        """
        Compute emergency step location for recovery.

        Steps toward the capture point to regain balance.

        Args:
            capture_point: Current capture point
            current_foot: Stance foot position
            swing_foot: Current swing foot position
            max_step_length: Maximum allowable step [m]

        Returns:
            Target position for swing foot
        """
        # Step toward capture point
        step_direction = capture_point - current_foot
        step_length = np.linalg.norm(step_direction)

        if step_length > max_step_length:
            step_direction = step_direction / step_length * max_step_length

        target = current_foot + step_direction

        return target

    def compute_recovery(self,
                         com_position: np.ndarray,
                         com_velocity: np.ndarray,
                         support: SupportPolygon,
                         current_foot: np.ndarray,
                         swing_foot: np.ndarray,
                         robot_mass: float,
                         upper_body_mass: float) -> dict:
        """
        Compute full recovery response.

        Args:
            com_position: 2D CoM position
            com_velocity: 2D CoM velocity
            support: Support polygon
            current_foot: Stance foot position
            swing_foot: Swing foot position
            robot_mass: Total mass
            upper_body_mass: Upper body mass

        Returns:
            Dictionary with recovery commands
        """
        # Compute capture point
        capture_point = com_position + com_velocity / self.omega

        # Select strategy
        strategy = self.select_strategy(capture_point, support)
        support_center = support.get_center()

        result = {
            'strategy': strategy,
            'capture_point': capture_point
        }

        if strategy == RecoveryStrategy.ANKLE:
            result['ankle_torque'] = self.ankle_recovery(
                capture_point, support_center, robot_mass
            )

        elif strategy == RecoveryStrategy.HIP:
            result['hip_pitch'], result['hip_roll'] = self.hip_recovery(
                capture_point, support_center, upper_body_mass
            )

        else:  # STEP
            result['step_target'] = self.step_recovery(
                capture_point, current_foot, swing_foot
            )

        return result
```

## Swing Foot Trajectory

The swing foot must lift, travel, and land smoothly during each step.

```python
"""
Swing Foot Trajectory Module

Generates smooth swing foot trajectories.
"""

import numpy as np
from typing import Tuple


class SwingFootTrajectory:
    """
    Generates smooth swing foot trajectories.

    Uses polynomial interpolation for smooth motion.
    """

    def __init__(self, lift_height: float = 0.05):
        """
        Initialize swing trajectory generator.

        Args:
            lift_height: Maximum foot lift height [m]
        """
        self.lift_height = lift_height

    def compute_trajectory(self,
                          start_pos: np.ndarray,
                          end_pos: np.ndarray,
                          phase: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute swing foot position and velocity.

        Uses quintic polynomial for horizontal motion
        and parabolic profile for vertical.

        Args:
            start_pos: 3D start position
            end_pos: 3D end position
            phase: Phase in [0, 1] (0=start, 1=end)

        Returns:
            Tuple of (position, velocity)
        """
        # Horizontal motion: quintic polynomial (smooth start/end)
        # x(s) = x0 + (xf-x0) * (10s³ - 15s⁴ + 6s⁵)
        s = phase
        s3 = s ** 3
        s4 = s ** 4
        s5 = s ** 5

        interp = 10 * s3 - 15 * s4 + 6 * s5
        interp_vel = 30 * s ** 2 - 60 * s3 + 30 * s4

        pos_xy = start_pos[:2] + (end_pos[:2] - start_pos[:2]) * interp
        vel_xy = (end_pos[:2] - start_pos[:2]) * interp_vel

        # Vertical motion: parabolic (lift in middle)
        # z(s) = z0 + h * 4 * s * (1-s)
        z_ground = min(start_pos[2], end_pos[2])
        pos_z = z_ground + self.lift_height * 4 * s * (1 - s)
        vel_z = self.lift_height * 4 * (1 - 2 * s)

        position = np.array([pos_xy[0], pos_xy[1], pos_z])
        velocity = np.array([vel_xy[0], vel_xy[1], vel_z])

        return position, velocity

    def compute_full_trajectory(self,
                                start_pos: np.ndarray,
                                end_pos: np.ndarray,
                                duration: float,
                                dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate complete swing trajectory.

        Args:
            start_pos: 3D start position
            end_pos: 3D end position
            duration: Swing phase duration [s]
            dt: Time step [s]

        Returns:
            Tuple of (positions, velocities) arrays
        """
        n_steps = int(duration / dt)
        positions = []
        velocities = []

        for i in range(n_steps + 1):
            phase = i / n_steps
            pos, vel = self.compute_trajectory(start_pos, end_pos, phase)
            positions.append(pos)
            velocities.append(vel / duration)  # Scale velocity by duration

        return np.array(positions), np.array(velocities)


class BezierSwingTrajectory:
    """
    Swing foot trajectory using Bezier curves.

    Provides more flexible trajectory shaping.
    """

    def __init__(self, lift_height: float = 0.05,
                 lift_ratio: float = 0.5):
        """
        Initialize Bezier trajectory generator.

        Args:
            lift_height: Maximum foot lift [m]
            lift_ratio: When peak height occurs (0-1)
        """
        self.lift_height = lift_height
        self.lift_ratio = lift_ratio

    def compute_trajectory(self,
                          start_pos: np.ndarray,
                          end_pos: np.ndarray,
                          phase: float) -> np.ndarray:
        """
        Compute swing position using cubic Bezier.

        Control points provide natural-looking motion.

        Args:
            start_pos: 3D start position
            end_pos: 3D end position
            phase: Phase [0, 1]

        Returns:
            3D foot position
        """
        # Control points for cubic Bezier
        p0 = start_pos
        p3 = end_pos

        # Intermediate points: lift up in middle
        p1 = start_pos + np.array([
            (end_pos[0] - start_pos[0]) * 0.25,
            (end_pos[1] - start_pos[1]) * 0.25,
            self.lift_height
        ])

        p2 = end_pos + np.array([
            -(end_pos[0] - start_pos[0]) * 0.25,
            -(end_pos[1] - start_pos[1]) * 0.25,
            self.lift_height * 0.5
        ])

        # Cubic Bezier: B(t) = (1-t)³P0 + 3(1-t)²tP1 + 3(1-t)t²P2 + t³P3
        t = phase
        t1 = 1 - t

        position = (t1 ** 3 * p0 +
                   3 * t1 ** 2 * t * p1 +
                   3 * t1 * t ** 2 * p2 +
                   t ** 3 * p3)

        return position
```

## Industry Perspective: Bipedal Robots Today

Modern bipedal robots showcase different locomotion approaches:

| Platform | Locomotion Approach | Notable Achievements |
|----------|--------------------|--------------------|
| **Boston Dynamics Atlas** | Model Predictive Control + Learning | Parkour, dynamic gymnastics |
| **Agility Digit** | Template-based + Capture Point | Warehouse logistics walking |
| **Tesla Optimus** | Classical control, improving | Basic walking, stair climbing |
| **Unitree H1** | Reinforcement Learning | Running at 3.3 m/s |
| **Figure 01/02** | Learning-based | Stable walking with manipulation |
| **MIT Humanoid** | MPC + Whole-body control | Research demonstrations |

**Key trends:**
1. **Learning-based methods** are increasingly used for robustness
2. **MPC** provides optimal trajectories for known dynamics
3. **Hybrid approaches** combine model-based and learning
4. **Sim-to-real transfer** enables training in simulation

## Summary

```
┌────────────────────────────────────────────────────────────────────┐
│                   Bipedal Locomotion Recap                         │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Balance Criteria                                                  │
│  ────────────────                                                  │
│  • ZMP inside support polygon → No tipping                        │
│  • Capture Point determines step location                          │
│  • Stability margin = distance to polygon edge                    │
│                                                                    │
│  Simplified Models                                                 │
│  ─────────────────                                                 │
│  • LIPM: Point mass on massless leg                               │
│  • Natural frequency ω = √(g/z_c)                                 │
│  • Analytical trajectory solutions                                │
│                                                                    │
│  Walking Pattern Generation                                        │
│  ──────────────────────────                                        │
│  • Plan footsteps to goal                                         │
│  • Generate ZMP trajectory through feet                           │
│  • Compute CoM trajectory from LIPM                               │
│  • Swing foot with smooth lift profile                            │
│                                                                    │
│  Push Recovery                                                     │
│  ─────────────                                                     │
│  • Ankle strategy: Small disturbances                             │
│  • Hip strategy: Medium disturbances                              │
│  • Stepping strategy: Large disturbances                          │
│                                                                    │
│  Capture Point Control                                             │
│  ─────────────────────                                             │
│  • CP = CoM + velocity/ω                                          │
│  • Step to CP to stop                                             │
│  • Step behind CP to continue walking                             │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### Key Equations Reference

| Concept | Equation |
|---------|----------|
| ZMP | p_zmp = p_com - (z_com/g)·a_com |
| LIPM dynamics | ẍ = ω²(x - p) |
| Natural frequency | ω = √(g/z_c) |
| LIPM solution | x(t) = (x₀-p)cosh(ωt) + (v₀/ω)sinh(ωt) + p |
| Capture Point | ξ = x + ẋ/ω |
| DCM trajectory | ξ(t) = (ξ_f - p)exp(ω(t-T)) + p |

### Implementation Checklist

- [ ] Implement ZMP computation from CoM state
- [ ] Build support polygon from foot contacts
- [ ] Create LIPM dynamics for trajectory generation
- [ ] Implement capture point computation
- [ ] Build DCM trajectory planner with backwards recursion
- [ ] Create footstep planner (A* or RRT)
- [ ] Implement swing foot trajectory generator
- [ ] Add push detection and recovery selection
- [ ] Test in simulation before hardware

## Further Reading

- Kajita, S. "Introduction to Humanoid Robotics" (2014)
- Pratt, J. "Capture Point: A Step toward Humanoid Push Recovery" (2006)
- Englsberger, J. "Three-Dimensional Bipedal Walking Control Based on DCM" (2015)
- Takenaka, T. "The Control System for the Honda Humanoid Robot" (2009)
- Wieber, P.B. "Trajectory Free Linear Model Predictive Control for Stable Walking" (2006)
