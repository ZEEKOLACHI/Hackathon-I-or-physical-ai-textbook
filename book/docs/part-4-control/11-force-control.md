---
id: ch-4-11
title: Force Control
sidebar_position: 2
difficulty: advanced
estimated_time: 50
prerequisites: [ch-4-10]
---

# Force Control

> "A robot that cannot feel cannot truly interact."
> — Oussama Khatib, Stanford Robotics Lab

While position control dominates traditional robotics, the future belongs to robots that can sense and control forces. From a robot gently holding a fragile egg to a humanoid maintaining balance on uneven terrain, force control is what transforms rigid machines into capable collaborators in the physical world.

## Why Force Control Matters

Consider the difference between a robot and a human picking up a glass:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  POSITION CONTROL vs FORCE CONTROL                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Position Control (Traditional)          Force Control (Interaction)       │
│                                                                             │
│   "Move to position (x,y,z)"              "Apply force F while moving"      │
│                                                                             │
│        │                                       │                            │
│        ▼                                       ▼                            │
│   ┌─────────┐                             ┌─────────┐                       │
│   │  Glass  │  CRACK!                     │  Glass  │  Safe!               │
│   └─────────┘                             └─────────┘                       │
│        │                                       │                            │
│   Environment is rigid?                   Adapt to environment              │
│   → Works perfectly                       → Works in any situation          │
│                                                                             │
│   Environment is compliant or             Sense and respond to              │
│   position uncertain?                     contact forces                    │
│   → May crush or miss                     → Gentle, robust interaction      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## The Interaction Control Spectrum

Force control exists on a spectrum from pure position to pure force control:

```
                        Interaction Control Spectrum
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  Pure Position                                               Pure Force     │
│  Control                                                     Control        │
│     │                                                           │           │
│     │◄─────────────────────────────────────────────────────────►│           │
│     │                                                           │           │
│     │     Stiff         Impedance        Admittance      Soft   │           │
│     │   Position        Control          Control        Force   │           │
│     │   Control                                        Control  │           │
│     │                                                           │           │
│     ▼                                                           ▼           │
│  High stiffness                                          Low stiffness      │
│  Position matters                                        Force matters      │
│  Tasks: Machining                                       Tasks: Polishing    │
│         Pick & Place                                           Assembly     │
│                                                                Massage      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Impedance Control

Impedance control regulates the dynamic relationship between force and motion. Instead of controlling position or force directly, we control how the robot *responds* to perturbations.

### The Impedance Concept

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     MECHANICAL IMPEDANCE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Physical Analogy: Mass-Spring-Damper System                              │
│                                                                             │
│                     ┌───────────────────┐                                  │
│   External    ────▶ │   ╭───╮   ┌───┐  │                                  │
│   Force F           │   │ M │═══│///│  │ ──▶ Motion x                     │
│                     │   ╰───╯   └───┘  │                                  │
│                     │    ‖      Damper │                                  │
│                     │   /\/\     (B)   │                                  │
│                     │  Spring         │                                   │
│                     │   (K)           │                                   │
│                     └───────────────────┘                                  │
│                                                                             │
│   Impedance Equation:                                                       │
│                                                                             │
│   F = M·ẍ + B·ẋ + K·(x - x₀)                                              │
│                                                                             │
│   Where:                                                                    │
│   - M = Virtual inertia (how sluggish the response is)                     │
│   - B = Virtual damping (energy dissipation)                               │
│   - K = Virtual stiffness (spring constant)                                │
│   - x₀ = Equilibrium position                                              │
│                                                                             │
│   Key insight: We don't control F or x directly.                           │
│   We control the RELATIONSHIP between them!                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Impedance Controller Implementation

```python
"""
Impedance Control Implementation

Impedance control makes the robot behave like a virtual mass-spring-damper
system. When external forces are applied, the robot moves according to
the specified impedance parameters.

This is fundamentally different from position control:
- Position control: "Go to position X regardless of forces"
- Impedance control: "Behave as if you have mass M, damping B, stiffness K"

Applications:
- Human-robot collaboration
- Contact tasks (polishing, insertion)
- Safe manipulation
- Haptic devices
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum


class ImpedanceType(Enum):
    """Types of impedance behavior."""
    COMPLIANT = "compliant"      # Low stiffness, robot yields to forces
    STIFF = "stiff"             # High stiffness, robot resists forces
    CRITICAL_DAMPED = "critical" # No oscillation, fast settling


@dataclass
class ImpedanceParams:
    """
    Impedance parameters for one degree of freedom.

    Choosing parameters:
    - Higher M: Slower response, more inertia-like
    - Higher B: More energy dissipation, overdamped
    - Higher K: Stiffer, more position-like behavior

    Critical damping: B = 2 * sqrt(M * K)
    """
    mass: float       # Virtual inertia [kg] or [kg·m²]
    damping: float    # Virtual damping [N·s/m] or [N·m·s/rad]
    stiffness: float  # Virtual stiffness [N/m] or [N·m/rad]

    def __post_init__(self):
        """Validate parameters."""
        if self.mass <= 0:
            raise ValueError("Mass must be positive")
        if self.damping < 0:
            raise ValueError("Damping must be non-negative")
        if self.stiffness < 0:
            raise ValueError("Stiffness must be non-negative")

    @property
    def damping_ratio(self) -> float:
        """
        Compute damping ratio ζ.

        ζ < 1: Underdamped (oscillatory)
        ζ = 1: Critically damped (fastest non-oscillatory)
        ζ > 1: Overdamped (slow, no oscillation)
        """
        if self.stiffness == 0:
            return float('inf')
        omega_n = np.sqrt(self.stiffness / self.mass)
        return self.damping / (2 * self.mass * omega_n)

    @property
    def natural_frequency(self) -> float:
        """Natural frequency in rad/s."""
        if self.stiffness == 0:
            return 0.0
        return np.sqrt(self.stiffness / self.mass)

    @classmethod
    def critical_damped(cls, mass: float, stiffness: float) -> 'ImpedanceParams':
        """Create critically damped impedance parameters."""
        damping = 2 * np.sqrt(mass * stiffness)
        return cls(mass=mass, damping=damping, stiffness=stiffness)


class ImpedanceController:
    """
    Cartesian space impedance controller.

    Makes the robot end-effector behave like a virtual mass-spring-damper
    system in Cartesian space.

    Block Diagram:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                                                                     │
    │   x_des ──┐                                                        │
    │           │    ┌──────────────┐                                    │
    │   ẋ_des ──┼───▶│  Impedance   │──▶ ẍ_des ──▶ [Dynamics] ──▶ τ     │
    │           │    │   Model      │                                    │
    │   F_ext ──┘    └──────────────┘                                    │
    │                                                                     │
    │   Impedance Model:                                                  │
    │   M·ẍ + B·ẋ + K·(x - x_des) = F_ext                               │
    │                                                                     │
    │   Solving for ẍ:                                                   │
    │   ẍ = M⁻¹ · (F_ext - B·ẋ - K·(x - x_des))                         │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        mass: np.ndarray,
        damping: np.ndarray,
        stiffness: np.ndarray,
        dt: float = 0.001
    ):
        """
        Initialize impedance controller.

        Args:
            mass: Virtual inertia matrix (6x6 for Cartesian space)
            damping: Virtual damping matrix (6x6)
            stiffness: Virtual stiffness matrix (6x6)
            dt: Control timestep

        Note: Matrices should be positive definite for stability.
        """
        self.M = np.atleast_2d(mass)
        self.B = np.atleast_2d(damping)
        self.K = np.atleast_2d(stiffness)
        self.dt = dt

        # Ensure matrices are correct size
        self.ndim = self.M.shape[0]
        self._validate_matrices()

        # Precompute inverse for efficiency
        self.M_inv = np.linalg.inv(self.M)

    def _validate_matrices(self):
        """Validate impedance matrices."""
        for name, mat in [('M', self.M), ('B', self.B), ('K', self.K)]:
            if mat.shape != (self.ndim, self.ndim):
                raise ValueError(f"{name} must be {self.ndim}x{self.ndim}")
            # Check positive definiteness for M
            if name == 'M' and np.any(np.linalg.eigvals(mat) <= 0):
                raise ValueError("Mass matrix must be positive definite")

    def compute_acceleration(
        self,
        x_des: np.ndarray,
        x_dot_des: np.ndarray,
        x: np.ndarray,
        x_dot: np.ndarray,
        f_ext: np.ndarray
    ) -> np.ndarray:
        """
        Compute desired acceleration based on impedance model.

        Args:
            x_des: Desired position (6D: position + orientation)
            x_dot_des: Desired velocity
            x: Current position
            x_dot: Current velocity
            f_ext: External force/torque (measured or estimated)

        Returns:
            Desired acceleration that realizes impedance behavior
        """
        # Position and velocity errors
        pos_error = x - x_des
        vel_error = x_dot - x_dot_des

        # Impedance equation: M·ẍ + B·(ẋ - ẋ_des) + K·(x - x_des) = f_ext
        # Solving for ẍ:
        x_ddot_des = self.M_inv @ (f_ext - self.B @ vel_error - self.K @ pos_error)

        return x_ddot_des

    def compute_torques(
        self,
        robot,
        x_des: np.ndarray,
        x_dot_des: np.ndarray,
        f_ext: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute joint torques for impedance behavior.

        Args:
            robot: Robot model with kinematics and dynamics
            x_des: Desired Cartesian pose
            x_dot_des: Desired Cartesian velocity
            f_ext: External force (if None, assumed zero)

        Returns:
            Joint torques
        """
        # Get current state
        x = robot.end_effector_pose()
        x_dot = robot.end_effector_velocity()

        if f_ext is None:
            f_ext = np.zeros(self.ndim)

        # Compute desired acceleration from impedance model
        x_ddot_des = self.compute_acceleration(x_des, x_dot_des, x, x_dot, f_ext)

        # Get Jacobian
        J = robot.jacobian()

        # Operational space inertia matrix
        M_q = robot.mass_matrix()
        Lambda = np.linalg.inv(J @ np.linalg.inv(M_q) @ J.T)

        # Coriolis and gravity in operational space
        mu = Lambda @ (J @ np.linalg.inv(M_q) @ robot.coriolis_vector() -
                       robot.jacobian_derivative() @ robot.joint_velocities())
        p = Lambda @ J @ np.linalg.inv(M_q) @ robot.gravity_vector()

        # Operational space dynamics: F = Lambda·ẍ + mu + p
        F_cmd = Lambda @ x_ddot_des + mu + p

        # Map to joint torques
        tau = J.T @ F_cmd

        return tau


class ImpedanceController1D:
    """
    Simplified 1D impedance controller for educational purposes.

    Perfect for understanding the core concept before moving to
    full Cartesian space implementation.
    """

    def __init__(
        self,
        mass: float,
        damping: float,
        stiffness: float,
        dt: float = 0.001
    ):
        """
        Initialize 1D impedance controller.

        Args:
            mass: Virtual inertia [kg]
            damping: Virtual damping [N·s/m]
            stiffness: Virtual stiffness [N/m]
            dt: Control timestep [s]

        Example:
            >>> controller = ImpedanceController1D(mass=1.0, damping=10.0, stiffness=100.0)
            >>> # With K=100 N/m, pushing with 10N yields 0.1m displacement
        """
        self.params = ImpedanceParams(mass, damping, stiffness)
        self.dt = dt

        # State for simulation
        self.x = 0.0
        self.x_dot = 0.0

    def compute(
        self,
        x_des: float,
        x: float,
        x_dot: float,
        f_ext: float
    ) -> float:
        """
        Compute desired acceleration.

        Args:
            x_des: Desired position
            x: Current position
            x_dot: Current velocity
            f_ext: External force (positive = pushing robot)

        Returns:
            Desired acceleration
        """
        # Impedance equation: M·ẍ + B·ẋ + K·(x - x_des) = f_ext
        x_ddot = (
            f_ext
            - self.params.damping * x_dot
            - self.params.stiffness * (x - x_des)
        ) / self.params.mass

        return x_ddot

    def simulate_step(self, x_des: float, f_ext: float) -> Tuple[float, float]:
        """
        Simulate one timestep of impedance behavior.

        Args:
            x_des: Desired position
            f_ext: External force

        Returns:
            Tuple of (position, velocity)
        """
        x_ddot = self.compute(x_des, self.x, self.x_dot, f_ext)

        # Simple Euler integration
        self.x_dot += x_ddot * self.dt
        self.x += self.x_dot * self.dt

        return self.x, self.x_dot


# Demonstration of impedance behavior
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Create three controllers with different characteristics
    stiff = ImpedanceController1D(mass=1.0, damping=20.0, stiffness=500.0)
    medium = ImpedanceController1D(mass=1.0, damping=14.0, stiffness=100.0)
    compliant = ImpedanceController1D(mass=1.0, damping=6.0, stiffness=20.0)

    # Simulate response to step force
    t = np.arange(0, 2.0, 0.001)
    f_ext = 10.0  # 10N push

    results = {'stiff': [], 'medium': [], 'compliant': []}

    for controller, name in [(stiff, 'stiff'), (medium, 'medium'), (compliant, 'compliant')]:
        controller.x = 0.0
        controller.x_dot = 0.0

        for _ in t:
            x, _ = controller.simulate_step(x_des=0.0, f_ext=f_ext)
            results[name].append(x)

    print("Steady-state displacements (x = F/K):")
    print(f"  Stiff (K=500): {results['stiff'][-1]:.4f} m (expected: {f_ext/500:.4f})")
    print(f"  Medium (K=100): {results['medium'][-1]:.4f} m (expected: {f_ext/100:.4f})")
    print(f"  Compliant (K=20): {results['compliant'][-1]:.4f} m (expected: {f_ext/20:.4f})")
```

## Admittance Control

While impedance control outputs motion commands, admittance control takes the opposite approach: it measures forces and outputs position/velocity commands.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              IMPEDANCE vs ADMITTANCE CONTROL                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   IMPEDANCE CONTROL:                                                        │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                  │
│   │   Motion    │────▶│  Impedance  │────▶│   Force     │                  │
│   │   Input     │     │  Controller │     │   Output    │                  │
│   └─────────────┘     └─────────────┘     └─────────────┘                  │
│                                                                             │
│   Best for: Torque-controlled robots (e.g., modern collaborative robots)   │
│                                                                             │
│   ─────────────────────────────────────────────────────────────────────    │
│                                                                             │
│   ADMITTANCE CONTROL:                                                       │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                  │
│   │   Force     │────▶│ Admittance  │────▶│   Motion    │                  │
│   │   Input     │     │  Controller │     │   Output    │                  │
│   └─────────────┘     └─────────────┘     └─────────────┘                  │
│                                                                             │
│   Best for: Position-controlled robots (e.g., traditional industrial)      │
│                                                                             │
│   ─────────────────────────────────────────────────────────────────────    │
│                                                                             │
│   Mathematical relationship:                                                │
│   Impedance Z(s) = F(s) / X(s) = Ms² + Bs + K                              │
│   Admittance Y(s) = X(s) / F(s) = 1 / (Ms² + Bs + K) = 1/Z(s)             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Admittance Controller Implementation

```python
"""
Admittance Control Implementation

Admittance control is the "dual" of impedance control:
- Impedance: Motion → Force (control effort is force/torque)
- Admittance: Force → Motion (control effort is position/velocity)

Use admittance control when:
1. Robot has stiff position control (most industrial robots)
2. Force/torque sensor is available
3. You want force-guided motion

The controller integrates external forces to produce position adjustments.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class AdmittanceState:
    """State of the admittance controller."""
    position: np.ndarray      # Current virtual position
    velocity: np.ndarray      # Current virtual velocity
    acceleration: np.ndarray  # Current acceleration


class AdmittanceController:
    """
    Cartesian space admittance controller.

    Takes force measurements as input and outputs position commands
    that a position-controlled robot can track.

    Block Diagram:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                                                                     │
    │   F_ext ──▶ ┌──────────────┐ ──▶ x_cmd ──▶ [Position    ] ──▶ x   │
    │             │  Admittance  │              [  Control    ]          │
    │   x_des ──▶ │   Model      │                                       │
    │             └──────────────┘                                       │
    │                                                                     │
    │   Admittance Model (state-space integration):                      │
    │   M·ẍ + B·ẋ + K·(x - x_des) = F_ext                               │
    │   x_cmd = x_des + Δx  (where Δx comes from integrating above)     │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        mass: np.ndarray,
        damping: np.ndarray,
        stiffness: np.ndarray,
        dt: float = 0.001
    ):
        """
        Initialize admittance controller.

        Args:
            mass: Virtual inertia matrix (diagonal for decoupled axes)
            damping: Virtual damping matrix
            stiffness: Virtual stiffness matrix (can be zero for pure velocity mode)
            dt: Control timestep
        """
        # Convert to diagonal matrices if 1D arrays provided
        if mass.ndim == 1:
            self.M = np.diag(mass)
            self.B = np.diag(damping)
            self.K = np.diag(stiffness)
        else:
            self.M = mass
            self.B = damping
            self.K = stiffness

        self.dt = dt
        self.ndim = self.M.shape[0]

        # Precompute inverse
        self.M_inv = np.linalg.inv(self.M)

        # State
        self._delta_x = np.zeros(self.ndim)      # Position offset
        self._delta_x_dot = np.zeros(self.ndim)  # Velocity

        # Force filtering (low-pass)
        self._f_filtered = np.zeros(self.ndim)
        self._filter_alpha = 0.1  # Lower = more filtering

    def reset(self, initial_position: Optional[np.ndarray] = None):
        """Reset controller state."""
        self._delta_x = np.zeros(self.ndim)
        self._delta_x_dot = np.zeros(self.ndim)
        self._f_filtered = np.zeros(self.ndim)

    def set_filter_cutoff(self, cutoff_freq: float):
        """
        Set force filter cutoff frequency.

        Args:
            cutoff_freq: Cutoff frequency in Hz
        """
        # First-order low-pass filter coefficient
        tau = 1.0 / (2 * np.pi * cutoff_freq)
        self._filter_alpha = self.dt / (tau + self.dt)

    def compute(
        self,
        f_ext: np.ndarray,
        x_des: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute position command from external force.

        Args:
            f_ext: Measured external force/torque
            x_des: Desired equilibrium position

        Returns:
            Tuple of (position_command, velocity_command)

        Note:
            The position command should be sent to a position controller.
            The returned position is x_des + delta_x, where delta_x is
            the admittance-generated offset.
        """
        # Filter force measurement
        self._f_filtered = (
            self._filter_alpha * f_ext +
            (1 - self._filter_alpha) * self._f_filtered
        )

        # Admittance equation: M·ẍ + B·ẋ + K·Δx = F_ext
        # Solving for ẍ:
        delta_x_ddot = self.M_inv @ (
            self._f_filtered
            - self.B @ self._delta_x_dot
            - self.K @ self._delta_x
        )

        # Integrate acceleration to velocity
        self._delta_x_dot += delta_x_ddot * self.dt

        # Integrate velocity to position
        self._delta_x += self._delta_x_dot * self.dt

        # Compute commanded position
        x_cmd = x_des + self._delta_x
        x_dot_cmd = self._delta_x_dot

        return x_cmd, x_dot_cmd

    def get_state(self) -> AdmittanceState:
        """Get current controller state."""
        delta_x_ddot = self.M_inv @ (
            self._f_filtered
            - self.B @ self._delta_x_dot
            - self.K @ self._delta_x
        )
        return AdmittanceState(
            position=self._delta_x.copy(),
            velocity=self._delta_x_dot.copy(),
            acceleration=delta_x_ddot
        )


class AdmittanceController1D:
    """
    1D Admittance controller for educational purposes.

    Demonstrates the force-to-motion transformation in its simplest form.
    """

    def __init__(
        self,
        mass: float,
        damping: float,
        stiffness: float,
        dt: float = 0.001
    ):
        """
        Initialize 1D admittance controller.

        Args:
            mass: Virtual inertia [kg]
            damping: Virtual damping [N·s/m]
            stiffness: Virtual stiffness [N/m] (0 for pure velocity control)
            dt: Control timestep [s]

        Common configurations:
        - Pure velocity mode: K=0, robot drifts with force
        - Spring return: K>0, robot returns to x_des when force removed
        - High damping: B>>sqrt(M*K), sluggish but stable
        """
        self.M = mass
        self.B = damping
        self.K = stiffness
        self.dt = dt

        # State
        self.x = 0.0       # Position offset from x_des
        self.x_dot = 0.0   # Velocity

    def compute(self, f_ext: float, x_des: float = 0.0) -> float:
        """
        Compute position command from measured force.

        Args:
            f_ext: External force (from F/T sensor)
            x_des: Desired equilibrium position

        Returns:
            Position command to send to position controller
        """
        # Admittance equation: M·ẍ + B·ẋ + K·x = f_ext
        x_ddot = (f_ext - self.B * self.x_dot - self.K * self.x) / self.M

        # Integrate
        self.x_dot += x_ddot * self.dt
        self.x += self.x_dot * self.dt

        return x_des + self.x

    def reset(self):
        """Reset to initial state."""
        self.x = 0.0
        self.x_dot = 0.0


# Example: Human guiding a robot arm
if __name__ == "__main__":
    print("Admittance Control Demo: Human-guided motion")
    print("=" * 50)

    # Create controller (moderate compliance for safe interaction)
    controller = AdmittanceController1D(
        mass=5.0,      # 5 kg virtual mass
        damping=50.0,  # Critically damped
        stiffness=0.0,  # No spring return (pure velocity mode)
        dt=0.01
    )

    # Simulate human pushing with 20N for 2 seconds, then releasing
    positions = []
    forces = []
    time = []

    for t in np.arange(0, 5, 0.01):
        # Human applies 20N push for first 2 seconds
        if t < 2.0:
            f_human = 20.0
        else:
            f_human = 0.0

        x_cmd = controller.compute(f_human, x_des=0.0)

        positions.append(x_cmd)
        forces.append(f_human)
        time.append(t)

    print(f"After 2s of 20N push: position = {positions[200]:.3f} m")
    print(f"After release (5s): position = {positions[-1]:.3f} m")
    print("(With K=0, robot maintains position after release)")
```

## Hybrid Position/Force Control

Many tasks require controlling position in some directions while controlling force in others. Hybrid control provides exactly this capability.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HYBRID POSITION/FORCE CONTROL                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Example: Surface wiping task                                              │
│                                                                             │
│            ──────────────────▶ X (position control: follow path)           │
│           │                                                                 │
│           │  Robot                                                          │
│           │  End-effector                                                   │
│           │      ▼                                                          │
│           │  ┌───────┐                                                     │
│       Z   │  │ Sponge│ ▬▬▬▬▬▬▬ Contact force                              │
│    force  │  └───────┘                                                     │
│   control │  ═════════════════════════════════ Surface                     │
│           ▼                                                                 │
│                                                                             │
│   Selection Matrix S:                                                       │
│   ┌─────────────────┐                                                      │
│   │ 1 0 0 0 0 0 │  Position control in X                                   │
│   │ 0 1 0 0 0 0 │  Position control in Y                                   │
│   │ 0 0 0 0 0 0 │  Force control in Z (0 means force control)              │
│   │ 0 0 0 1 0 0 │  Position control in Rx                                  │
│   │ 0 0 0 0 1 0 │  Position control in Ry                                  │
│   │ 0 0 0 0 0 1 │  Position control in Rz                                  │
│   └─────────────────┘                                                      │
│                                                                             │
│   Control Law:                                                              │
│   u = S·u_position + (I - S)·u_force                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Hybrid Controller Implementation

```python
"""
Hybrid Position/Force Control

Combines position control in some directions with force control in others.
The key is the selection matrix S that determines which DOFs are
position-controlled (S=1) vs force-controlled (S=0).

Classic reference: Raibert & Craig (1981) "Hybrid Position/Force Control
of Manipulators"
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class ControlMode(Enum):
    """Control mode for each degree of freedom."""
    POSITION = "position"
    FORCE = "force"


@dataclass
class HybridControlConfig:
    """Configuration for hybrid control."""
    modes: list  # List of ControlMode for each DOF
    position_gains: np.ndarray  # [Kp, Kd] for position control
    force_gains: np.ndarray     # [Kp, Ki] for force control

    @property
    def selection_matrix(self) -> np.ndarray:
        """Generate selection matrix from modes."""
        n = len(self.modes)
        S = np.diag([1.0 if m == ControlMode.POSITION else 0.0 for m in self.modes])
        return S


class PIDController:
    """Simple PID controller for hybrid control."""

    def __init__(self, kp: float, ki: float, kd: float, dt: float = 0.001):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error: float) -> float:
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0


class HybridController:
    """
    Hybrid position/force controller in Cartesian space.

    Controls some DOFs in position, others in force, based on
    the selection matrix.

    Mathematical formulation:
        u = S @ u_pos + (I - S) @ u_force

    Where:
        S: Selection matrix (diagonal, 1=position, 0=force)
        u_pos: Output of position controller
        u_force: Output of force controller

    Important: Position and force are controlled in orthogonal
    subspaces - you cannot control both in the same direction!
    """

    def __init__(
        self,
        n_dof: int = 6,
        dt: float = 0.001
    ):
        """
        Initialize hybrid controller.

        Args:
            n_dof: Number of degrees of freedom (typically 6 for Cartesian)
            dt: Control timestep
        """
        self.n_dof = n_dof
        self.dt = dt

        # Default: position control everywhere
        self.S = np.eye(n_dof)

        # Position controllers (one per DOF)
        self.pos_controllers = [
            PIDController(kp=100.0, ki=0.0, kd=20.0, dt=dt)
            for _ in range(n_dof)
        ]

        # Force controllers (one per DOF)
        self.force_controllers = [
            PIDController(kp=0.001, ki=0.0001, kd=0.0, dt=dt)
            for _ in range(n_dof)
        ]

    def set_selection_matrix(self, S: np.ndarray):
        """
        Set the selection matrix.

        Args:
            S: Diagonal matrix where 1=position control, 0=force control

        Example:
            # Force control only in Z direction
            S = np.diag([1, 1, 0, 1, 1, 1])
            controller.set_selection_matrix(S)
        """
        if S.shape != (self.n_dof, self.n_dof):
            raise ValueError(f"S must be {self.n_dof}x{self.n_dof}")
        self.S = S

    def set_position_gains(self, kp: float, ki: float, kd: float):
        """Set gains for all position controllers."""
        for ctrl in self.pos_controllers:
            ctrl.kp = kp
            ctrl.ki = ki
            ctrl.kd = kd

    def set_force_gains(self, kp: float, ki: float, kd: float = 0.0):
        """Set gains for all force controllers."""
        for ctrl in self.force_controllers:
            ctrl.kp = kp
            ctrl.ki = ki
            ctrl.kd = kd

    def compute(
        self,
        x_des: np.ndarray,
        f_des: np.ndarray,
        x: np.ndarray,
        f: np.ndarray,
        x_dot: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute control output using hybrid position/force control.

        Args:
            x_des: Desired position (6D Cartesian pose)
            f_des: Desired force (6D wrench)
            x: Current position
            f: Current measured force
            x_dot: Current velocity (optional, for derivative term)

        Returns:
            Control output (can be velocity command or force command
            depending on robot interface)
        """
        # Ensure arrays
        x_des = np.atleast_1d(x_des)
        f_des = np.atleast_1d(f_des)
        x = np.atleast_1d(x)
        f = np.atleast_1d(f)

        # Position control output
        pos_output = np.zeros(self.n_dof)
        for i, ctrl in enumerate(self.pos_controllers):
            pos_error = x_des[i] - x[i]
            pos_output[i] = ctrl.compute(pos_error)

        # Force control output
        force_output = np.zeros(self.n_dof)
        for i, ctrl in enumerate(self.force_controllers):
            force_error = f_des[i] - f[i]
            force_output[i] = ctrl.compute(force_error)

        # Hybrid combination
        I = np.eye(self.n_dof)
        output = self.S @ pos_output + (I - self.S) @ force_output

        return output

    def reset(self):
        """Reset all controller states."""
        for ctrl in self.pos_controllers:
            ctrl.reset()
        for ctrl in self.force_controllers:
            ctrl.reset()


class SurfaceFollowingController(HybridController):
    """
    Specialized hybrid controller for surface following tasks.

    Common in applications like:
    - Polishing/grinding
    - Painting
    - Inspection
    - Deburring

    Controls:
    - Force normal to surface (maintain contact)
    - Position tangent to surface (follow trajectory)
    """

    def __init__(self, contact_axis: int = 2, dt: float = 0.001):
        """
        Initialize surface following controller.

        Args:
            contact_axis: Axis normal to surface (0=X, 1=Y, 2=Z)
            dt: Control timestep
        """
        super().__init__(n_dof=6, dt=dt)

        # Set selection matrix: force control on contact axis
        S = np.eye(6)
        S[contact_axis, contact_axis] = 0.0  # Force control on this axis
        self.set_selection_matrix(S)

        self.contact_axis = contact_axis

        # Tune for surface following
        self.set_position_gains(kp=50.0, ki=0.0, kd=10.0)
        self.set_force_gains(kp=0.002, ki=0.0002, kd=0.0)

    def set_contact_force(self, force: float):
        """
        Set desired contact force.

        Args:
            force: Desired normal force in Newtons (positive = pushing)
        """
        self._desired_force = force

    def compute_surface_following(
        self,
        trajectory_point: np.ndarray,
        current_pose: np.ndarray,
        current_force: np.ndarray,
        desired_normal_force: float = 10.0
    ) -> np.ndarray:
        """
        Compute control for surface following.

        Args:
            trajectory_point: Desired position on surface
            current_pose: Current end-effector pose
            current_force: Current measured force
            desired_normal_force: Target contact force [N]

        Returns:
            Control output for robot
        """
        # Construct desired force vector
        f_des = np.zeros(6)
        f_des[self.contact_axis] = desired_normal_force

        return self.compute(
            x_des=trajectory_point,
            f_des=f_des,
            x=current_pose,
            f=current_force
        )


# Example: Wiping a table
if __name__ == "__main__":
    print("Hybrid Control Demo: Table Wiping")
    print("=" * 50)

    # Create surface following controller (force control in Z)
    controller = SurfaceFollowingController(contact_axis=2, dt=0.01)

    # Simulate wiping motion
    print("\nSimulating wiping trajectory...")
    print("- Position control in X, Y (follow path)")
    print("- Force control in Z (maintain 5N contact)")

    # Simple wiping path: straight line in X
    path = np.array([
        [0.0, 0.0, 0.0, 0, 0, 0],
        [0.1, 0.0, 0.0, 0, 0, 0],
        [0.2, 0.0, 0.0, 0, 0, 0],
    ])

    # Simulate (simplified)
    current_pose = np.array([0.0, 0.0, 0.01, 0, 0, 0])  # Slightly above surface
    current_force = np.array([0, 0, 3.0, 0, 0, 0])  # 3N contact force

    for target in path:
        output = controller.compute_surface_following(
            trajectory_point=target,
            current_pose=current_pose,
            current_force=current_force,
            desired_normal_force=5.0
        )
        print(f"Target X={target[0]:.2f}, Output: X={output[0]:.2f}, Z={output[2]:.4f}")
```

## Force Estimation Without Sensors

Not all robots have force/torque sensors. Several techniques can estimate external forces:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FORCE ESTIMATION METHODS                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   1. MOMENTUM OBSERVER                                                      │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │                                                                 │      │
│   │   τ_ext = K_O · (p - ∫(τ_cmd + τ_ext - C·q̇ - g)dt)             │      │
│   │                                                                 │      │
│   │   where p = M(q)·q̇ is generalized momentum                     │      │
│   │                                                                 │      │
│   │   ✓ Uses only joint position and torque                        │      │
│   │   ✗ Requires accurate dynamics model                           │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│   2. CURRENT-BASED ESTIMATION (for SEA/torque motors)                      │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │                                                                 │      │
│   │   τ_measured = K_t · I_motor                                    │      │
│   │   τ_ext = τ_measured - τ_model                                  │      │
│   │                                                                 │      │
│   │   ✓ Direct measurement from motor current                      │      │
│   │   ✓ No additional sensors needed                               │      │
│   │   ✗ Affected by friction, requires calibration                 │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│   3. DISTURBANCE OBSERVER                                                   │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │                                                                 │      │
│   │         ┌─────────┐    ┌─────────┐                             │      │
│   │   τ ──▶ │  Plant  │──▶ │ Q(s)/   │ ──▶ τ_ext_estimated        │      │
│   │         │  P(s)   │    │ (1-Q(s))│                             │      │
│   │         └─────────┘    └─────────┘                             │      │
│   │                                                                 │      │
│   │   Q(s) is a low-pass filter                                    │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

```python
"""
Force Estimation Without Sensors

These techniques estimate external forces using only joint measurements
and robot dynamics models. Essential for robots without F/T sensors.
"""

import numpy as np
from typing import Optional


class MomentumObserver:
    """
    Momentum-based external torque observer.

    Estimates external torques by comparing expected vs actual
    generalized momentum changes.

    Theory:
        Generalized momentum: p = M(q)·q̇
        Momentum dynamics: ṗ = τ + τ_ext - C·q̇ - g

        Observer: τ̂_ext = K_O·(p - p_estimated)

    Reference: De Luca et al., "Sensorless Robot Collision Detection
               and Hybrid Force/Motion Control"
    """

    def __init__(self, n_joints: int, observer_gain: float = 10.0, dt: float = 0.001):
        """
        Initialize momentum observer.

        Args:
            n_joints: Number of robot joints
            observer_gain: Observer bandwidth (higher = faster but noisier)
            dt: Control timestep
        """
        self.n = n_joints
        self.K_O = observer_gain * np.eye(n_joints)
        self.dt = dt

        # Observer state
        self.integral = np.zeros(n_joints)
        self.tau_ext_estimated = np.zeros(n_joints)

    def estimate(
        self,
        q: np.ndarray,
        q_dot: np.ndarray,
        tau_cmd: np.ndarray,
        robot_dynamics
    ) -> np.ndarray:
        """
        Estimate external torque.

        Args:
            q: Joint positions
            q_dot: Joint velocities
            tau_cmd: Commanded torques
            robot_dynamics: Object providing M(q), C(q,q̇), g(q)

        Returns:
            Estimated external torque
        """
        # Get dynamics matrices
        M = robot_dynamics.mass_matrix(q)
        C_qdot = robot_dynamics.coriolis_vector(q, q_dot)
        g = robot_dynamics.gravity_vector(q)

        # Current momentum
        p = M @ q_dot

        # Momentum residual (what we integrate)
        residual = tau_cmd - C_qdot - g + self.tau_ext_estimated

        # Update integral
        self.integral += residual * self.dt

        # Observer output
        self.tau_ext_estimated = self.K_O @ (p - self.integral)

        return self.tau_ext_estimated

    def reset(self):
        """Reset observer state."""
        self.integral = np.zeros(self.n)
        self.tau_ext_estimated = np.zeros(self.n)


class CurrentBasedForceEstimator:
    """
    Estimate joint torque from motor current.

    For robots with torque motors or series elastic actuators,
    motor current provides a direct measure of applied torque.
    """

    def __init__(
        self,
        n_joints: int,
        torque_constants: np.ndarray,
        gear_ratios: np.ndarray
    ):
        """
        Initialize current-based estimator.

        Args:
            n_joints: Number of joints
            torque_constants: Motor torque constant K_t [Nm/A] per joint
            gear_ratios: Gear reduction ratio per joint
        """
        self.n = n_joints
        self.K_t = torque_constants
        self.N = gear_ratios

    def estimate_joint_torque(self, motor_currents: np.ndarray) -> np.ndarray:
        """
        Estimate joint torques from motor currents.

        Args:
            motor_currents: Measured motor currents [A]

        Returns:
            Estimated joint torques [Nm]
        """
        # Motor torque = K_t * I
        # Joint torque = Motor torque * Gear ratio
        motor_torques = self.K_t * motor_currents
        joint_torques = motor_torques * self.N

        return joint_torques

    def estimate_external_torque(
        self,
        motor_currents: np.ndarray,
        expected_torque: np.ndarray
    ) -> np.ndarray:
        """
        Estimate external torque by comparing measured vs expected.

        Args:
            motor_currents: Measured motor currents
            expected_torque: Expected torque from dynamics model

        Returns:
            Estimated external torque
        """
        measured_torque = self.estimate_joint_torque(motor_currents)
        return measured_torque - expected_torque


class DisturbanceObserver:
    """
    First-order disturbance observer.

    Estimates disturbance torque using a low-pass filtered
    comparison of expected vs actual behavior.
    """

    def __init__(
        self,
        n_joints: int,
        cutoff_freq: float = 10.0,
        dt: float = 0.001
    ):
        """
        Initialize disturbance observer.

        Args:
            n_joints: Number of joints
            cutoff_freq: Observer bandwidth [Hz]
            dt: Control timestep
        """
        self.n = n_joints
        self.dt = dt

        # Low-pass filter coefficient
        tau = 1.0 / (2 * np.pi * cutoff_freq)
        self.alpha = dt / (tau + dt)

        # State
        self.d_hat = np.zeros(n_joints)  # Estimated disturbance

    def estimate(
        self,
        q_ddot_measured: np.ndarray,
        q_ddot_expected: np.ndarray,
        M: np.ndarray
    ) -> np.ndarray:
        """
        Estimate disturbance torque.

        Args:
            q_ddot_measured: Measured acceleration
            q_ddot_expected: Expected acceleration from model
            M: Mass matrix

        Returns:
            Estimated disturbance torque
        """
        # Acceleration error indicates disturbance
        accel_error = q_ddot_measured - q_ddot_expected

        # Disturbance in torque space
        d_measured = M @ accel_error

        # Low-pass filter
        self.d_hat = self.alpha * d_measured + (1 - self.alpha) * self.d_hat

        return self.d_hat

    def reset(self):
        """Reset observer state."""
        self.d_hat = np.zeros(self.n)
```

## Industry Perspective: Force Control in Practice

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 FORCE CONTROL IN INDUSTRY                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   COLLABORATIVE ROBOTS (Cobots)                                             │
│   ┌─────────────────────────────────────────────────────────────┐          │
│   │ • Universal Robots: Current-based force estimation          │          │
│   │ • KUKA iiwa: Joint torque sensors in every joint           │          │
│   │ • Franka Emika: Torque sensors + impedance control         │          │
│   │ • ABB YuMi: Dual-arm with force limiting                   │          │
│   │                                                             │          │
│   │ Key insight: Safety through force control!                  │          │
│   │ ISO 10218-1 limits: 150N transient, 80N quasi-static       │          │
│   └─────────────────────────────────────────────────────────────┘          │
│                                                                             │
│   ASSEMBLY APPLICATIONS                                                     │
│   ┌─────────────────────────────────────────────────────────────┐          │
│   │ Peg-in-hole insertion:                                      │          │
│   │   Position control → approach                               │          │
│   │   Force control → search & insertion                        │          │
│   │   Success requires <0.1mm position + <5N force control     │          │
│   │                                                             │          │
│   │ Gear meshing, connector insertion, snap-fit assembly       │          │
│   └─────────────────────────────────────────────────────────────┘          │
│                                                                             │
│   HUMANOID ROBOTS                                                           │
│   ┌─────────────────────────────────────────────────────────────┐          │
│   │ • Boston Dynamics Atlas: Hydraulic + force feedback        │          │
│   │ • Tesla Optimus: Joint torque sensing for dexterous tasks  │          │
│   │ • Figure: Whole-body impedance for human interaction       │          │
│   │                                                             │          │
│   │ Challenge: Coordinating force control across 30+ joints    │          │
│   └─────────────────────────────────────────────────────────────┘          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Comparison of Control Approaches

| Aspect | Position Control | Impedance Control | Admittance Control | Hybrid Control |
|--------|-----------------|-------------------|-------------------|----------------|
| **Input** | Desired position | Desired pose + impedance params | Measured force | Position + force targets |
| **Output** | Position command | Force/torque | Position/velocity | Mixed commands |
| **Best for** | Free space motion | Compliant interaction | Position-controlled robots | Surface contact tasks |
| **Requires** | Position sensors | Torque actuators | Force sensor | Both sensors |
| **Stability** | Easy | Passivity-based | Can be tricky | Careful design |
| **Example** | Pick & place | Human-robot collaboration | Hand guiding | Polishing |

## Summary

### Key Takeaways

1. **Force control enables interaction**: Position control alone cannot handle contact with uncertain environments safely

2. **Impedance vs Admittance**:
   - Impedance: Control apparent dynamics, output force
   - Admittance: Measure force, output motion
   - Choose based on robot's control interface

3. **Hybrid control combines both**: Control position where you need trajectory tracking, force where you need contact control

4. **No sensor? No problem**: Momentum observers and current sensing can estimate forces

5. **Safety through compliance**: Modern cobots achieve safety through force-sensitive control, not just padding

### Design Checklist

- [ ] Identified task requirements (free motion vs contact)
- [ ] Selected control approach (impedance/admittance/hybrid)
- [ ] Determined force sensing method (sensor or estimation)
- [ ] Tuned impedance parameters for task
- [ ] Verified stability (passivity, energy bounds)
- [ ] Tested with actual contact scenarios
- [ ] Validated safety compliance (force limits)

## Further Reading

### Foundational Papers
- Hogan, N. (1985). "Impedance Control: An Approach to Manipulation"
- Raibert, M. & Craig, J. (1981). "Hybrid Position/Force Control of Manipulators"

### Textbooks
- Siciliano, B. et al. "Robotics: Modelling, Planning and Control" - Chapter 9
- Spong, M. et al. "Robot Modeling and Control" - Chapter 10

### Modern Developments
- De Luca, A. "Collision Detection and Safe Reaction with the DLR-III Lightweight Robot Arm"
- Khatib, O. "Unified Approach for Motion and Force Control of Robot Manipulators"

---

*"In robotics, the ability to feel and respond to forces is what separates a tool from a collaborator."*
