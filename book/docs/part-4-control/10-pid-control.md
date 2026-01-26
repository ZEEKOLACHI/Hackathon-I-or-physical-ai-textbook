---
id: ch-4-10
title: PID Control
sidebar_position: 1
difficulty: beginner
estimated_time: 45
prerequisites: [ch-1-03]
---

# PID Control

> "The best controller is the simplest one that achieves the desired performance."
> — Karl Johan Åström, Control Theory Pioneer

PID (Proportional-Integral-Derivative) control stands as the most widely deployed control strategy in engineering history. From thermostats to spacecraft, from industrial robots to autonomous vehicles, PID controllers form the backbone of feedback control systems. In robotics, understanding PID control is essential—it's the foundation upon which more sophisticated control strategies are built.

## The Essence of Feedback Control

Before diving into PID, let's understand why feedback control matters. Consider a robot arm trying to reach a target position:

```
Without Feedback (Open-Loop):
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Command   │────▶│  Controller │────▶│    Robot    │────▶ Actual Position
│  (Target)   │     │             │     │    Arm      │         (Unknown)
└─────────────┘     └─────────────┘     └─────────────┘

Problem: No way to know if we reached the target!

With Feedback (Closed-Loop):
                    ┌─────────────────────────────────────────┐
                    │                                         │
                    ▼                                         │
┌─────────────┐   ┌───┐   ┌─────────────┐   ┌─────────────┐  │
│   Command   │──▶│ + │──▶│  Controller │──▶│    Robot    │──┼──▶ Actual
│  (Target)   │   │ - │   │    (PID)    │   │    Arm      │  │    Position
└─────────────┘   └───┘   └─────────────┘   └─────────────┘  │
                    ▲                                         │
                    │         ┌─────────────┐                 │
                    └─────────│   Sensor    │◀────────────────┘
                              │  (Encoder)  │
                              └─────────────┘

Solution: Continuous correction based on measured error!
```

## Historical Context: The Origins of PID

The PID controller has a fascinating history spanning over a century:

```
Timeline of PID Control Development:
────────────────────────────────────────────────────────────────────────────────

1788 │ James Watt's Centrifugal Governor
     │ First mechanical feedback controller for steam engines
     │
1922 │ Nicolas Minorsky
     │ First theoretical analysis of three-term control
     │ Developed for automatic ship steering
     │
1942 │ Ziegler-Nichols Tuning Rules
     │ First systematic method for PID tuning
     │ Still widely used today!
     │
1950s│ Pneumatic PID Controllers
     │ Industrial adoption in process control
     │
1970s│ Digital PID Implementation
     │ Microprocessors enable discrete-time PID
     │
Today│ Over 95% of industrial control loops use PID
     │ Foundation for modern robotics control
     │
────────────────────────────────────────────────────────────────────────────────
```

## The PID Equation

The PID controller combines three terms, each addressing a different aspect of control:

```
                          PID Control Law
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                    t                                                        │
│                   ⌠                              de(t)                      │
│   u(t) = Kp·e(t) + Ki·│ e(τ)dτ  +  Kd · ────                               │
│                   ⌡                               dt                        │
│                    0                                                        │
│                                                                             │
│   ├──────────┤   ├───────────────┤   ├────────────────┤                    │
│   Proportional      Integral           Derivative                          │
│   "Present"         "Past"             "Future"                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

Where:
  u(t) = Control output (e.g., motor torque)
  e(t) = Error = Setpoint - Measured value
  Kp   = Proportional gain
  Ki   = Integral gain
  Kd   = Derivative gain
```

### Understanding Each Term

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PROPORTIONAL TERM (P)                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Response ∝ Current Error                                                  │
│                                                                             │
│   Error ──────────────┬──────────────▶ Response                            │
│   Large               │               Strong                                │
│   ████████████████████│████████████████████████                            │
│                       │                                                     │
│   Small               │               Weak                                  │
│   ████                │               ████                                  │
│                       │                                                     │
│   ✓ Fast response to errors                                                │
│   ✗ Cannot eliminate steady-state error alone                              │
│   ✗ Too high → oscillations                                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                       INTEGRAL TERM (I)                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Response ∝ Accumulated Error Over Time                                    │
│                                                                             │
│   Error                   Integral                                          │
│     │                       │                                               │
│     │    ████               │              ████████████████████████        │
│     │   █████               │         █████████████████████████████        │
│     │  ██████               │    ██████████████████████████████████        │
│     │ ███████               │ █████████████████████████████████████        │
│     └────────▶ time         └────────────────────────────────▶ time        │
│                                                                             │
│   ✓ Eliminates steady-state error                                          │
│   ✓ Accounts for systematic biases (gravity, friction)                     │
│   ✗ Can cause overshoot                                                    │
│   ✗ Integral windup if saturated                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                      DERIVATIVE TERM (D)                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Response ∝ Rate of Change of Error                                        │
│                                                                             │
│   Error                   Derivative Response                               │
│     │                       │                                               │
│     │   /\                  │  ██                                           │
│     │  /  \                 │ ████                                          │
│     │ /    \____            │██████                      (dampens)          │
│     │/                      │        ████████████████                       │
│     └───────────▶ time      └────────────────────────▶ time                │
│                                                                             │
│   ✓ Predicts future error (anticipatory)                                   │
│   ✓ Dampens oscillations                                                   │
│   ✗ Amplifies measurement noise                                            │
│   ✗ Can cause instability with noisy signals                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## The PID Controller Implementation

Let's build a robust PID controller step by step:

```python
"""
PID Controller Implementation for Robotics

This module provides a complete PID controller with features essential
for real-world robotics applications including anti-windup, derivative
filtering, and output saturation.

Theory:
    The discrete-time PID control law:

    u[k] = Kp * e[k] + Ki * Σe[i]*dt + Kd * (e[k] - e[k-1]) / dt

    Where:
    - u[k] is the control output at time step k
    - e[k] is the error (setpoint - measurement)
    - dt is the sampling time
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class PIDGains:
    """Container for PID gains with validation."""
    kp: float  # Proportional gain
    ki: float  # Integral gain
    kd: float  # Derivative gain

    def __post_init__(self):
        """Validate gain values."""
        if self.kp < 0 or self.ki < 0 or self.kd < 0:
            raise ValueError("PID gains must be non-negative")

    def __str__(self) -> str:
        return f"Kp={self.kp:.3f}, Ki={self.ki:.3f}, Kd={self.kd:.3f}"


class PIDController:
    """
    A production-ready PID controller for robotics applications.

    Features:
    - Anti-windup protection
    - Derivative filtering (reduces noise sensitivity)
    - Output saturation
    - Setpoint weighting
    - Bumpless transfer for gain changes

    Example:
        >>> pid = PIDController(kp=1.0, ki=0.1, kd=0.05, dt=0.01)
        >>> pid.set_limits(output_min=-10, output_max=10)
        >>> control_signal = pid.compute(setpoint=5.0, measured=3.2)
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        dt: float = 0.01,
        derivative_filter_coeff: float = 0.1
    ):
        """
        Initialize PID controller.

        Args:
            kp: Proportional gain (response to current error)
            ki: Integral gain (response to accumulated error)
            kd: Derivative gain (response to error rate of change)
            dt: Sample time in seconds
            derivative_filter_coeff: Low-pass filter coefficient for
                                     derivative term (0-1, lower = more filtering)
        """
        self.gains = PIDGains(kp, ki, kd)
        self.dt = dt
        self.alpha = derivative_filter_coeff

        # State variables
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_derivative = 0.0

        # Limits
        self._output_min = -np.inf
        self._output_max = np.inf
        self._integral_min = -np.inf
        self._integral_max = np.inf

        # Setpoint weighting (reduces overshoot)
        self._setpoint_weight_p = 1.0  # Weight for proportional term
        self._setpoint_weight_d = 0.0  # Weight for derivative term

    def set_limits(
        self,
        output_min: float = -np.inf,
        output_max: float = np.inf,
        integral_min: Optional[float] = None,
        integral_max: Optional[float] = None
    ) -> None:
        """
        Set output and integral limits for anti-windup.

        Args:
            output_min: Minimum control output
            output_max: Maximum control output
            integral_min: Minimum integral term (defaults to output_min/Ki)
            integral_max: Maximum integral term (defaults to output_max/Ki)
        """
        self._output_min = output_min
        self._output_max = output_max

        # Set integral limits based on output limits if not specified
        if integral_min is None and self.gains.ki > 0:
            self._integral_min = output_min / self.gains.ki
        elif integral_min is not None:
            self._integral_min = integral_min

        if integral_max is None and self.gains.ki > 0:
            self._integral_max = output_max / self.gains.ki
        elif integral_max is not None:
            self._integral_max = integral_max

    def set_setpoint_weighting(self, weight_p: float = 1.0, weight_d: float = 0.0) -> None:
        """
        Set setpoint weighting to reduce overshoot.

        The weighted error for P term: e_p = weight_p * setpoint - measured
        The weighted error for D term: e_d = weight_d * setpoint - measured

        Args:
            weight_p: Weight for setpoint in P term (0-1)
            weight_d: Weight for setpoint in D term (0-1)
        """
        self._setpoint_weight_p = np.clip(weight_p, 0, 1)
        self._setpoint_weight_d = np.clip(weight_d, 0, 1)

    def compute(self, setpoint: float, measured: float) -> float:
        """
        Compute the PID control output.

        Args:
            setpoint: Desired value (target)
            measured: Current measured value (feedback)

        Returns:
            Control signal (e.g., motor command)
        """
        # Calculate errors with setpoint weighting
        error = setpoint - measured
        error_p = self._setpoint_weight_p * setpoint - measured
        error_d = self._setpoint_weight_d * setpoint - measured

        # Proportional term
        p_term = self.gains.kp * error_p

        # Integral term with anti-windup
        self._integral += error * self.dt
        self._integral = np.clip(
            self._integral,
            self._integral_min,
            self._integral_max
        )
        i_term = self.gains.ki * self._integral

        # Derivative term with filtering
        # Using filtered derivative to reduce noise sensitivity
        raw_derivative = (error_d - self._prev_error) / self.dt
        filtered_derivative = (
            self.alpha * raw_derivative +
            (1 - self.alpha) * self._prev_derivative
        )
        d_term = self.gains.kd * filtered_derivative

        # Store state for next iteration
        self._prev_error = error_d
        self._prev_derivative = filtered_derivative

        # Combine terms and apply output limits
        output = p_term + i_term + d_term
        output = np.clip(output, self._output_min, self._output_max)

        return output

    def reset(self) -> None:
        """Reset controller state (integral and derivative history)."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_derivative = 0.0

    def update_gains(self, kp: float, ki: float, kd: float) -> None:
        """
        Update PID gains with bumpless transfer.

        Adjusts integral term to prevent sudden output jumps
        when gains are changed during operation.
        """
        # Bumpless transfer: adjust integral to maintain output
        if self.gains.ki > 0 and ki > 0:
            self._integral = self._integral * self.gains.ki / ki

        self.gains = PIDGains(kp, ki, kd)

    def get_components(self) -> Tuple[float, float, float]:
        """
        Get the individual P, I, D components of the last computation.

        Useful for tuning and debugging.

        Returns:
            Tuple of (p_term, i_term, d_term)
        """
        p = self.gains.kp * self._prev_error
        i = self.gains.ki * self._integral
        d = self.gains.kd * self._prev_derivative
        return (p, i, d)


# Example usage demonstrating position control
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Simulate a simple second-order system (like a motor with inertia)
    class SimpleMotor:
        """Simulated DC motor with inertia and damping."""
        def __init__(self, inertia=1.0, damping=0.5):
            self.J = inertia
            self.b = damping
            self.position = 0.0
            self.velocity = 0.0

        def step(self, torque, dt):
            """Apply torque and update state."""
            acceleration = (torque - self.b * self.velocity) / self.J
            self.velocity += acceleration * dt
            self.position += self.velocity * dt
            return self.position

    # Create controller and motor
    dt = 0.01
    pid = PIDController(kp=10.0, ki=2.0, kd=1.0, dt=dt)
    pid.set_limits(output_min=-100, output_max=100)
    motor = SimpleMotor()

    # Simulate step response
    time = np.arange(0, 5, dt)
    setpoint = 1.0
    positions = []

    for t in time:
        control = pid.compute(setpoint, motor.position)
        motor.step(control, dt)
        positions.append(motor.position)

    print(f"Final position: {positions[-1]:.4f} (target: {setpoint})")
    print(f"Steady-state error: {abs(setpoint - positions[-1]):.6f}")
```

## Effect of Each Gain

Understanding how each gain affects system response is crucial for tuning:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EFFECT OF INCREASING Kp                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Response                                                                   │
│     │                    ┌───────────────── Setpoint                       │
│     │    High Kp ───────/│\                                                │
│     │                  /  │ \                                              │
│     │   Medium Kp ────/   │  \__________                                   │
│     │                /    │                                                │
│     │   Low Kp ─────/─────│──────────────                                  │
│     │             /       │                                                │
│     └────────────/────────┴────────────────▶ Time                          │
│                                                                             │
│  ↑ Faster rise time           ↑ More overshoot                             │
│  ↑ Reduced steady-state error ↑ Potential oscillation                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                    EFFECT OF INCREASING Ki                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Response                 With Ki                                           │
│     │                    ┌─────────────────── Setpoint                     │
│     │                  ┌─┤                                                  │
│     │   With Ki ──────/  │                                                 │
│     │                /   │                                                 │
│     │   No Ki ──────/────│── (steady-state error remains)                  │
│     │             /      │                                                 │
│     └────────────/───────┴────────────────▶ Time                           │
│                                                                             │
│  ↑ Eliminates steady-state error  ↑ Slower response                        │
│  ↑ Handles constant disturbances  ↑ Can cause overshoot                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                    EFFECT OF INCREASING Kd                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Response                                                                   │
│     │                    ┌───────────────── Setpoint                       │
│     │   No Kd (oscillates)  /\  /\                                         │
│     │                      /  \/  \____                                    │
│     │                     /                                                │
│     │   With Kd (damped)─/─────────────                                    │
│     │                   /                                                  │
│     └──────────────────/───────────────▶ Time                              │
│                                                                             │
│  ↑ Reduces overshoot          ↑ Can slow response                          │
│  ↑ Dampens oscillations       ↑ Amplifies noise                            │
│  ↑ Anticipates future error   ↑ Can cause instability                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Summary Table of Gain Effects

| Parameter | Rise Time | Overshoot | Settling Time | Steady-State Error | Stability |
|-----------|-----------|-----------|---------------|--------------------|-----------|
| ↑ Kp | Decreases | Increases | Small change | Decreases | Degrades |
| ↑ Ki | Decreases | Increases | Increases | Eliminates | Degrades |
| ↑ Kd | Minor change | Decreases | Decreases | No effect | Improves (if Kd small) |

## Tuning Methods

### Ziegler-Nichols Method

The Ziegler-Nichols method is a classic approach developed in 1942:

```python
"""
Ziegler-Nichols Tuning Methods

Two approaches:
1. Step Response Method - for systems without oscillation
2. Ultimate Gain Method - for systems that can oscillate safely

Historical note: Developed by John G. Ziegler and Nathaniel B. Nichols
at Taylor Instruments in 1942. Still widely used today!
"""

from dataclasses import dataclass
from typing import Optional, List
import numpy as np


@dataclass
class ZNGains:
    """Container for Ziegler-Nichols tuned gains."""
    kp: float
    ki: float
    kd: float
    method: str

    def to_pid_time_form(self) -> tuple:
        """Convert to Ti (integral time) and Td (derivative time) form."""
        ti = self.kp / self.ki if self.ki > 0 else np.inf
        td = self.kd / self.kp if self.kp > 0 else 0
        return self.kp, ti, td


def ziegler_nichols_ultimate_gain(ku: float, tu: float, controller_type: str = "PID") -> ZNGains:
    """
    Calculate PID gains using Ziegler-Nichols Ultimate Gain method.

    Procedure:
    1. Set Ki = 0 and Kd = 0
    2. Increase Kp until system oscillates with constant amplitude
    3. Record Ku (ultimate gain) and Tu (oscillation period)
    4. Calculate gains using the formulas below

    Args:
        ku: Ultimate gain (gain at which sustained oscillation occurs)
        tu: Ultimate period (period of oscillation in seconds)
        controller_type: "P", "PI", or "PID"

    Returns:
        ZNGains with calculated gains

    Example:
        >>> gains = ziegler_nichols_ultimate_gain(ku=5.0, tu=2.0)
        >>> print(f"Kp={gains.kp:.2f}, Ki={gains.ki:.2f}, Kd={gains.kd:.2f}")
    """
    """
    Ziegler-Nichols Ultimate Gain Tuning Table:

    ┌─────────────┬─────────┬─────────┬─────────┐
    │ Controller  │   Kp    │   Ti    │   Td    │
    ├─────────────┼─────────┼─────────┼─────────┤
    │ P           │ 0.5*Ku  │    -    │    -    │
    │ PI          │ 0.45*Ku │ Tu/1.2  │    -    │
    │ PID         │ 0.6*Ku  │ Tu/2    │ Tu/8    │
    │ Pessen      │ 0.7*Ku  │ Tu/2.5  │ 3Tu/20  │
    │ Some OS     │ 0.33*Ku │ Tu/2    │ Tu/3    │
    │ No OS       │ 0.2*Ku  │ Tu/2    │ Tu/3    │
    └─────────────┴─────────┴─────────┴─────────┘
    """

    if controller_type == "P":
        kp = 0.5 * ku
        ki = 0.0
        kd = 0.0
    elif controller_type == "PI":
        kp = 0.45 * ku
        ti = tu / 1.2
        ki = kp / ti
        kd = 0.0
    elif controller_type == "PID":
        kp = 0.6 * ku
        ti = tu / 2.0
        td = tu / 8.0
        ki = kp / ti
        kd = kp * td
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")

    return ZNGains(kp=kp, ki=ki, kd=kd, method="Ultimate Gain")


def ziegler_nichols_step_response(k: float, l: float, t: float, controller_type: str = "PID") -> ZNGains:
    """
    Calculate PID gains using Ziegler-Nichols Step Response method.

    This method is used when the system shows an S-shaped step response
    (first-order plus dead time approximation).

    Procedure:
    1. Apply a step input to the open-loop system
    2. Measure the response curve
    3. Draw a tangent line at the inflection point
    4. Determine K (steady-state gain), L (delay), T (time constant)

    Args:
        k: Process gain (steady-state change / input change)
        l: Apparent dead time (delay before response starts)
        t: Time constant (time from inflection to 63% of final value)
        controller_type: "P", "PI", or "PID"

    Returns:
        ZNGains with calculated gains

    Response Curve Analysis:

        Output
          │
          │                    ┌─────────── Final Value
          │               ┌────┘
          │          ┌───/│
          │         /│  / │
          │        / │ /  │
          │       /  │/   │
          │      /   │    │
          │─────/────│    │
          │    │     │    │
          └────┼─────┼────┼────────▶ Time
               L     T
               │◀───▶│
               Delay  Time Constant
    """

    if controller_type == "P":
        kp = t / (k * l)
        ki = 0.0
        kd = 0.0
    elif controller_type == "PI":
        kp = 0.9 * t / (k * l)
        ti = l / 0.3
        ki = kp / ti
        kd = 0.0
    elif controller_type == "PID":
        kp = 1.2 * t / (k * l)
        ti = 2.0 * l
        td = 0.5 * l
        ki = kp / ti
        kd = kp * td
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")

    return ZNGains(kp=kp, ki=ki, kd=kd, method="Step Response")


class AutoTuner:
    """
    Automatic PID tuner using relay feedback method.

    The relay feedback method provides a safer alternative to
    Ziegler-Nichols by using a relay (on/off) controller to
    induce controlled oscillations.

    Theory:
        A relay controller causes limit cycle oscillations.
        The amplitude (a) and period (Tu) of these oscillations
        can be used to estimate Ku:

        Ku = 4d / (π * a)

        Where d is the relay amplitude and a is the output amplitude.
    """

    def __init__(self, relay_amplitude: float = 1.0, hysteresis: float = 0.0):
        """
        Initialize auto-tuner.

        Args:
            relay_amplitude: Amplitude of relay output (d)
            hysteresis: Hysteresis band to prevent chattering
        """
        self.d = relay_amplitude
        self.hysteresis = hysteresis
        self.measurements: List[tuple] = []
        self.relay_state = False

    def relay_output(self, error: float) -> float:
        """
        Compute relay output with hysteresis.

        Args:
            error: Current error (setpoint - measured)

        Returns:
            Relay output (+d or -d)
        """
        if error > self.hysteresis:
            self.relay_state = True
        elif error < -self.hysteresis:
            self.relay_state = False

        return self.d if self.relay_state else -self.d

    def record_measurement(self, time: float, output: float) -> None:
        """Record time and output for analysis."""
        self.measurements.append((time, output))

    def analyze(self) -> Optional[ZNGains]:
        """
        Analyze recorded measurements to find Ku and Tu.

        Returns:
            ZNGains if successful, None if insufficient data
        """
        if len(self.measurements) < 10:
            return None

        # Convert to numpy arrays
        times = np.array([m[0] for m in self.measurements])
        outputs = np.array([m[1] for m in self.measurements])

        # Find zero crossings
        zero_crossings = []
        for i in range(1, len(outputs)):
            if outputs[i-1] * outputs[i] < 0:
                # Linear interpolation for more accurate crossing time
                t_cross = times[i-1] + (times[i] - times[i-1]) * (
                    -outputs[i-1] / (outputs[i] - outputs[i-1])
                )
                zero_crossings.append(t_cross)

        if len(zero_crossings) < 4:
            return None

        # Calculate period from zero crossings (half periods)
        half_periods = np.diff(zero_crossings)
        tu = 2 * np.mean(half_periods)

        # Calculate amplitude
        a = (np.max(outputs) - np.min(outputs)) / 2

        # Calculate ultimate gain
        ku = 4 * self.d / (np.pi * a)

        return ziegler_nichols_ultimate_gain(ku, tu)
```

### Cohen-Coon Method

The Cohen-Coon method offers improved performance for processes with significant dead time:

```python
def cohen_coon_tuning(k: float, tau: float, theta: float) -> ZNGains:
    """
    Cohen-Coon tuning method for first-order plus dead time systems.

    Provides better performance than Z-N for systems with
    significant dead time (theta/tau > 0.3).

    Args:
        k: Process gain
        tau: Time constant
        theta: Dead time (delay)

    Returns:
        ZNGains with calculated gains

    System Model: G(s) = k * exp(-theta*s) / (tau*s + 1)
    """
    r = theta / tau  # Dead time ratio

    # Cohen-Coon PID formulas
    kp = (1.35 / k) * (tau / theta) * (1 + 0.18 * r / (1 - r))
    ti = theta * (2.5 - 2 * r) / (1 - 0.39 * r)
    td = 0.37 * theta * (1 - r) / (1 - 0.81 * r)

    ki = kp / ti
    kd = kp * td

    return ZNGains(kp=kp, ki=ki, kd=kd, method="Cohen-Coon")
```

### Comparison of Tuning Methods

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| Ziegler-Nichols (Step) | First-order + dead time | Easy, no oscillation needed | Can be aggressive |
| Ziegler-Nichols (Ultimate) | Systems that can oscillate | Accurate Ku and Tu | Requires oscillation |
| Cohen-Coon | High dead time systems | Better for delays | More complex |
| Relay Feedback | Production systems | Safe, automated | Requires software |
| IMC/Lambda | Conservative tuning | Smooth response | May be slow |

## Anti-Windup Strategies

Integral windup occurs when the controller output saturates but the integral term continues to accumulate:

```python
"""
Anti-Windup Strategies for PID Controllers

Integral windup occurs when:
1. The control output is saturated (at limits)
2. The error persists (can't be corrected due to saturation)
3. The integral term continues to grow

This causes:
- Large overshoot when the system finally responds
- Slow recovery from saturation
- Oscillations

Solutions:
1. Integral clamping - simple but can be slow
2. Back-calculation - better performance
3. Conditional integration - prevents windup entirely
"""

import numpy as np


class PIDWithClampingAntiWindup:
    """
    Anti-windup using integral clamping.

    Simple approach: limit the integral term to bounds.

    ┌─────────────────────────────────────────────┐
    │           Clamping Anti-Windup              │
    │                                             │
    │   integral = clamp(integral, min, max)      │
    │                                             │
    │   ✓ Simple to implement                     │
    │   ✗ Integral still frozen at limit          │
    │   ✗ Can be slow to recover                  │
    └─────────────────────────────────────────────┘
    """

    def __init__(self, kp: float, ki: float, kd: float, dt: float = 0.01,
                 integral_limit: float = 100.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integral_limit = integral_limit

        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, setpoint: float, measured: float) -> float:
        error = setpoint - measured

        # Proportional
        p_term = self.kp * error

        # Integral with clamping
        self.integral += error * self.dt
        self.integral = np.clip(
            self.integral,
            -self.integral_limit,
            self.integral_limit
        )
        i_term = self.ki * self.integral

        # Derivative
        derivative = (error - self.prev_error) / self.dt
        d_term = self.kd * derivative
        self.prev_error = error

        return p_term + i_term + d_term


class PIDWithBackCalculation:
    """
    Anti-windup using back-calculation.

    When output saturates, adjust the integral term based
    on the difference between desired and actual output.

    ┌─────────────────────────────────────────────────────────┐
    │              Back-Calculation Anti-Windup               │
    │                                                         │
    │   u_unsat = P + I + D                                   │
    │   u_sat = clamp(u_unsat, min, max)                      │
    │   integral += (Ki * e + Kb * (u_sat - u_unsat)) * dt    │
    │                                                         │
    │   where Kb is back-calculation gain (typically 1/Ti)    │
    │                                                         │
    │   ✓ Faster recovery from saturation                     │
    │   ✓ Better transient response                           │
    │   ✗ Requires additional tuning (Kb)                     │
    └─────────────────────────────────────────────────────────┘
    """

    def __init__(self, kp: float, ki: float, kd: float, dt: float = 0.01,
                 output_min: float = -100, output_max: float = 100,
                 kb: float = None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.output_min = output_min
        self.output_max = output_max

        # Back-calculation gain (default to Ki if not specified)
        self.kb = kb if kb is not None else ki

        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, setpoint: float, measured: float) -> float:
        error = setpoint - measured

        # Proportional
        p_term = self.kp * error

        # Integral (accumulated from previous iteration)
        i_term = self.ki * self.integral

        # Derivative
        derivative = (error - self.prev_error) / self.dt
        d_term = self.kd * derivative
        self.prev_error = error

        # Compute unsaturated output
        u_unsat = p_term + i_term + d_term

        # Saturate output
        u_sat = np.clip(u_unsat, self.output_min, self.output_max)

        # Back-calculation: adjust integral based on saturation
        self.integral += (error + self.kb * (u_sat - u_unsat)) * self.dt

        return u_sat


class PIDWithConditionalIntegration:
    """
    Anti-windup using conditional integration.

    Only integrate when certain conditions are met:
    - Error is small enough (within band)
    - Output is not saturated
    - Error and control have same sign

    ┌─────────────────────────────────────────────────────────┐
    │           Conditional Integration Anti-Windup           │
    │                                                         │
    │   if |error| < threshold and not saturated:             │
    │       integral += error * dt                            │
    │                                                         │
    │   ✓ Prevents windup entirely                            │
    │   ✓ Simple logic                                        │
    │   ✗ May not eliminate steady-state error during         │
    │     saturation                                          │
    └─────────────────────────────────────────────────────────┘
    """

    def __init__(self, kp: float, ki: float, kd: float, dt: float = 0.01,
                 output_min: float = -100, output_max: float = 100,
                 error_threshold: float = 10.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.output_min = output_min
        self.output_max = output_max
        self.error_threshold = error_threshold

        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_output = 0.0

    def compute(self, setpoint: float, measured: float) -> float:
        error = setpoint - measured

        # Proportional
        p_term = self.kp * error

        # Conditional integration
        output_saturated = (
            self.prev_output >= self.output_max or
            self.prev_output <= self.output_min
        )
        error_small = abs(error) < self.error_threshold
        same_sign = error * self.prev_output > 0

        # Only integrate if not saturated and error is manageable
        if not (output_saturated and same_sign):
            self.integral += error * self.dt

        i_term = self.ki * self.integral

        # Derivative
        derivative = (error - self.prev_error) / self.dt
        d_term = self.kd * derivative
        self.prev_error = error

        # Compute and saturate output
        output = p_term + i_term + d_term
        output = np.clip(output, self.output_min, self.output_max)
        self.prev_output = output

        return output
```

## ROS 2 Integration

A complete ROS 2 PID controller node for robot velocity control:

```python
#!/usr/bin/env python3
"""
ROS 2 PID Velocity Controller

This node implements a PID-based velocity controller for mobile robots.
It subscribes to odometry for feedback and publishes velocity commands.

Features:
- Configurable gains via ROS parameters
- Real-time gain updates via dynamic reconfigure
- Diagnostic information publishing
- Anti-windup protection

Topics:
    Subscriptions:
        /odom (nav_msgs/Odometry): Robot odometry for feedback
        /cmd_vel_target (geometry_msgs/Twist): Target velocity setpoint

    Publications:
        /cmd_vel (geometry_msgs/Twist): Computed velocity commands
        /pid_debug (custom_msgs/PIDDebug): Debug information

Parameters:
    kp_linear, ki_linear, kd_linear: Linear velocity gains
    kp_angular, ki_angular, kd_angular: Angular velocity gains
    control_rate: Control loop frequency (Hz)
    max_linear_velocity: Maximum linear velocity (m/s)
    max_angular_velocity: Maximum angular velocity (rad/s)
"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
import numpy as np


class PIDController:
    """Reusable PID controller with anti-windup."""

    def __init__(self, kp: float, ki: float, kd: float,
                 output_limit: float = float('inf'),
                 integral_limit: float = float('inf')):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.integral_limit = integral_limit

        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None

    def compute(self, setpoint: float, measured: float, current_time: float) -> float:
        """Compute PID output with variable time step."""

        # Calculate dt
        if self.prev_time is None:
            dt = 0.01  # Default for first iteration
        else:
            dt = current_time - self.prev_time
            dt = max(0.001, min(dt, 0.1))  # Bound dt for safety

        self.prev_time = current_time

        # Calculate error
        error = setpoint - measured

        # Proportional term
        p_term = self.kp * error

        # Integral term with anti-windup
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        i_term = self.ki * self.integral

        # Derivative term
        if dt > 0:
            derivative = (error - self.prev_error) / dt
        else:
            derivative = 0.0
        d_term = self.kd * derivative

        self.prev_error = error

        # Compute output with saturation
        output = p_term + i_term + d_term
        output = np.clip(output, -self.output_limit, self.output_limit)

        return output

    def reset(self):
        """Reset controller state."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None

    def update_gains(self, kp: float, ki: float, kd: float):
        """Update gains with bumpless transfer."""
        if self.ki > 0 and ki > 0:
            self.integral = self.integral * self.ki / ki
        self.kp = kp
        self.ki = ki
        self.kd = kd


class VelocityControllerNode(Node):
    """
    ROS 2 node for PID-based velocity control.

    Block Diagram:

    ┌──────────────┐     ┌─────────┐     ┌───────────┐     ┌─────────┐
    │ /cmd_vel_    │────▶│   PID   │────▶│  Velocity │────▶│  Robot  │
    │    target    │     │Controller│     │   Limit   │     │         │
    └──────────────┘     └─────────┘     └───────────┘     └────┬────┘
                              ▲                                   │
                              │         ┌───────────┐            │
                              └─────────│   /odom   │◀───────────┘
                                        └───────────┘
    """

    def __init__(self):
        super().__init__('velocity_controller')

        # Declare parameters with defaults
        self.declare_parameters(
            namespace='',
            parameters=[
                ('kp_linear', 1.0),
                ('ki_linear', 0.1),
                ('kd_linear', 0.05),
                ('kp_angular', 2.0),
                ('ki_angular', 0.2),
                ('kd_angular', 0.1),
                ('control_rate', 50.0),
                ('max_linear_velocity', 1.0),
                ('max_angular_velocity', 2.0),
            ]
        )

        # Get initial parameters
        self._load_parameters()

        # Initialize PID controllers
        self.linear_pid = PIDController(
            self.kp_linear, self.ki_linear, self.kd_linear,
            output_limit=self.max_linear_vel,
            integral_limit=self.max_linear_vel / self.ki_linear if self.ki_linear > 0 else 100
        )

        self.angular_pid = PIDController(
            self.kp_angular, self.ki_angular, self.kd_angular,
            output_limit=self.max_angular_vel,
            integral_limit=self.max_angular_vel / self.ki_angular if self.ki_angular > 0 else 100
        )

        # State variables
        self.target_linear = 0.0
        self.target_angular = 0.0
        self.current_linear = 0.0
        self.current_angular = 0.0

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.debug_pub = self.create_publisher(Float64MultiArray, '/pid_debug', 10)

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.target_sub = self.create_subscription(
            Twist, '/cmd_vel_target', self.target_callback, 10)

        # Control loop timer
        control_period = 1.0 / self.control_rate
        self.control_timer = self.create_timer(control_period, self.control_loop)

        # Parameter callback for dynamic reconfigure
        self.add_on_set_parameters_callback(self.parameter_callback)

        self.get_logger().info('Velocity controller initialized')
        self.get_logger().info(f'Linear PID: Kp={self.kp_linear}, Ki={self.ki_linear}, Kd={self.kd_linear}')
        self.get_logger().info(f'Angular PID: Kp={self.kp_angular}, Ki={self.ki_angular}, Kd={self.kd_angular}')

    def _load_parameters(self):
        """Load parameters from ROS parameter server."""
        self.kp_linear = self.get_parameter('kp_linear').value
        self.ki_linear = self.get_parameter('ki_linear').value
        self.kd_linear = self.get_parameter('kd_linear').value
        self.kp_angular = self.get_parameter('kp_angular').value
        self.ki_angular = self.get_parameter('ki_angular').value
        self.kd_angular = self.get_parameter('kd_angular').value
        self.control_rate = self.get_parameter('control_rate').value
        self.max_linear_vel = self.get_parameter('max_linear_velocity').value
        self.max_angular_vel = self.get_parameter('max_angular_velocity').value

    def parameter_callback(self, params) -> SetParametersResult:
        """Handle dynamic parameter updates."""
        for param in params:
            if param.name == 'kp_linear':
                self.kp_linear = param.value
            elif param.name == 'ki_linear':
                self.ki_linear = param.value
            elif param.name == 'kd_linear':
                self.kd_linear = param.value
            elif param.name == 'kp_angular':
                self.kp_angular = param.value
            elif param.name == 'ki_angular':
                self.ki_angular = param.value
            elif param.name == 'kd_angular':
                self.kd_angular = param.value

        # Update PID controllers
        self.linear_pid.update_gains(self.kp_linear, self.ki_linear, self.kd_linear)
        self.angular_pid.update_gains(self.kp_angular, self.ki_angular, self.kd_angular)

        self.get_logger().info('PID gains updated')
        return SetParametersResult(successful=True)

    def odom_callback(self, msg: Odometry):
        """Process odometry feedback."""
        self.current_linear = msg.twist.twist.linear.x
        self.current_angular = msg.twist.twist.angular.z

    def target_callback(self, msg: Twist):
        """Process velocity setpoint."""
        self.target_linear = msg.linear.x
        self.target_angular = msg.angular.z

    def control_loop(self):
        """Main control loop - compute and publish velocity commands."""
        current_time = self.get_clock().now().nanoseconds / 1e9

        # Compute PID outputs
        linear_cmd = self.linear_pid.compute(
            self.target_linear, self.current_linear, current_time)
        angular_cmd = self.angular_pid.compute(
            self.target_angular, self.current_angular, current_time)

        # Create and publish command
        cmd = Twist()
        cmd.linear.x = linear_cmd
        cmd.angular.z = angular_cmd
        self.cmd_pub.publish(cmd)

        # Publish debug information
        debug_msg = Float64MultiArray()
        debug_msg.data = [
            self.target_linear, self.current_linear, linear_cmd,
            self.target_angular, self.current_angular, angular_cmd,
            self.linear_pid.integral, self.angular_pid.integral
        ]
        self.debug_pub.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    node = VelocityControllerNode()

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

## Practical Considerations

### Derivative Kick and Filtering

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       DERIVATIVE KICK PROBLEM                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   When setpoint changes suddenly (step change):                             │
│                                                                             │
│   Setpoint                  Derivative of Error                             │
│      │                         │                                            │
│      │     ┌────────           │    ████                                    │
│      │     │                   │    ████  (Spike!)                          │
│      │─────┘                   │    ████                                    │
│      └──────────▶ t            └────────────▶ t                             │
│                                                                             │
│   Solution: Differentiate measurement instead of error                      │
│                                                                             │
│   d_term = -Kd * d(measured)/dt   instead of   Kd * d(error)/dt            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

```python
class PIDWithDerivativeOnMeasurement:
    """
    PID controller that avoids derivative kick by differentiating
    the measurement instead of the error.
    """

    def __init__(self, kp: float, ki: float, kd: float, dt: float = 0.01):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt

        self.integral = 0.0
        self.prev_measured = None

    def compute(self, setpoint: float, measured: float) -> float:
        error = setpoint - measured

        # Proportional
        p_term = self.kp * error

        # Integral
        self.integral += error * self.dt
        i_term = self.ki * self.integral

        # Derivative on measurement (not error!)
        if self.prev_measured is None:
            d_term = 0.0
        else:
            # Negative sign because we're differentiating measurement
            d_term = -self.kd * (measured - self.prev_measured) / self.dt

        self.prev_measured = measured

        return p_term + i_term + d_term
```

### Low-Pass Filter for Derivative

```python
class DerivativeFilter:
    """
    Low-pass filter for derivative term to reduce noise sensitivity.

    Transfer function: D(s) = Kd * s / (1 + tau_f * s)

    This limits high-frequency gain while maintaining low-frequency behavior.
    """

    def __init__(self, kd: float, tau_f: float, dt: float):
        """
        Args:
            kd: Derivative gain
            tau_f: Filter time constant (smaller = less filtering)
            dt: Sample time
        """
        self.kd = kd
        self.tau_f = tau_f
        self.dt = dt

        self.prev_derivative = 0.0
        self.prev_error = 0.0

    def compute(self, error: float) -> float:
        """Compute filtered derivative term."""
        # Tustin (bilinear) discretization
        alpha = 2 * self.tau_f / self.dt

        # Filtered derivative
        derivative = (
            (2 * self.kd * (error - self.prev_error) +
             (alpha - 1) * self.prev_derivative) /
            (alpha + 1)
        )

        self.prev_error = error
        self.prev_derivative = derivative

        return derivative
```

## Industry Perspective: PID in Modern Robotics

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   PID IN INDUSTRIAL ROBOTICS                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Modern Industrial Robot Controllers:                                      │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────┐          │
│   │             Cascaded Control Architecture                    │          │
│   │                                                              │          │
│   │   Position      ┌────┐    Velocity     ┌────┐    Torque     │          │
│   │   Setpoint ────▶│PID │───▶Setpoint ───▶│PID │───▶Command    │          │
│   │            ▲    └────┘           ▲     └────┘               │          │
│   │            │                     │                          │          │
│   │     Position                  Velocity                       │          │
│   │     Feedback                  Feedback                       │          │
│   └─────────────────────────────────────────────────────────────┘          │
│                                                                             │
│   Companies using PID-based control:                                        │
│   • FANUC, KUKA, ABB - Industrial arms (cascaded PID)                      │
│   • Boston Dynamics - Joint-level PID with impedance overlay               │
│   • Tesla Bot - Model-based control with PID for tracking                  │
│   • Universal Robots - Adaptive PID with gravity compensation              │
│                                                                             │
│   Key insight: PID is rarely used alone in advanced robots.                │
│   It's typically combined with:                                            │
│   • Feedforward (model-based prediction)                                   │
│   • Impedance control (for contact)                                        │
│   • Learning-based adaptation                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Common Pitfalls and Solutions

| Pitfall | Symptom | Solution |
|---------|---------|----------|
| Integral windup | Large overshoot after saturation | Add anti-windup (clamping or back-calculation) |
| Derivative kick | Spike when setpoint changes | Differentiate measurement, not error |
| Noisy derivative | Jerky control output | Add low-pass filter to derivative |
| High-frequency oscillation | Buzzing/chattering | Reduce Kd, add derivative filter |
| Slow settling | Takes too long to reach setpoint | Increase Kp or Ki (carefully) |
| Steady-state error | Never reaches setpoint | Add or increase Ki |
| Sample time jitter | Inconsistent response | Use real dt in calculations |

## Summary

### Key Takeaways

1. **PID is foundational**: Over 95% of control loops use PID or PID variants
2. **Each term has a purpose**: P for present error, I for past (accumulated) error, D for future (predicted) error
3. **Tuning matters**: Ziegler-Nichols provides starting points, but real systems need fine-tuning
4. **Anti-windup is essential**: Always implement integral limits for systems with saturation
5. **Derivative needs care**: Filter the derivative term and consider using derivative-on-measurement
6. **Context determines complexity**: Simple PID works for many cases; advanced robotics combines PID with feedforward and model-based methods

### When to Use PID

- **Good fit**: Velocity control, temperature regulation, simple position control
- **With modifications**: Joint control (add feedforward), mobile robot navigation
- **Consider alternatives**: Highly nonlinear systems, multi-input systems, systems requiring optimal performance

### Design Checklist

- [ ] Identified system type (first-order, second-order, with delay?)
- [ ] Selected appropriate tuning method
- [ ] Implemented anti-windup protection
- [ ] Added derivative filtering for noisy measurements
- [ ] Set appropriate output limits
- [ ] Tested with step response and disturbance rejection
- [ ] Validated stability margins

## Further Reading

### Textbooks
- Åström, K.J. & Murray, R.M. "Feedback Systems: An Introduction for Scientists and Engineers" (Free online)
- Franklin, G.F., Powell, J.D., & Emami-Naeini, A. "Feedback Control of Dynamic Systems"
- Ogata, K. "Modern Control Engineering"

### Papers
- Ziegler, J.G. & Nichols, N.B. (1942). "Optimum settings for automatic controllers"
- Åström, K.J. & Hägglund, T. (1984). "Automatic tuning of simple regulators with specifications on phase and amplitude margins"

### Online Resources
- [PID Control on Wikipedia](https://en.wikipedia.org/wiki/PID_controller)
- [ROS 2 Control Documentation](https://control.ros.org/)
- [MATLAB PID Tuner](https://www.mathworks.com/help/control/ref/pidtuner-app.html)

---

*"PID controllers are like democracy - the worst form of control except for all the others that have been tried."*
— Adapted from Winston Churchill
