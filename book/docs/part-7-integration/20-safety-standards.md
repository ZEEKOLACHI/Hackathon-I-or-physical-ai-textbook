---
id: ch-7-20
title: Safety Standards
sidebar_position: 2
difficulty: advanced
estimated_time: 35
prerequisites: [ch-7-19]
---

# Safety Standards

Safety is paramount in Physical AI systems that operate alongside humans. This chapter covers standards, design patterns, and implementation strategies for safe humanoid robots.

## Safety Standards Overview

### ISO 10218: Industrial Robots

```python
# Key safety requirements from ISO 10218
SAFETY_REQUIREMENTS = {
    'emergency_stop': {
        'category': 0,  # Immediate power removal
        'response_time_ms': 500,
        'redundancy': 'dual_channel'
    },
    'protective_stop': {
        'category': 1,  # Controlled stop then power off
        'response_time_ms': 1000
    },
    'speed_limits': {
        'reduced_speed_mode': 250,  # mm/s
        'collaborative_mode': 500    # mm/s for ISO 10218-2
    },
    'force_limits': {
        'quasi_static': 150,   # N
        'transient': 500       # N (based on body region)
    }
}
```

### ISO/TS 15066: Collaborative Robots

```python
class CollaborativeRobotSafety:
    """
    Implements ISO/TS 15066 biomechanical limits.

    Limits vary by body region contacted.
    """

    # Pressure and force limits by body region (simplified)
    BODY_REGION_LIMITS = {
        'skull': {'pressure_kPa': 175, 'force_N': 130},
        'face': {'pressure_kPa': 110, 'force_N': 65},
        'chest': {'pressure_kPa': 140, 'force_N': 140},
        'hand': {'pressure_kPa': 260, 'force_N': 190},
        'leg': {'pressure_kPa': 220, 'force_N': 210},
    }

    def __init__(self, robot_mass, max_velocity):
        self.robot_mass = robot_mass
        self.max_velocity = max_velocity

    def compute_max_safe_velocity(self, body_region, effective_mass):
        """
        Compute maximum velocity to stay within force limits.

        Uses energy-based method from ISO/TS 15066 Annex A.
        """
        limit = self.BODY_REGION_LIMITS[body_region]
        max_force = limit['force_N']

        # v_max = sqrt(2 * F_max / m_eff)
        # where m_eff is reduced mass of collision
        reduced_mass = (self.robot_mass * effective_mass) / \
                       (self.robot_mass + effective_mass)

        v_max = (2 * max_force / reduced_mass) ** 0.5
        return min(v_max, self.max_velocity)
```

## Safety-Rated Components

### Redundant Safety Controller

```python
class DualChannelSafetyController:
    """
    Dual-channel safety system with cross-monitoring.

    Both channels must agree for operation to continue.
    """

    def __init__(self):
        self.channel_a = SafetyChannel('A')
        self.channel_b = SafetyChannel('B')
        self.cross_check_interval = 0.01  # 100Hz

    def check_safety(self, sensor_data):
        result_a = self.channel_a.evaluate(sensor_data)
        result_b = self.channel_b.evaluate(sensor_data)

        # Cross-check between channels
        if result_a != result_b:
            self.log_discrepancy(result_a, result_b)
            return SafetyResult.STOP  # Fail-safe

        return result_a

    def verify_integrity(self):
        """Periodic self-test of safety functions."""
        # Test emergency stop circuit
        assert self.test_estop_circuit()

        # Test safety sensors
        for sensor in self.safety_sensors:
            assert sensor.self_test()

        # Test brake engagement
        assert self.test_brakes()

        return True


class SafetyChannel:
    def __init__(self, name):
        self.name = name
        self.monitors = []

    def evaluate(self, sensor_data):
        for monitor in self.monitors:
            if not monitor.is_safe(sensor_data):
                return SafetyResult.STOP
        return SafetyResult.OK
```

### Safe Motion Monitoring

```python
class SafetyRatedMonitor:
    """
    Implements safety-rated monitored stop and speed limiting.
    """

    def __init__(self, joint_limits, velocity_limits, force_limits):
        self.joint_limits = joint_limits
        self.velocity_limits = velocity_limits
        self.force_limits = force_limits

    def check_position(self, positions):
        """Safety-rated position monitoring."""
        for i, (pos, (min_pos, max_pos)) in enumerate(
            zip(positions, self.joint_limits)):

            margin = 0.05  # 5% safety margin
            soft_min = min_pos + margin * (max_pos - min_pos)
            soft_max = max_pos - margin * (max_pos - min_pos)

            if pos < soft_min or pos > soft_max:
                return SafetyViolation(
                    type='POSITION_LIMIT',
                    joint=i,
                    value=pos,
                    limit=(soft_min, soft_max)
                )
        return None

    def check_velocity(self, velocities):
        """Safety-rated speed monitoring."""
        for i, (vel, limit) in enumerate(
            zip(velocities, self.velocity_limits)):

            if abs(vel) > limit:
                return SafetyViolation(
                    type='VELOCITY_LIMIT',
                    joint=i,
                    value=vel,
                    limit=limit
                )
        return None

    def check_force(self, forces):
        """External force monitoring for collision detection."""
        for i, (force, limit) in enumerate(
            zip(forces, self.force_limits)):

            if abs(force) > limit:
                return SafetyViolation(
                    type='FORCE_LIMIT',
                    joint=i,
                    value=force,
                    limit=limit
                )
        return None
```

## Collision Detection and Response

```python
import numpy as np

class CollisionDetector:
    """
    Multi-modal collision detection system.
    """

    def __init__(self, robot_model, thresholds):
        self.robot = robot_model
        self.thresholds = thresholds
        self.momentum_observer = MomentumObserver(robot_model)

    def detect_collision(self, joint_state, external_forces):
        """
        Detect collisions using multiple methods.

        Returns collision info if detected, None otherwise.
        """
        # Method 1: Direct force/torque sensing
        if self.check_force_threshold(external_forces):
            return Collision(method='force_sensing',
                           magnitude=np.linalg.norm(external_forces))

        # Method 2: Momentum observer
        residual = self.momentum_observer.compute(joint_state)
        if np.linalg.norm(residual) > self.thresholds['momentum']:
            return Collision(method='momentum_observer',
                           residual=residual)

        # Method 3: Motor current monitoring
        current_error = self.check_current_residual(joint_state)
        if current_error > self.thresholds['current']:
            return Collision(method='current_monitoring',
                           error=current_error)

        return None

    def respond_to_collision(self, collision):
        """
        Execute collision response strategy.
        """
        if collision.magnitude > self.thresholds['hard_collision']:
            # Hard collision: immediate stop
            self.emergency_stop()
        else:
            # Soft collision: compliant response
            self.enable_compliance_mode()
            self.retract_from_contact()


class MomentumObserver:
    """
    Generalized momentum observer for collision detection.

    Based on De Luca et al. "Collision Detection and Safe Reaction"
    """

    def __init__(self, robot_model):
        self.robot = robot_model
        self.r = np.zeros(robot_model.n_joints)  # Residual
        self.K = np.eye(robot_model.n_joints) * 10  # Observer gain

    def compute(self, state):
        """Compute momentum residual."""
        q, qd, tau = state.position, state.velocity, state.torque

        # Generalized momentum
        M = self.robot.mass_matrix(q)
        p = M @ qd

        # Expected momentum derivative
        C = self.robot.coriolis_matrix(q, qd)
        g = self.robot.gravity_vector(q)
        p_dot_expected = tau - C.T @ qd - g

        # Update residual (filtered integral of discrepancy)
        self.r += self.K @ (p - self.r) * self.dt

        return self.r - p_dot_expected * self.dt
```

## Safe Human-Robot Interaction

```python
class HumanAwareController:
    """
    Adjusts robot behavior based on human proximity.
    """

    SAFETY_ZONES = {
        'danger': 0.3,      # meters - immediate stop
        'warning': 0.8,     # meters - reduced speed
        'caution': 1.5,     # meters - limited speed
        'normal': float('inf')
    }

    def __init__(self, base_controller):
        self.controller = base_controller
        self.human_tracker = HumanTracker()

    def compute_safe_command(self, state, command):
        """Scale command based on human proximity."""
        humans = self.human_tracker.get_tracked_humans()

        min_distance = float('inf')
        for human in humans:
            distance = self.compute_min_distance(state, human)
            min_distance = min(min_distance, distance)

        # Determine safety zone
        zone = self.get_zone(min_distance)

        # Apply speed scaling
        scale = self.get_speed_scale(zone)
        scaled_command = self.scale_velocities(command, scale)

        return scaled_command

    def get_speed_scale(self, zone):
        scales = {
            'danger': 0.0,
            'warning': 0.25,
            'caution': 0.5,
            'normal': 1.0
        }
        return scales[zone]
```

## Safety Validation

```python
class SafetyValidator:
    """
    Validates safety system implementation.
    """

    def validate_estop_response(self, robot):
        """Test emergency stop response time."""
        # Start motion
        robot.move_to_velocity([1.0] * robot.n_joints)

        # Trigger e-stop
        start_time = time.time()
        robot.emergency_stop()

        # Measure stop time
        while robot.is_moving():
            if time.time() - start_time > 0.5:  # 500ms limit
                return ValidationResult(
                    passed=False,
                    message="E-stop response exceeded 500ms"
                )

        stop_time = time.time() - start_time
        return ValidationResult(
            passed=True,
            response_time_ms=stop_time * 1000
        )

    def validate_force_limits(self, robot, test_surface):
        """Validate force limiting during contact."""
        results = []

        for body_region in ['chest', 'hand', 'leg']:
            limit = CollaborativeRobotSafety.BODY_REGION_LIMITS[body_region]

            # Move toward surface
            robot.move_toward(test_surface)

            # Measure contact force
            max_force = robot.get_max_contact_force()

            passed = max_force <= limit['force_N']
            results.append(ValidationResult(
                body_region=body_region,
                passed=passed,
                measured_force=max_force,
                limit=limit['force_N']
            ))

        return results
```

## Summary

- ISO 10218 defines industrial robot safety requirements
- ISO/TS 15066 provides collaborative robot biomechanical limits
- Dual-channel safety systems ensure fail-safe operation
- Collision detection uses force sensing and momentum observers
- Speed and force limiting protects humans in shared workspaces
- Safety validation verifies compliance with standards

## Further Reading

- ISO 10218-1:2011 "Robots and robotic devices - Safety requirements"
- ISO/TS 15066:2016 "Robots and robotic devices - Collaborative robots"
- Haddadin, S. "Towards Safe Robots"
- RIA TR R15.606 "Collaborative Robot Safety"
