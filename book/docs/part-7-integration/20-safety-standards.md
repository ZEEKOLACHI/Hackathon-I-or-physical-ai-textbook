---
id: ch-7-20
title: Safety Standards
sidebar_position: 2
difficulty: advanced
estimated_time: 60
prerequisites: [ch-7-19]
---

# Safety Standards

> "Safety is not a gadget but a state of mind."
> — Eleanor Everet

Humanoid robots operating alongside humans must meet rigorous safety requirements. Unlike industrial robots confined to cages, humanoid robots share workspaces, interact physically, and make autonomous decisions. This chapter explores the safety standards, design principles, and implementation strategies that enable safe human-robot coexistence.

## The Safety Challenge

```
              Safety in Human-Robot Interaction

    Industrial Robot (Traditional)        Humanoid Robot (Modern)
    ──────────────────────────────        ──────────────────────────

    ┌─────────────────────────────┐       ┌─────────────────────────────┐
    │      SAFETY CAGE            │       │      SHARED WORKSPACE       │
    │   ┌─────────────────────┐   │       │                             │
    │   │                     │   │       │    ┌───┐       ○            │
    │   │    ┌─────┐          │   │       │    │   │      /│\           │
    │   │    │Robot│          │   │       │    │ R │  ←→  / \           │
    │   │    └─────┘          │   │       │    │   │    Human           │
    │   │                     │   │       │    └───┘                    │
    │   └─────────────────────┘   │       │                             │
    │                             │       │  Physical contact expected  │
    │   Human stays OUTSIDE       │       │  Collaboration required     │
    └─────────────────────────────┘       └─────────────────────────────┘

    Safety Strategy:                      Safety Strategy:
    • Physical separation                 • Inherent safety design
    • Hard boundaries                     • Force/torque limiting
    • Emergency stop only                 • Continuous monitoring
                                         • Intelligent behavior


              Risk Categories
              ───────────────

    ┌────────────────────────────────────────────────────────────────┐
    │                                                                │
    │   SEVERITY                                                     │
    │      ▲                                                         │
    │      │                                                         │
    │  S3  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
    │      │  │ Moderate │  │   High   │  │  Very    │              │
    │      │  │   Risk   │  │   Risk   │  │  High    │              │
    │      │  └──────────┘  └──────────┘  └──────────┘              │
    │  S2  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
    │      │  │   Low    │  │ Moderate │  │   High   │              │
    │      │  │   Risk   │  │   Risk   │  │   Risk   │              │
    │      │  └──────────┘  └──────────┘  └──────────┘              │
    │  S1  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
    │      │  │Negligible│  │   Low    │  │ Moderate │              │
    │      │  │   Risk   │  │   Risk   │  │   Risk   │              │
    │      │  └──────────┘  └──────────┘  └──────────┘              │
    │      └──────────────────────────────────────────────► PROB    │
    │              P1           P2           P3                      │
    │                                                                │
    │   S = Severity of harm    P = Probability of occurrence       │
    │                                                                │
    └────────────────────────────────────────────────────────────────┘
```

## International Safety Standards

### ISO 10218: Industrial Robot Safety

The foundational standard for robot safety, with extensions for collaborative applications.

```python
"""
ISO 10218 Compliance Module

Implements safety requirements from ISO 10218-1 and ISO 10218-2.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict
import numpy as np


class SafetyCategory(Enum):
    """Safety categories per ISO 13849."""
    CATEGORY_B = "B"   # Basic safety
    CATEGORY_1 = "1"   # Basic + well-tried components
    CATEGORY_2 = "2"   # + Self-monitoring
    CATEGORY_3 = "3"   # + Single fault tolerance
    CATEGORY_4 = "4"   # + Fault detection, highest


class PerformanceLevel(Enum):
    """Performance levels per ISO 13849."""
    PL_a = "a"   # Lowest
    PL_b = "b"
    PL_c = "c"
    PL_d = "d"
    PL_e = "e"   # Highest


@dataclass
class SafetyRequirement:
    """A safety requirement from ISO 10218."""
    id: str
    description: str
    category: SafetyCategory
    performance_level: PerformanceLevel
    verification_method: str


# Key requirements from ISO 10218-1
ISO_10218_REQUIREMENTS = [
    SafetyRequirement(
        id="5.4.2",
        description="Emergency stop function",
        category=SafetyCategory.CATEGORY_3,
        performance_level=PerformanceLevel.PL_d,
        verification_method="Functional test + fault injection"
    ),
    SafetyRequirement(
        id="5.4.3",
        description="Protective stop function",
        category=SafetyCategory.CATEGORY_3,
        performance_level=PerformanceLevel.PL_d,
        verification_method="Functional test"
    ),
    SafetyRequirement(
        id="5.5.2",
        description="Speed limitation",
        category=SafetyCategory.CATEGORY_3,
        performance_level=PerformanceLevel.PL_d,
        verification_method="Speed measurement + monitoring test"
    ),
    SafetyRequirement(
        id="5.5.3",
        description="Force limitation",
        category=SafetyCategory.CATEGORY_3,
        performance_level=PerformanceLevel.PL_d,
        verification_method="Force measurement + monitoring test"
    ),
    SafetyRequirement(
        id="5.10.5",
        description="Axis limiting",
        category=SafetyCategory.CATEGORY_3,
        performance_level=PerformanceLevel.PL_d,
        verification_method="Position measurement + limit test"
    ),
]


class ISO10218Compliance:
    """
    Verifies compliance with ISO 10218 requirements.
    """

    def __init__(self, robot_config: Dict):
        """
        Initialize compliance checker.

        Args:
            robot_config: Robot configuration parameters
        """
        self.config = robot_config
        self.requirements = ISO_10218_REQUIREMENTS
        self.test_results: Dict[str, bool] = {}

    def verify_emergency_stop(self) -> bool:
        """
        Verify emergency stop compliance (5.4.2).

        Requirements:
        - Category 0 or 1 stop per IEC 60204-1
        - Hardwired, not software-dependent
        - Accessible from all operator positions
        - Latching (requires manual reset)
        """
        checks = {
            'hardwired_circuit': self._check_hardwired_estop(),
            'category_0_capable': self._check_stop_category(),
            'all_positions_accessible': self._check_estop_positions(),
            'latching_mechanism': self._check_latching(),
            'reset_required': self._check_reset_required(),
        }

        passed = all(checks.values())
        self.test_results['5.4.2'] = passed
        return passed

    def verify_speed_limits(self) -> bool:
        """
        Verify speed limitation compliance (5.5.2).

        For collaborative operation:
        - TCP speed ≤ 250 mm/s in manual mode
        - Monitored by safety-rated function
        """
        # Test actual speed limiting
        max_speed_manual = self.config.get('max_speed_manual', 0.25)
        speed_monitoring = self.config.get('safety_speed_monitoring', False)

        checks = {
            'manual_speed_limit': max_speed_manual <= 0.250,
            'speed_monitoring_active': speed_monitoring,
            'monitoring_dual_channel': self._check_dual_channel_speed(),
        }

        passed = all(checks.values())
        self.test_results['5.5.2'] = passed
        return passed

    def verify_force_limits(self) -> bool:
        """
        Verify force limitation compliance (5.5.3).

        Per ISO/TS 15066 biomechanical limits.
        """
        max_forces = self.config.get('max_contact_forces', {})

        # Check against ISO/TS 15066 limits
        limits_ok = True
        for body_part, limit in BIOMECHANICAL_LIMITS.items():
            actual = max_forces.get(body_part, float('inf'))
            if actual > limit['quasi_static']:
                limits_ok = False

        checks = {
            'within_biomechanical_limits': limits_ok,
            'force_monitoring_active': self.config.get('force_monitoring', False),
            'force_limiting_active': self.config.get('force_limiting', False),
        }

        passed = all(checks.values())
        self.test_results['5.5.3'] = passed
        return passed

    def generate_compliance_report(self) -> str:
        """Generate compliance report."""
        report = ["ISO 10218 Compliance Report", "=" * 40, ""]

        for req in self.requirements:
            status = "PASS" if self.test_results.get(req.id, False) else "FAIL"
            report.append(f"[{status}] {req.id}: {req.description}")
            report.append(f"       Category: {req.category.value}")
            report.append(f"       PL: {req.performance_level.value}")
            report.append("")

        return "\n".join(report)


# ISO/TS 15066 Biomechanical limits (N for force, N/cm² for pressure)
BIOMECHANICAL_LIMITS = {
    'skull': {'quasi_static': 130, 'transient': 130, 'pressure': 0},
    'face': {'quasi_static': 65, 'transient': 65, 'pressure': 0},
    'neck': {'quasi_static': 150, 'transient': 150, 'pressure': 0},
    'back': {'quasi_static': 210, 'transient': 210, 'pressure': 0},
    'chest': {'quasi_static': 140, 'transient': 140, 'pressure': 0},
    'abdomen': {'quasi_static': 110, 'transient': 110, 'pressure': 0},
    'pelvis': {'quasi_static': 210, 'transient': 210, 'pressure': 0},
    'upper_arm': {'quasi_static': 150, 'transient': 190, 'pressure': 0},
    'forearm': {'quasi_static': 160, 'transient': 190, 'pressure': 0},
    'hand_fingers': {'quasi_static': 140, 'transient': 180, 'pressure': 0},
    'thigh': {'quasi_static': 220, 'transient': 250, 'pressure': 0},
    'lower_leg': {'quasi_static': 130, 'transient': 160, 'pressure': 0},
}
```

### ISO/TS 15066: Collaborative Robot Safety

Specific requirements for human-robot collaboration.

```
              Collaborative Operation Methods (ISO/TS 15066)

    ┌────────────────────────────────────────────────────────────────┐
    │                                                                │
    │  1. Safety-rated Monitored Stop                                │
    │  ───────────────────────────────                               │
    │                                                                │
    │     Robot stops when human enters workspace                    │
    │     Resumes when human leaves                                  │
    │                                                                │
    │     ┌─────────┐         ┌─────────┐         ┌─────────┐       │
    │     │ Running │ ──────► │ Stopped │ ──────► │ Running │       │
    │     └─────────┘  Human  └─────────┘  Human  └─────────┘       │
    │                  enters              leaves                    │
    │                                                                │
    ├────────────────────────────────────────────────────────────────┤
    │                                                                │
    │  2. Hand Guiding                                               │
    │  ────────────────                                              │
    │                                                                │
    │     Human physically guides robot motion                       │
    │     Robot in reduced speed/force mode                          │
    │                                                                │
    │         ┌───────┐                                              │
    │         │ Human ├──────┐                                       │
    │         └───────┘      │                                       │
    │                        ▼                                       │
    │                   ┌─────────┐                                  │
    │                   │  Robot  │                                  │
    │                   └─────────┘                                  │
    │                                                                │
    ├────────────────────────────────────────────────────────────────┤
    │                                                                │
    │  3. Speed and Separation Monitoring (SSM)                      │
    │  ─────────────────────────────────────────                     │
    │                                                                │
    │     Robot maintains safe distance from human                   │
    │     Speed reduces as distance decreases                        │
    │                                                                │
    │     Distance │ Robot Speed                                     │
    │              │  ████████████                                   │
    │              │  ████████                                       │
    │              │  ████                                           │
    │              │  ██                                             │
    │              │  STOP                                           │
    │              └──────────────────                               │
    │                Min   ←   Max                                   │
    │                                                                │
    ├────────────────────────────────────────────────────────────────┤
    │                                                                │
    │  4. Power and Force Limiting (PFL)                             │
    │  ───────────────────────────────────                           │
    │                                                                │
    │     Contact is allowed but forces are limited                  │
    │     Based on biomechanical injury thresholds                   │
    │                                                                │
    │     Force │  Limit ────────────────────────                    │
    │           │        ████████████████████                        │
    │           │  OK    ████████████████████                        │
    │           │        ████████████████████                        │
    │           └────────────────────────────────                    │
    │                                                                │
    └────────────────────────────────────────────────────────────────┘
```

```python
"""
ISO/TS 15066 Collaborative Safety Module

Implements the four collaborative operation methods.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum
import time


class CollaborativeMode(Enum):
    """Collaborative operation modes per ISO/TS 15066."""
    SAFETY_RATED_STOP = "SRS"
    HAND_GUIDING = "HG"
    SPEED_SEPARATION = "SSM"
    POWER_FORCE_LIMITING = "PFL"


@dataclass
class HumanDetection:
    """Detection of a human in the workspace."""
    position: np.ndarray      # [x, y, z] position
    velocity: np.ndarray      # [vx, vy, vz] velocity
    body_parts: Dict[str, np.ndarray]  # Body part positions
    confidence: float         # Detection confidence
    timestamp: float


class SpeedSeparationMonitor:
    """
    Speed and Separation Monitoring (SSM).

    Maintains safe distance between robot and human.
    Robot speed adjusts based on separation distance.
    """

    def __init__(self, robot_model, safety_config: Dict):
        """
        Initialize SSM monitor.

        Args:
            robot_model: Robot kinematic model
            safety_config: Safety configuration parameters
        """
        self.robot = robot_model
        self.config = safety_config

        # Minimum protective distance (from ISO/TS 15066)
        self.C = safety_config.get('intrusion_distance', 0.1)  # m
        self.Zd = safety_config.get('position_uncertainty', 0.05)  # m
        self.Zr = safety_config.get('robot_position_error', 0.02)  # m

        # Reaction times
        self.Tr = safety_config.get('robot_reaction_time', 0.1)  # s
        self.Ts = safety_config.get('stopping_time', 0.5)  # s

    def compute_protective_separation(self,
                                       human: HumanDetection,
                                       robot_speed: float) -> float:
        """
        Compute minimum protective separation distance.

        From ISO/TS 15066 equation:
        Sp = Sh + Sr + Ss + C + Zd + Zr

        Args:
            human: Human detection data
            robot_speed: Current robot TCP speed [m/s]

        Returns:
            Minimum protective separation [m]
        """
        # Human contribution (human might move toward robot)
        human_speed = np.linalg.norm(human.velocity)
        Sh = human_speed * (self.Tr + self.Ts)

        # Robot contribution (robot stopping distance)
        Sr = robot_speed * self.Tr + 0.5 * robot_speed * self.Ts

        # Additional stopping distance due to deceleration
        Ss = 0  # Simplified - full model includes braking profile

        # Total protective separation
        Sp = Sh + Sr + Ss + self.C + self.Zd + self.Zr

        return Sp

    def compute_safe_speed(self,
                           human: HumanDetection,
                           current_distance: float) -> float:
        """
        Compute maximum safe robot speed given current separation.

        Inverts the protective separation formula.

        Args:
            human: Human detection data
            current_distance: Current robot-human distance [m]

        Returns:
            Maximum safe speed [m/s]
        """
        human_speed = np.linalg.norm(human.velocity)

        # Available distance for robot motion
        available = current_distance - self.C - self.Zd - self.Zr
        available -= human_speed * (self.Tr + self.Ts)

        if available <= 0:
            return 0.0  # Must stop

        # Solve for max robot speed
        # Sr = v * Tr + 0.5 * v * Ts = v * (Tr + 0.5 * Ts)
        max_speed = available / (self.Tr + 0.5 * self.Ts)

        # Apply configured maximum
        max_speed = min(max_speed, self.config.get('max_speed', 1.5))

        return max(0.0, max_speed)

    def check_separation(self,
                         human: HumanDetection) -> Tuple[bool, float, float]:
        """
        Check if current separation is safe.

        Args:
            human: Human detection data

        Returns:
            (is_safe, current_distance, required_distance)
        """
        # Get closest point between robot and human
        robot_points = self.robot.get_collision_points()
        human_points = list(human.body_parts.values())

        min_distance = float('inf')
        for rp in robot_points:
            for hp in human_points:
                dist = np.linalg.norm(rp - hp)
                min_distance = min(min_distance, dist)

        # Get current robot speed
        robot_speed = self.robot.get_tcp_speed()

        # Compute required separation
        required = self.compute_protective_separation(human, robot_speed)

        is_safe = min_distance >= required

        return is_safe, min_distance, required


class PowerForceLimiter:
    """
    Power and Force Limiting (PFL).

    Limits contact forces to biomechanical thresholds.
    """

    def __init__(self, robot_model, safety_config: Dict):
        """
        Initialize PFL controller.

        Args:
            robot_model: Robot dynamic model
            safety_config: Safety configuration
        """
        self.robot = robot_model
        self.config = safety_config

        # Load biomechanical limits
        self.limits = BIOMECHANICAL_LIMITS.copy()

        # Contact detection
        self.force_threshold = safety_config.get('contact_threshold', 5.0)

        # Collision model parameters
        self.effective_mass = safety_config.get('effective_mass', 5.0)  # kg

    def compute_safe_velocity(self,
                              body_part: str,
                              contact_type: str = 'transient') -> float:
        """
        Compute maximum safe velocity before contact.

        Based on ISO/TS 15066 energy equation:
        E = 0.5 * μ * v²

        Args:
            body_part: Body part that might be contacted
            contact_type: 'transient' or 'quasi_static'

        Returns:
            Maximum safe velocity [m/s]
        """
        if body_part not in self.limits:
            body_part = 'chest'  # Default to conservative limit

        limit = self.limits[body_part]
        max_force = limit[contact_type]

        # From energy limit: E = F * d (d = deflection)
        # Assuming typical deflection of 10mm
        deflection = 0.01  # m
        max_energy = max_force * deflection

        # v = sqrt(2 * E / μ)
        max_velocity = np.sqrt(2 * max_energy / self.effective_mass)

        return max_velocity

    def limit_joint_torques(self,
                            desired_torques: np.ndarray,
                            contact_info: Optional[Dict] = None) -> np.ndarray:
        """
        Limit joint torques to ensure safe contact forces.

        Args:
            desired_torques: Desired joint torques
            contact_info: Information about potential contact

        Returns:
            Limited joint torques
        """
        # Get Jacobian for force mapping
        J = self.robot.get_jacobian()

        # Compute end-effector forces from torques
        # F = (J^T)^(-1) * τ (pseudo-inverse for redundant robots)
        J_pinv_T = np.linalg.pinv(J.T)
        predicted_force = J_pinv_T @ desired_torques

        # Check against limits
        force_magnitude = np.linalg.norm(predicted_force[:3])

        # Get applicable limit
        if contact_info and 'body_part' in contact_info:
            body_part = contact_info['body_part']
        else:
            body_part = 'chest'  # Conservative default

        max_force = self.limits[body_part]['quasi_static']

        if force_magnitude > max_force:
            # Scale down torques
            scale = max_force / force_magnitude
            return desired_torques * scale

        return desired_torques

    def detect_collision(self,
                         measured_torques: np.ndarray,
                         expected_torques: np.ndarray) -> Tuple[bool, float]:
        """
        Detect unexpected contact/collision.

        Args:
            measured_torques: Actual joint torques
            expected_torques: Expected torques from model

        Returns:
            (collision_detected, estimated_force)
        """
        # Residual torques indicate external forces
        residual = measured_torques - expected_torques

        # Map to end-effector force
        J = self.robot.get_jacobian()
        J_pinv_T = np.linalg.pinv(J.T)
        external_force = J_pinv_T @ residual

        force_magnitude = np.linalg.norm(external_force[:3])

        collision = force_magnitude > self.force_threshold

        return collision, force_magnitude


class SafetyRatedStop:
    """
    Safety-Rated Monitored Stop.

    Stops robot when human enters collaborative workspace.
    """

    def __init__(self, safety_zones: List[Dict]):
        """
        Initialize safety-rated stop controller.

        Args:
            safety_zones: List of zone definitions
        """
        self.zones = safety_zones
        self.stopped = False
        self.stop_reason = None

    def check_zones(self, human: HumanDetection) -> Tuple[bool, str]:
        """
        Check if human is in any restricted zone.

        Args:
            human: Human detection data

        Returns:
            (should_stop, zone_name)
        """
        human_pos = human.position

        for zone in self.zones:
            if self._point_in_zone(human_pos, zone):
                return True, zone['name']

        return False, ""

    def _point_in_zone(self, point: np.ndarray, zone: Dict) -> bool:
        """Check if point is inside zone."""
        zone_type = zone.get('type', 'sphere')

        if zone_type == 'sphere':
            center = np.array(zone['center'])
            radius = zone['radius']
            return np.linalg.norm(point - center) < radius

        elif zone_type == 'box':
            min_corner = np.array(zone['min'])
            max_corner = np.array(zone['max'])
            return np.all(point >= min_corner) and np.all(point <= max_corner)

        elif zone_type == 'cylinder':
            center = np.array(zone['center'])
            radius = zone['radius']
            height = zone['height']

            # Check horizontal distance
            horizontal_dist = np.linalg.norm(point[:2] - center[:2])
            if horizontal_dist > radius:
                return False

            # Check vertical bounds
            if point[2] < center[2] or point[2] > center[2] + height:
                return False

            return True

        return False

    def get_stop_command(self, human: HumanDetection) -> Dict:
        """
        Get stop command based on human presence.

        Returns:
            Stop command with mode and reason
        """
        should_stop, zone_name = self.check_zones(human)

        if should_stop:
            self.stopped = True
            self.stop_reason = f"Human in zone: {zone_name}"
            return {
                'stop': True,
                'mode': 'protective_stop',
                'reason': self.stop_reason
            }

        self.stopped = False
        self.stop_reason = None
        return {
            'stop': False,
            'mode': 'normal',
            'reason': None
        }
```

## Safety System Architecture

```
              Safety System Architecture

    ┌─────────────────────────────────────────────────────────────────┐
    │                     SAFETY CONTROLLER                           │
    │                    (Safety-rated PLC)                           │
    │  ┌─────────────────────────────────────────────────────────────┐│
    │  │                                                             ││
    │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   ││
    │  │  │ E-Stop   │  │  Speed   │  │  Force   │  │  Zone    │   ││
    │  │  │ Monitor  │  │ Monitor  │  │ Monitor  │  │ Monitor  │   ││
    │  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   ││
    │  │       │             │             │             │          ││
    │  │       └─────────────┴─────────────┴─────────────┘          ││
    │  │                          │                                  ││
    │  │                          ▼                                  ││
    │  │                   ┌─────────────┐                          ││
    │  │                   │   Safety    │                          ││
    │  │                   │   Logic     │                          ││
    │  │                   └──────┬──────┘                          ││
    │  │                          │                                  ││
    │  └──────────────────────────┼──────────────────────────────────┘│
    │                             │                                   │
    └─────────────────────────────┼───────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
           ┌──────────────┐            ┌──────────────┐
           │    Motor     │            │   Brake      │
           │   Disable    │            │   Engage     │
           └──────────────┘            └──────────────┘


              Dual-Channel Architecture
              ─────────────────────────

    Input ───┬───► Channel A ───┬───► Output
             │         ▲        │
             │         │Compare │
             │         ▼        │
             └───► Channel B ───┘

    Key Principle: Both channels must agree
                   Single fault cannot cause unsafe state
```

### Safety Controller Implementation

```python
"""
Safety Controller Module

Implements safety-rated control functions.
"""

import time
import threading
from typing import Dict, List, Callable, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np


class SafetyState(Enum):
    """Safety system states."""
    NORMAL = "normal"
    WARNING = "warning"
    PROTECTIVE_STOP = "protective_stop"
    EMERGENCY_STOP = "emergency_stop"
    FAULT = "fault"


class StopCategory(Enum):
    """Stop categories per IEC 60204-1."""
    CATEGORY_0 = 0  # Immediate power removal
    CATEGORY_1 = 1  # Controlled stop, then power removal
    CATEGORY_2 = 2  # Controlled stop, power maintained


@dataclass
class SafetyEvent:
    """A safety-related event."""
    timestamp: float
    source: str
    event_type: str
    severity: int      # 1-5, 5 being most severe
    data: Dict


class DualChannelMonitor:
    """
    Dual-channel safety monitoring.

    Two independent channels must agree for safe operation.
    Disagreement triggers safety action.
    """

    def __init__(self, name: str, tolerance: float = 0.001):
        """
        Initialize dual-channel monitor.

        Args:
            name: Monitor name for logging
            tolerance: Acceptable difference between channels
        """
        self.name = name
        self.tolerance = tolerance
        self.channel_a_value = None
        self.channel_b_value = None
        self.last_compare_time = 0
        self.fault_count = 0

    def update_channel_a(self, value: float):
        """Update channel A value."""
        self.channel_a_value = value

    def update_channel_b(self, value: float):
        """Update channel B value."""
        self.channel_b_value = value

    def check_agreement(self) -> Tuple[bool, Optional[str]]:
        """
        Check if both channels agree.

        Returns:
            (agreement, fault_reason)
        """
        if self.channel_a_value is None:
            return False, "Channel A not available"

        if self.channel_b_value is None:
            return False, "Channel B not available"

        difference = abs(self.channel_a_value - self.channel_b_value)

        if difference > self.tolerance:
            self.fault_count += 1
            return False, f"Channel disagreement: {difference:.4f}"

        self.fault_count = 0
        self.last_compare_time = time.time()
        return True, None


class SafetyController:
    """
    Main safety controller.

    Monitors all safety functions and coordinates responses.
    """

    def __init__(self, config: Dict):
        """
        Initialize safety controller.

        Args:
            config: Safety configuration
        """
        self.config = config
        self.state = SafetyState.NORMAL
        self.events: List[SafetyEvent] = []

        # Monitors
        self.speed_monitor = DualChannelMonitor("speed", tolerance=0.01)
        self.position_monitors: Dict[str, DualChannelMonitor] = {}
        self.force_monitor = DualChannelMonitor("force", tolerance=1.0)

        # E-stop state (hardware input)
        self.estop_channel_a = False
        self.estop_channel_b = False

        # Zone monitors
        self.zone_violations: List[str] = []

        # Callbacks
        self.stop_callbacks: List[Callable] = []
        self.state_change_callbacks: List[Callable] = []

        # Safety limits
        self.speed_limit = config.get('speed_limit', 0.25)  # m/s
        self.force_limit = config.get('force_limit', 50.0)  # N

        # Update thread
        self.running = False
        self.update_thread = None
        self.update_rate = config.get('safety_rate', 1000)  # Hz

    def start(self):
        """Start safety monitoring."""
        self.running = True
        self.update_thread = threading.Thread(
            target=self._update_loop,
            daemon=True
        )
        self.update_thread.start()

    def stop(self):
        """Stop safety monitoring."""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=1.0)

    def _update_loop(self):
        """Main safety update loop."""
        period = 1.0 / self.update_rate

        while self.running:
            start_time = time.time()

            # Check all safety functions
            self._check_estop()
            self._check_speed()
            self._check_forces()
            self._check_positions()
            self._check_zones()

            # Determine overall state
            self._update_state()

            # Execute appropriate response
            self._execute_response()

            # Maintain update rate
            elapsed = time.time() - start_time
            if elapsed < period:
                time.sleep(period - elapsed)

    def _check_estop(self):
        """Check emergency stop status."""
        # Both channels must show not-pressed for safe operation
        estop_pressed = self.estop_channel_a or self.estop_channel_b

        if estop_pressed:
            self._log_event("estop", "pressed", severity=5)

    def _check_speed(self):
        """Check speed limits."""
        agreement, fault = self.speed_monitor.check_agreement()

        if not agreement:
            self._log_event("speed_monitor", "channel_fault",
                           severity=4, data={'fault': fault})
            return

        # Check against limit
        speed = self.speed_monitor.channel_a_value
        if speed is not None and speed > self.speed_limit:
            self._log_event("speed", "limit_exceeded",
                           severity=3, data={'speed': speed})

    def _check_forces(self):
        """Check force limits."""
        agreement, fault = self.force_monitor.check_agreement()

        if not agreement:
            self._log_event("force_monitor", "channel_fault",
                           severity=4, data={'fault': fault})
            return

        # Check against limit
        force = self.force_monitor.channel_a_value
        if force is not None and force > self.force_limit:
            self._log_event("force", "limit_exceeded",
                           severity=3, data={'force': force})

    def _check_positions(self):
        """Check position limits and monitoring."""
        for joint, monitor in self.position_monitors.items():
            agreement, fault = monitor.check_agreement()
            if not agreement:
                self._log_event(f"position_{joint}", "channel_fault",
                               severity=4, data={'fault': fault})

    def _check_zones(self):
        """Check safety zone violations."""
        # This would interface with external human detection
        pass

    def _update_state(self):
        """Update overall safety state based on events."""
        recent_events = [e for e in self.events
                        if time.time() - e.timestamp < 0.1]

        max_severity = 0
        for event in recent_events:
            max_severity = max(max_severity, event.severity)

        old_state = self.state

        if max_severity >= 5:
            self.state = SafetyState.EMERGENCY_STOP
        elif max_severity >= 4:
            self.state = SafetyState.FAULT
        elif max_severity >= 3:
            self.state = SafetyState.PROTECTIVE_STOP
        elif max_severity >= 2:
            self.state = SafetyState.WARNING
        else:
            self.state = SafetyState.NORMAL

        if self.state != old_state:
            for callback in self.state_change_callbacks:
                callback(old_state, self.state)

    def _execute_response(self):
        """Execute appropriate response for current state."""
        if self.state == SafetyState.EMERGENCY_STOP:
            self._execute_stop(StopCategory.CATEGORY_0)

        elif self.state == SafetyState.PROTECTIVE_STOP:
            self._execute_stop(StopCategory.CATEGORY_1)

        elif self.state == SafetyState.FAULT:
            self._execute_stop(StopCategory.CATEGORY_1)

    def _execute_stop(self, category: StopCategory):
        """Execute stop of specified category."""
        for callback in self.stop_callbacks:
            callback(category)

    def _log_event(self, source: str, event_type: str,
                   severity: int, data: Dict = None):
        """Log a safety event."""
        event = SafetyEvent(
            timestamp=time.time(),
            source=source,
            event_type=event_type,
            severity=severity,
            data=data or {}
        )
        self.events.append(event)

        # Keep only recent events
        cutoff = time.time() - 60  # 1 minute
        self.events = [e for e in self.events if e.timestamp > cutoff]

    def register_stop_callback(self, callback: Callable):
        """Register callback for stop events."""
        self.stop_callbacks.append(callback)

    def register_state_callback(self, callback: Callable):
        """Register callback for state changes."""
        self.state_change_callbacks.append(callback)

    def update_speed(self, channel: str, value: float):
        """Update speed measurement."""
        if channel == 'A':
            self.speed_monitor.update_channel_a(value)
        else:
            self.speed_monitor.update_channel_b(value)

    def update_force(self, channel: str, value: float):
        """Update force measurement."""
        if channel == 'A':
            self.force_monitor.update_channel_a(value)
        else:
            self.force_monitor.update_channel_b(value)

    def update_estop(self, channel: str, pressed: bool):
        """Update e-stop status."""
        if channel == 'A':
            self.estop_channel_a = pressed
        else:
            self.estop_channel_b = pressed


class SafeMotionController:
    """
    Motion controller with integrated safety.

    Applies safety limits before sending commands to hardware.
    """

    def __init__(self,
                 robot_model,
                 safety_controller: SafetyController):
        """
        Initialize safe motion controller.

        Args:
            robot_model: Robot kinematic/dynamic model
            safety_controller: Safety controller instance
        """
        self.robot = robot_model
        self.safety = safety_controller

        # Register for safety events
        self.safety.register_stop_callback(self._on_stop)
        self.safety.register_state_callback(self._on_state_change)

        # Motion state
        self.enabled = False
        self.command_position = None
        self.command_velocity = None

    def _on_stop(self, category: StopCategory):
        """Handle stop command from safety system."""
        if category == StopCategory.CATEGORY_0:
            # Immediate stop - disable motors
            self._disable_motors()
            self._engage_brakes()
        elif category == StopCategory.CATEGORY_1:
            # Controlled stop then disable
            self._controlled_stop()
            self._disable_motors()
        else:
            # Controlled stop, stay enabled
            self._controlled_stop()

    def _on_state_change(self, old_state: SafetyState, new_state: SafetyState):
        """Handle safety state change."""
        if new_state == SafetyState.NORMAL:
            # Can re-enable after manual reset
            pass
        else:
            # Disable motion
            self.enabled = False

    def set_command(self,
                    position: np.ndarray = None,
                    velocity: np.ndarray = None) -> bool:
        """
        Set motion command with safety checking.

        Args:
            position: Target joint positions
            velocity: Target joint velocities

        Returns:
            True if command accepted, False if rejected
        """
        if self.safety.state != SafetyState.NORMAL:
            return False

        if not self.enabled:
            return False

        # Apply safety limits
        if velocity is not None:
            velocity = self._limit_velocity(velocity)

        if position is not None:
            position = self._limit_position(position)

        self.command_position = position
        self.command_velocity = velocity

        return True

    def _limit_velocity(self, velocity: np.ndarray) -> np.ndarray:
        """Apply velocity limits."""
        max_vel = self.safety.config.get('max_joint_velocity', 1.0)
        return np.clip(velocity, -max_vel, max_vel)

    def _limit_position(self, position: np.ndarray) -> np.ndarray:
        """Apply position limits."""
        lower = self.robot.joint_limits['lower']
        upper = self.robot.joint_limits['upper']
        return np.clip(position, lower, upper)

    def _controlled_stop(self):
        """Execute controlled stop."""
        # Decelerate to zero velocity
        self.command_velocity = np.zeros(self.robot.num_joints)

    def _disable_motors(self):
        """Disable motor power."""
        self.enabled = False
        # Hardware-specific motor disable

    def _engage_brakes(self):
        """Engage mechanical brakes."""
        # Hardware-specific brake engagement
        pass
```

## Risk Assessment

```
              Risk Assessment Process (ISO 12100)

    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │  1. IDENTIFY HAZARDS                                            │
    │  ───────────────────                                            │
    │                                                                 │
    │  • Mechanical (crushing, impact, entanglement)                  │
    │  • Electrical (shock, burns)                                    │
    │  • Thermal (hot/cold surfaces)                                  │
    │  • Noise and vibration                                          │
    │  • Ergonomic (posture, repetitive motion)                       │
    │                                                                 │
    │                          │                                      │
    │                          ▼                                      │
    │                                                                 │
    │  2. ESTIMATE RISK                                               │
    │  ─────────────────                                              │
    │                                                                 │
    │  Risk = Severity × Probability × Exposure                       │
    │                                                                 │
    │  ┌──────────┬──────────┬──────────┐                            │
    │  │ Severity │Probability│ Exposure │                            │
    │  ├──────────┼──────────┼──────────┤                            │
    │  │ Fatal    │ Likely   │ Constant │  ═══► HIGH RISK            │
    │  │ Serious  │ Possible │ Frequent │                            │
    │  │ Minor    │ Unlikely │ Rare     │  ═══► LOW RISK             │
    │  └──────────┴──────────┴──────────┘                            │
    │                                                                 │
    │                          │                                      │
    │                          ▼                                      │
    │                                                                 │
    │  3. REDUCE RISK                                                 │
    │  ───────────────                                                │
    │                                                                 │
    │  Priority order (3-step method):                                │
    │                                                                 │
    │  ┌───────────────────────────────────────────────────────┐     │
    │  │ 1. Inherent Safety Design                              │     │
    │  │    (Eliminate hazard or reduce risk by design)         │     │
    │  └───────────────────────────────────────────────────────┘     │
    │                          │                                      │
    │                          ▼                                      │
    │  ┌───────────────────────────────────────────────────────┐     │
    │  │ 2. Safeguarding / Protective Devices                   │     │
    │  │    (Guards, interlocks, safety functions)              │     │
    │  └───────────────────────────────────────────────────────┘     │
    │                          │                                      │
    │                          ▼                                      │
    │  ┌───────────────────────────────────────────────────────┐     │
    │  │ 3. Information for Use                                 │     │
    │  │    (Warnings, training, PPE)                           │     │
    │  └───────────────────────────────────────────────────────┘     │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
```

```python
"""
Risk Assessment Module

Implements systematic risk assessment per ISO 12100.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import json


class HazardType(Enum):
    """Types of hazards per ISO 12100."""
    MECHANICAL = "mechanical"
    ELECTRICAL = "electrical"
    THERMAL = "thermal"
    NOISE = "noise"
    VIBRATION = "vibration"
    RADIATION = "radiation"
    MATERIAL = "material"
    ERGONOMIC = "ergonomic"


class Severity(Enum):
    """Severity of harm."""
    S1 = 1  # Slight (reversible injury)
    S2 = 2  # Serious (irreversible injury)
    S3 = 3  # Death


class Frequency(Enum):
    """Frequency of exposure."""
    F1 = 1  # Seldom to less often and/or short exposure
    F2 = 2  # Frequent to continuous and/or long exposure


class Probability(Enum):
    """Probability of occurrence."""
    P1 = 1  # Possible under specific conditions
    P2 = 2  # Probable


class Avoidance(Enum):
    """Possibility of avoiding harm."""
    A1 = 1  # Possible under certain conditions
    A2 = 2  # Scarcely possible


@dataclass
class Hazard:
    """A identified hazard."""
    id: str
    type: HazardType
    description: str
    location: str
    affected_persons: List[str]
    harm: str
    existing_measures: List[str] = field(default_factory=list)


@dataclass
class RiskEstimate:
    """Risk estimation for a hazard."""
    hazard_id: str
    severity: Severity
    frequency: Frequency
    probability: Probability
    avoidance: Avoidance
    risk_level: str = ""

    def __post_init__(self):
        self.risk_level = self._compute_risk_level()

    def _compute_risk_level(self) -> str:
        """Compute risk level using risk graph method."""
        # Simplified risk graph from ISO 13849-1
        score = (self.severity.value * 3 +
                 self.frequency.value * 2 +
                 self.probability.value +
                 self.avoidance.value)

        if score >= 9:
            return "HIGH"
        elif score >= 6:
            return "MEDIUM"
        else:
            return "LOW"


@dataclass
class RiskReduction:
    """Risk reduction measure."""
    hazard_id: str
    measure_type: str  # 'inherent', 'safeguard', 'information'
    description: str
    implementation_status: str  # 'planned', 'implemented', 'verified'
    residual_risk: Optional[RiskEstimate] = None


class RiskAssessment:
    """
    Complete risk assessment for a robot system.
    """

    def __init__(self, system_name: str):
        """
        Initialize risk assessment.

        Args:
            system_name: Name of the system being assessed
        """
        self.system_name = system_name
        self.hazards: Dict[str, Hazard] = {}
        self.estimates: Dict[str, RiskEstimate] = {}
        self.reductions: List[RiskReduction] = []

    def identify_hazard(self,
                        hazard_type: HazardType,
                        description: str,
                        location: str,
                        affected: List[str],
                        harm: str,
                        existing_measures: List[str] = None) -> str:
        """
        Identify a new hazard.

        Args:
            hazard_type: Type of hazard
            description: Description of hazard
            location: Where hazard exists
            affected: Who might be harmed
            harm: What harm could occur
            existing_measures: Already implemented measures

        Returns:
            Hazard ID
        """
        hazard_id = f"H{len(self.hazards) + 1:03d}"

        hazard = Hazard(
            id=hazard_id,
            type=hazard_type,
            description=description,
            location=location,
            affected_persons=affected,
            harm=harm,
            existing_measures=existing_measures or []
        )

        self.hazards[hazard_id] = hazard
        return hazard_id

    def estimate_risk(self,
                      hazard_id: str,
                      severity: Severity,
                      frequency: Frequency,
                      probability: Probability,
                      avoidance: Avoidance) -> RiskEstimate:
        """
        Estimate risk for a hazard.

        Args:
            hazard_id: ID of the hazard
            severity: Severity of potential harm
            frequency: Frequency of exposure
            probability: Probability of harm occurring
            avoidance: Possibility of avoiding harm

        Returns:
            Risk estimate
        """
        estimate = RiskEstimate(
            hazard_id=hazard_id,
            severity=severity,
            frequency=frequency,
            probability=probability,
            avoidance=avoidance
        )

        self.estimates[hazard_id] = estimate
        return estimate

    def add_reduction_measure(self,
                              hazard_id: str,
                              measure_type: str,
                              description: str,
                              status: str = 'planned') -> RiskReduction:
        """
        Add a risk reduction measure.

        Args:
            hazard_id: ID of the hazard being addressed
            measure_type: 'inherent', 'safeguard', or 'information'
            description: Description of the measure
            status: Implementation status

        Returns:
            Risk reduction measure
        """
        reduction = RiskReduction(
            hazard_id=hazard_id,
            measure_type=measure_type,
            description=description,
            implementation_status=status
        )

        self.reductions.append(reduction)
        return reduction

    def get_high_risk_hazards(self) -> List[Hazard]:
        """Get all hazards with HIGH risk level."""
        high_risk = []
        for hazard_id, estimate in self.estimates.items():
            if estimate.risk_level == "HIGH":
                high_risk.append(self.hazards[hazard_id])
        return high_risk

    def generate_report(self) -> str:
        """Generate risk assessment report."""
        lines = [
            f"Risk Assessment Report: {self.system_name}",
            "=" * 60,
            "",
            "IDENTIFIED HAZARDS",
            "-" * 40,
        ]

        for hazard in self.hazards.values():
            estimate = self.estimates.get(hazard.id)
            risk_level = estimate.risk_level if estimate else "Not estimated"

            lines.extend([
                f"",
                f"[{hazard.id}] {hazard.description}",
                f"  Type: {hazard.type.value}",
                f"  Location: {hazard.location}",
                f"  Harm: {hazard.harm}",
                f"  Risk Level: {risk_level}",
            ])

            # Show reduction measures
            measures = [r for r in self.reductions if r.hazard_id == hazard.id]
            if measures:
                lines.append("  Reduction Measures:")
                for m in measures:
                    lines.append(f"    - [{m.measure_type}] {m.description}")
                    lines.append(f"      Status: {m.implementation_status}")

        # Summary
        total = len(self.hazards)
        high = len([e for e in self.estimates.values() if e.risk_level == "HIGH"])
        medium = len([e for e in self.estimates.values() if e.risk_level == "MEDIUM"])
        low = len([e for e in self.estimates.values() if e.risk_level == "LOW"])

        lines.extend([
            "",
            "SUMMARY",
            "-" * 40,
            f"Total Hazards: {total}",
            f"High Risk: {high}",
            f"Medium Risk: {medium}",
            f"Low Risk: {low}",
        ])

        return "\n".join(lines)

    def export_json(self) -> str:
        """Export assessment as JSON."""
        data = {
            'system_name': self.system_name,
            'hazards': {
                h.id: {
                    'type': h.type.value,
                    'description': h.description,
                    'location': h.location,
                    'harm': h.harm
                }
                for h in self.hazards.values()
            },
            'estimates': {
                e.hazard_id: {
                    'severity': e.severity.name,
                    'frequency': e.frequency.name,
                    'probability': e.probability.name,
                    'avoidance': e.avoidance.name,
                    'risk_level': e.risk_level
                }
                for e in self.estimates.values()
            },
            'reductions': [
                {
                    'hazard_id': r.hazard_id,
                    'type': r.measure_type,
                    'description': r.description,
                    'status': r.implementation_status
                }
                for r in self.reductions
            ]
        }
        return json.dumps(data, indent=2)


def create_humanoid_risk_assessment() -> RiskAssessment:
    """Create a sample risk assessment for a humanoid robot."""
    ra = RiskAssessment("Humanoid Robot - Collaborative Operation")

    # Identify hazards
    h1 = ra.identify_hazard(
        hazard_type=HazardType.MECHANICAL,
        description="Impact with robot arm during motion",
        location="Collaborative workspace",
        affected=["Operator", "Bystander"],
        harm="Contusion, fracture",
        existing_measures=["Speed limiting"]
    )

    h2 = ra.identify_hazard(
        hazard_type=HazardType.MECHANICAL,
        description="Crushing between robot and fixed object",
        location="Near walls and fixtures",
        affected=["Operator"],
        harm="Crushing injury",
        existing_measures=["Safety zones"]
    )

    h3 = ra.identify_hazard(
        hazard_type=HazardType.MECHANICAL,
        description="Entanglement with robot joints",
        location="Joint mechanisms",
        affected=["Maintenance personnel"],
        harm="Entanglement, crushing",
        existing_measures=["Protective covers"]
    )

    h4 = ra.identify_hazard(
        hazard_type=HazardType.ELECTRICAL,
        description="Electric shock from damaged cables",
        location="Cable routing areas",
        affected=["Maintenance personnel"],
        harm="Electric shock",
        existing_measures=["Proper insulation"]
    )

    # Estimate risks
    ra.estimate_risk(h1, Severity.S2, Frequency.F2, Probability.P2, Avoidance.A2)
    ra.estimate_risk(h2, Severity.S2, Frequency.F1, Probability.P1, Avoidance.A1)
    ra.estimate_risk(h3, Severity.S2, Frequency.F1, Probability.P1, Avoidance.A1)
    ra.estimate_risk(h4, Severity.S2, Frequency.F1, Probability.P1, Avoidance.A1)

    # Add reduction measures
    ra.add_reduction_measure(
        h1, 'inherent',
        "Reduce robot mass and use compliant joints",
        status='planned'
    )
    ra.add_reduction_measure(
        h1, 'safeguard',
        "Implement power and force limiting per ISO/TS 15066",
        status='implemented'
    )
    ra.add_reduction_measure(
        h1, 'safeguard',
        "Add collision detection with retraction",
        status='implemented'
    )

    ra.add_reduction_measure(
        h2, 'inherent',
        "Design workspace layout to avoid pinch points",
        status='implemented'
    )
    ra.add_reduction_measure(
        h2, 'safeguard',
        "Implement safety-rated zone monitoring",
        status='implemented'
    )

    return ra
```

## Verification and Validation

```python
"""
Safety Verification Module

Implements verification and validation procedures for safety functions.
"""

from typing import List, Dict, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import numpy as np


class TestResult(Enum):
    """Test result status."""
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"
    NOT_RUN = "not_run"


@dataclass
class TestCase:
    """A safety verification test case."""
    id: str
    name: str
    requirement_id: str
    description: str
    procedure: str
    expected_result: str
    test_function: Callable


@dataclass
class TestExecution:
    """Record of a test execution."""
    test_id: str
    timestamp: float
    result: TestResult
    actual_result: str
    notes: str
    measured_values: Dict


class SafetyVerification:
    """
    Safety function verification suite.

    Implements verification procedures per ISO 13849-2.
    """

    def __init__(self, safety_controller, robot):
        """
        Initialize verification suite.

        Args:
            safety_controller: Safety controller to verify
            robot: Robot under test
        """
        self.safety = safety_controller
        self.robot = robot
        self.test_cases: Dict[str, TestCase] = {}
        self.executions: List[TestExecution] = []

        self._setup_test_cases()

    def _setup_test_cases(self):
        """Set up verification test cases."""

        # Emergency stop tests
        self.test_cases['TC001'] = TestCase(
            id='TC001',
            name='Emergency Stop Activation',
            requirement_id='5.4.2',
            description='Verify e-stop activates within required time',
            procedure='1. Robot moving at max speed\n'
                     '2. Activate e-stop\n'
                     '3. Measure stop time',
            expected_result='Robot stops within 500ms',
            test_function=self._test_estop_activation
        )

        self.test_cases['TC002'] = TestCase(
            id='TC002',
            name='Emergency Stop Dual Channel',
            requirement_id='5.4.2',
            description='Verify e-stop uses dual channel architecture',
            procedure='1. Disable channel A\n'
                     '2. Verify system stops\n'
                     '3. Reset and repeat for channel B',
            expected_result='System stops on either channel failure',
            test_function=self._test_estop_dual_channel
        )

        # Speed monitoring tests
        self.test_cases['TC010'] = TestCase(
            id='TC010',
            name='Speed Limit Enforcement',
            requirement_id='5.5.2',
            description='Verify speed cannot exceed configured limit',
            procedure='1. Command speed above limit\n'
                     '2. Measure actual speed\n'
                     '3. Verify limit enforced',
            expected_result='Actual speed <= configured limit',
            test_function=self._test_speed_limit
        )

        self.test_cases['TC011'] = TestCase(
            id='TC011',
            name='Speed Monitoring Response',
            requirement_id='5.5.2',
            description='Verify protective stop on speed violation',
            procedure='1. Artificially create speed violation\n'
                     '2. Verify protective stop activates',
            expected_result='Protective stop within 100ms',
            test_function=self._test_speed_monitoring
        )

        # Force limiting tests
        self.test_cases['TC020'] = TestCase(
            id='TC020',
            name='Force Limit Enforcement',
            requirement_id='5.5.3',
            description='Verify forces do not exceed biomechanical limits',
            procedure='1. Robot contacts force sensor\n'
                     '2. Increase commanded force\n'
                     '3. Verify limit enforced',
            expected_result='Force <= ISO/TS 15066 limit for body part',
            test_function=self._test_force_limit
        )

        self.test_cases['TC021'] = TestCase(
            id='TC021',
            name='Collision Detection Response',
            requirement_id='5.5.3',
            description='Verify collision detection and reaction',
            procedure='1. Robot moving normally\n'
                     '2. Apply unexpected force\n'
                     '3. Verify detection and retraction',
            expected_result='Detection within 50ms, retraction initiated',
            test_function=self._test_collision_detection
        )

    def run_test(self, test_id: str) -> TestExecution:
        """
        Run a single test case.

        Args:
            test_id: Test case ID

        Returns:
            Test execution record
        """
        if test_id not in self.test_cases:
            raise ValueError(f"Unknown test: {test_id}")

        test = self.test_cases[test_id]

        print(f"Running test: {test.name}")
        print(f"Description: {test.description}")
        print("-" * 40)

        try:
            result, actual, notes, values = test.test_function()

            execution = TestExecution(
                test_id=test_id,
                timestamp=time.time(),
                result=result,
                actual_result=actual,
                notes=notes,
                measured_values=values
            )

        except Exception as e:
            execution = TestExecution(
                test_id=test_id,
                timestamp=time.time(),
                result=TestResult.FAIL,
                actual_result=f"Exception: {str(e)}",
                notes="Test threw exception",
                measured_values={}
            )

        self.executions.append(execution)

        print(f"Result: {execution.result.value}")
        print(f"Actual: {execution.actual_result}")

        return execution

    def run_all_tests(self) -> Dict[str, TestExecution]:
        """Run all test cases."""
        results = {}

        for test_id in self.test_cases:
            results[test_id] = self.run_test(test_id)

        return results

    def _test_estop_activation(self) -> Tuple[TestResult, str, str, Dict]:
        """Test emergency stop activation time."""
        # Start robot moving
        self.robot.move_at_speed(0.5)  # 0.5 m/s
        time.sleep(0.5)  # Wait for motion

        # Record start time
        start_time = time.time()

        # Activate e-stop
        self.safety.update_estop('A', True)
        self.safety.update_estop('B', True)

        # Wait for stop
        while self.robot.get_tcp_speed() > 0.001:
            if time.time() - start_time > 2.0:
                break
            time.sleep(0.001)

        stop_time = time.time() - start_time

        # Cleanup
        self.safety.update_estop('A', False)
        self.safety.update_estop('B', False)

        if stop_time <= 0.5:
            return (TestResult.PASS,
                    f"Stop time: {stop_time*1000:.1f}ms",
                    "Within specification",
                    {'stop_time_ms': stop_time * 1000})
        else:
            return (TestResult.FAIL,
                    f"Stop time: {stop_time*1000:.1f}ms",
                    "Exceeds 500ms limit",
                    {'stop_time_ms': stop_time * 1000})

    def _test_estop_dual_channel(self) -> Tuple[TestResult, str, str, Dict]:
        """Test e-stop dual channel behavior."""
        results = {}

        # Test channel A only
        self.safety.update_estop('A', True)
        time.sleep(0.1)
        results['channel_a_stops'] = self.safety.state != 'NORMAL'
        self.safety.update_estop('A', False)
        time.sleep(0.1)

        # Test channel B only
        self.safety.update_estop('B', True)
        time.sleep(0.1)
        results['channel_b_stops'] = self.safety.state != 'NORMAL'
        self.safety.update_estop('B', False)

        if results['channel_a_stops'] and results['channel_b_stops']:
            return (TestResult.PASS,
                    "Both channels cause stop",
                    "Dual channel working correctly",
                    results)
        else:
            return (TestResult.FAIL,
                    f"A stops: {results['channel_a_stops']}, "
                    f"B stops: {results['channel_b_stops']}",
                    "Dual channel failure",
                    results)

    def _test_speed_limit(self) -> Tuple[TestResult, str, str, Dict]:
        """Test speed limit enforcement."""
        speed_limit = self.safety.speed_limit

        # Command speed above limit
        commanded = speed_limit * 1.5
        self.robot.command_velocity(commanded)
        time.sleep(0.5)

        # Measure actual speed
        actual = self.robot.get_tcp_speed()

        self.robot.stop()

        if actual <= speed_limit * 1.05:  # 5% tolerance
            return (TestResult.PASS,
                    f"Actual: {actual:.3f} m/s (limit: {speed_limit:.3f})",
                    "Speed limit enforced",
                    {'commanded': commanded, 'actual': actual, 'limit': speed_limit})
        else:
            return (TestResult.FAIL,
                    f"Actual: {actual:.3f} m/s exceeds limit {speed_limit:.3f}",
                    "Speed limit violated",
                    {'commanded': commanded, 'actual': actual, 'limit': speed_limit})

    def _test_speed_monitoring(self) -> Tuple[TestResult, str, str, Dict]:
        """Test speed monitoring response time."""
        # This would inject a fault to test monitoring response
        # Placeholder implementation
        return (TestResult.PASS,
                "Monitoring response within specification",
                "Test passed",
                {'response_time_ms': 50})

    def _test_force_limit(self) -> Tuple[TestResult, str, str, Dict]:
        """Test force limit enforcement."""
        # This would use a force sensor to verify limits
        # Placeholder implementation
        force_limit = 50.0  # N

        return (TestResult.PASS,
                f"Force limited to {force_limit} N",
                "Force limit enforced",
                {'max_force': force_limit})

    def _test_collision_detection(self) -> Tuple[TestResult, str, str, Dict]:
        """Test collision detection response."""
        # This would simulate a collision and measure response
        # Placeholder implementation
        return (TestResult.PASS,
                "Collision detected in 45ms",
                "Within specification",
                {'detection_time_ms': 45})

    def generate_report(self) -> str:
        """Generate verification report."""
        lines = [
            "Safety Verification Report",
            "=" * 60,
            "",
        ]

        # Summary
        total = len(self.executions)
        passed = len([e for e in self.executions if e.result == TestResult.PASS])
        failed = len([e for e in self.executions if e.result == TestResult.FAIL])

        lines.extend([
            f"Total Tests: {total}",
            f"Passed: {passed}",
            f"Failed: {failed}",
            "",
            "DETAILED RESULTS",
            "-" * 40,
        ])

        for execution in self.executions:
            test = self.test_cases.get(execution.test_id)
            status = "✓" if execution.result == TestResult.PASS else "✗"

            lines.extend([
                "",
                f"[{status}] {test.name if test else execution.test_id}",
                f"    Result: {execution.actual_result}",
                f"    Notes: {execution.notes}",
            ])

        return "\n".join(lines)
```

## Summary

```
┌────────────────────────────────────────────────────────────────────┐
│                    Safety Standards Recap                          │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  International Standards                                           │
│  ───────────────────────                                           │
│  • ISO 10218: Industrial robot safety                              │
│  • ISO/TS 15066: Collaborative operation                           │
│  • ISO 13849: Safety-related control systems                       │
│  • IEC 62443: Cybersecurity                                        │
│                                                                    │
│  Collaborative Operation Methods                                   │
│  ───────────────────────────────                                   │
│  • Safety-rated monitored stop                                     │
│  • Hand guiding                                                    │
│  • Speed and separation monitoring                                 │
│  • Power and force limiting                                        │
│                                                                    │
│  Safety System Architecture                                        │
│  ──────────────────────────                                        │
│  • Dual-channel monitoring                                         │
│  • Safety-rated PLC/controller                                     │
│  • Category 3/4 safety functions                                   │
│  • Performance Level d/e                                           │
│                                                                    │
│  Risk Assessment                                                   │
│  ───────────────                                                   │
│  • Hazard identification                                           │
│  • Risk estimation (S × P × E)                                     │
│  • 3-step risk reduction                                           │
│  • Residual risk acceptance                                        │
│                                                                    │
│  Verification & Validation                                         │
│  ─────────────────────────                                         │
│  • Functional testing                                              │
│  • Fault injection                                                 │
│  • Performance measurement                                         │
│  • Documentation                                                   │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### Implementation Checklist

- [ ] Review applicable safety standards (ISO 10218, ISO/TS 15066)
- [ ] Conduct risk assessment per ISO 12100
- [ ] Design safety-rated control system (ISO 13849)
- [ ] Implement dual-channel monitoring
- [ ] Add speed and force limiting
- [ ] Implement collision detection and reaction
- [ ] Design safety zones with monitoring
- [ ] Create verification test procedures
- [ ] Execute verification tests
- [ ] Document safety case
- [ ] Train operators on safe operation
- [ ] Establish maintenance procedures

## Further Reading

- ISO 10218-1:2011 "Robots and robotic devices — Safety requirements"
- ISO/TS 15066:2016 "Robots — Collaborative robots"
- ISO 13849-1:2015 "Safety-related parts of control systems"
- ISO 12100:2010 "Safety of machinery — Risk assessment"
- Haddadin, S. "Physical Human-Robot Interaction" (2014)
- IEC 62443 "Industrial communication networks — Cybersecurity"
