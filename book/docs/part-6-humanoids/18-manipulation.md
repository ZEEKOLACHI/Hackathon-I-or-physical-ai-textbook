---
id: ch-6-18
title: Humanoid Manipulation
sidebar_position: 3
difficulty: advanced
estimated_time: 60
prerequisites: [ch-6-16, ch-6-17]
---

# Humanoid Manipulation

> "The hand is the visible part of the brain."
> — Immanuel Kant

Humanoid manipulation is where locomotion meets dexterity. Unlike fixed-base industrial manipulators, humanoid robots must coordinate their entire body while manipulating objects, maintaining balance while applying forces, and adapting to the dynamic nature of their base. This chapter explores the unique challenges and solutions for manipulation on humanoid platforms.

## The Humanoid Manipulation Challenge

```
              Humanoid vs Fixed-Base Manipulation

    Fixed-Base Robot                    Humanoid Robot
    ─────────────────                   ──────────────
                                              ┌───┐
         ┌────┐                               │   │ Head/Vision
         │    │ End-effector                  └─┬─┘
         ├────┤                                │
        /      \                           ┌───┴───┐
       │        │                          │       │ Torso
       │   ARM  │                          ├───┬───┤
       │        │                         /│   │   │\
        \      /                         / │   │   │ \
         │    │                         /  └───┴───┘  \
         │    │                        ARM          ARM
         │    │                         \            /
    ━━━━━│    │━━━━━━                    \    ▼    /
         │    │ Fixed base                └───┬───┘
    ═════╧════╧═════                         │ Pelvis
                                            / \
    Advantages:                            /   \ Legs
    ✓ Stable base                         ●     ●
    ✓ High precision                    Feet (Moving base!)
    ✓ High payload

    Challenges:                         Additional Challenges:
    • Limited workspace                 • Balance during manipulation
                                       • Floating base coordination
                                       • Reaction forces affect stance
                                       • Lower precision, more DOFs
```

### Key Differences from Industrial Manipulation

```python
"""
Humanoid Manipulation Fundamentals Module

Defines the key concepts and constraints unique to humanoid manipulation.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class ManipulationContext:
    """Context information for humanoid manipulation tasks."""

    # Base state (unlike fixed robots, this changes!)
    base_position: np.ndarray     # [x, y, z]
    base_orientation: np.ndarray  # Quaternion [w, x, y, z]
    base_velocity: np.ndarray     # [vx, vy, vz, wx, wy, wz]

    # Support state
    support_foot: str             # 'left', 'right', 'both'
    support_polygon: np.ndarray   # Vertices of support region

    # Balance margins
    com_position: np.ndarray      # Center of mass
    zmp_margin: float             # Distance to support edge [m]

    # Manipulation state
    left_hand_pose: np.ndarray    # 4x4 homogeneous transform
    right_hand_pose: np.ndarray   # 4x4 homogeneous transform

    def is_stable_for_manipulation(self,
                                    required_margin: float = 0.03) -> bool:
        """Check if balance state permits manipulation."""
        return self.zmp_margin >= required_margin


@dataclass
class ManipulationConstraints:
    """Physical constraints for humanoid manipulation."""

    # Workspace limits (relative to torso)
    max_reach: float = 0.80           # Maximum arm reach [m]
    min_reach: float = 0.20           # Minimum comfortable reach [m]

    # Force limits
    max_hand_force: float = 50.0      # Maximum hand force [N]
    max_hand_torque: float = 10.0     # Maximum hand torque [Nm]

    # Balance constraints
    min_zmp_margin: float = 0.02      # Minimum stability margin [m]
    max_com_shift: float = 0.10       # Maximum CoM shift during task [m]

    # Coordination constraints
    dual_arm_sync_tolerance: float = 0.01  # Max desync between arms [s]


class HumanoidManipulator:
    """
    Base class for humanoid manipulation control.

    Coordinates arm motion with whole-body balance.
    """

    def __init__(self,
                 robot_model,
                 balance_controller,
                 left_arm_controller,
                 right_arm_controller):
        """
        Initialize humanoid manipulator.

        Args:
            robot_model: Kinematic/dynamic model
            balance_controller: Whole-body balance controller
            left_arm_controller: Left arm controller
            right_arm_controller: Right arm controller
        """
        self.robot = robot_model
        self.balance = balance_controller
        self.left_arm = left_arm_controller
        self.right_arm = right_arm_controller
        self.constraints = ManipulationConstraints()

    def compute_reachable_workspace(self,
                                     context: ManipulationContext) -> np.ndarray:
        """
        Compute reachable workspace given current stance.

        The workspace depends on:
        - Current support configuration
        - Balance margins required
        - Arm kinematics

        Args:
            context: Current manipulation context

        Returns:
            Point cloud of reachable positions
        """
        workspace_points = []

        # Sample potential end-effector positions
        for x in np.linspace(-self.constraints.max_reach,
                             self.constraints.max_reach, 20):
            for y in np.linspace(-self.constraints.max_reach,
                                 self.constraints.max_reach, 20):
                for z in np.linspace(0, self.constraints.max_reach, 20):
                    point = np.array([x, y, z])

                    # Check arm kinematics
                    if not self._is_kinematically_reachable(point):
                        continue

                    # Check if reaching here maintains balance
                    if not self._maintains_balance(point, context):
                        continue

                    workspace_points.append(point)

        return np.array(workspace_points)

    def _is_kinematically_reachable(self, point: np.ndarray) -> bool:
        """Check if point is within arm kinematic limits."""
        distance = np.linalg.norm(point[:2])  # Horizontal distance

        if distance > self.constraints.max_reach:
            return False
        if distance < self.constraints.min_reach:
            return False

        # Check inverse kinematics solution exists
        ik_solution = self.robot.inverse_kinematics(point)
        return ik_solution is not None

    def _maintains_balance(self,
                          target: np.ndarray,
                          context: ManipulationContext) -> bool:
        """Check if reaching target maintains balance."""
        # Estimate CoM shift from arm motion
        arm_mass = self.robot.get_arm_mass()
        com_shift = arm_mass * target / self.robot.total_mass

        # Check if new CoM is within support
        new_com = context.com_position + com_shift
        new_margin = self._compute_support_margin(new_com,
                                                   context.support_polygon)

        return new_margin >= self.constraints.min_zmp_margin
```

## Workspace Analysis for Humanoids

Understanding the manipulation workspace is critical for task planning.

```
            Humanoid Manipulation Workspace

                    Top View
                    ────────

              ┌─────────────────┐
             /                   \
            /    Comfortable      \
           /       Zone           \
          │    ┌─────────┐         │
          │   ╱           ╲        │
         │   │   Optimal   │        │
         │   │    Zone     │        │
         │   │  ┌─────┐    │        │
         │   │  │     │    │        │
         │   │  │ ●   │    │        │   ← Robot
         │   │  │Torso│    │        │
         │   │  └─────┘    │        │
         │   │             │        │
          │   ╲           ╱        │
          │    └─────────┘         │
           \     Extended         /
            \      Zone          /
             \                  /
              └────────────────┘
                Maximum Reach


                Side View
                ─────────

            Maximum height
                  ▲
                  │    ╱───────╲
                  │   ╱ Extended ╲
        Shoulder ─┼──●───────────●── Comfortable
                  │   ╲ Optimal  ╱
                  │    ╲───────╱
                  │
        Waist ────┼─────────────────
                  │
                  │   Cannot reach
                  │   (blocked by body)
                  ▼
             Floor level
```

### Workspace Computation

```python
"""
Workspace Analysis Module

Computes and visualizes humanoid manipulation workspace.
"""

import numpy as np
from typing import Tuple, List, Dict
from dataclasses import dataclass
from enum import Enum


class WorkspaceRegion(Enum):
    """Classification of workspace regions."""
    OPTIMAL = "optimal"           # High dexterity, low effort
    COMFORTABLE = "comfortable"   # Sustainable for extended tasks
    EXTENDED = "extended"         # Reachable but fatiguing
    UNREACHABLE = "unreachable"   # Cannot reach


@dataclass
class WorkspacePoint:
    """A point in the manipulation workspace."""
    position: np.ndarray         # [x, y, z] relative to torso
    region: WorkspaceRegion      # Classification
    manipulability: float        # Manipulability index (0-1)
    balance_margin: float        # Stability margin when reaching here
    arm_config: str              # 'left', 'right', or 'either'


class WorkspaceAnalyzer:
    """
    Analyzes humanoid manipulation workspace.

    Considers arm kinematics, balance constraints, and
    task requirements.
    """

    def __init__(self, robot_model):
        """
        Initialize workspace analyzer.

        Args:
            robot_model: Robot kinematic model
        """
        self.robot = robot_model
        self.cache = {}

    def compute_workspace(self,
                          stance: str = 'double',
                          resolution: float = 0.05) -> List[WorkspacePoint]:
        """
        Compute full manipulation workspace.

        Args:
            stance: Support configuration ('double', 'left', 'right')
            resolution: Spatial resolution [m]

        Returns:
            List of workspace points with classifications
        """
        workspace = []

        # Define bounding box
        x_range = np.arange(-0.8, 0.8, resolution)
        y_range = np.arange(-0.8, 0.8, resolution)
        z_range = np.arange(0.0, 1.5, resolution)

        for x in x_range:
            for y in y_range:
                for z in z_range:
                    point = np.array([x, y, z])
                    wp = self._classify_point(point, stance)

                    if wp.region != WorkspaceRegion.UNREACHABLE:
                        workspace.append(wp)

        return workspace

    def _classify_point(self,
                        point: np.ndarray,
                        stance: str) -> WorkspacePoint:
        """Classify a workspace point."""

        # Check if reachable by either arm
        left_reachable = self._check_arm_reach(point, 'left')
        right_reachable = self._check_arm_reach(point, 'right')

        if not left_reachable and not right_reachable:
            return WorkspacePoint(
                position=point,
                region=WorkspaceRegion.UNREACHABLE,
                manipulability=0.0,
                balance_margin=0.0,
                arm_config='none'
            )

        # Compute manipulability
        manipulability = self._compute_manipulability(point)

        # Compute balance margin
        balance_margin = self._compute_balance_margin(point, stance)

        # Classify region
        if manipulability > 0.7 and balance_margin > 0.05:
            region = WorkspaceRegion.OPTIMAL
        elif manipulability > 0.4 and balance_margin > 0.02:
            region = WorkspaceRegion.COMFORTABLE
        else:
            region = WorkspaceRegion.EXTENDED

        # Determine arm configuration
        if left_reachable and right_reachable:
            arm_config = 'either'
        elif left_reachable:
            arm_config = 'left'
        else:
            arm_config = 'right'

        return WorkspacePoint(
            position=point,
            region=region,
            manipulability=manipulability,
            balance_margin=balance_margin,
            arm_config=arm_config
        )

    def _check_arm_reach(self, point: np.ndarray, arm: str) -> bool:
        """Check if arm can reach point."""
        # Get shoulder position
        shoulder = self.robot.get_shoulder_position(arm)

        # Distance from shoulder
        distance = np.linalg.norm(point - shoulder)

        # Check against arm length
        arm_length = self.robot.get_arm_length(arm)

        if distance > arm_length * 0.95:  # 95% of max reach
            return False
        if distance < arm_length * 0.2:   # Too close
            return False

        # Check IK solution exists
        ik_result = self.robot.compute_ik(point, arm)
        return ik_result.success

    def _compute_manipulability(self, point: np.ndarray) -> float:
        """
        Compute manipulability index at point.

        Manipulability measures the robot's ability to
        generate velocities in all directions.

        w = sqrt(det(J * J^T))
        """
        # Get Jacobian at this configuration
        J = self.robot.compute_jacobian_at_point(point)

        # Compute manipulability
        JJT = J @ J.T
        w = np.sqrt(max(0, np.linalg.det(JJT)))

        # Normalize to [0, 1]
        w_normalized = min(1.0, w / self.robot.max_manipulability)

        return w_normalized

    def _compute_balance_margin(self,
                                point: np.ndarray,
                                stance: str) -> float:
        """Compute stability margin when reaching point."""
        # Get support polygon
        support = self.robot.get_support_polygon(stance)

        # Estimate CoM shift from reaching
        com_shift = self._estimate_com_shift(point)

        # Compute margin from support polygon edge
        current_com = self.robot.get_com_position()
        new_com = current_com + com_shift

        margin = self._point_to_polygon_distance(new_com[:2], support)

        return margin


class DualArmWorkspace:
    """
    Analyzes workspace for dual-arm manipulation.

    Considers reachability constraints for both arms
    simultaneously.
    """

    def __init__(self, robot_model):
        self.robot = robot_model

    def compute_bimanual_workspace(self,
                                    relative_pose: np.ndarray = None
                                    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Compute workspace for bimanual manipulation.

        Args:
            relative_pose: Fixed relative pose between hands (optional)

        Returns:
            List of valid (left_pos, right_pos) pairs
        """
        valid_pairs = []

        # Sample left hand positions
        for left_pos in self._sample_positions():
            if not self._left_arm_reachable(left_pos):
                continue

            # For each left position, find valid right positions
            if relative_pose is not None:
                # Fixed relative pose
                right_pos = left_pos + relative_pose[:3, 3]
                if self._right_arm_reachable(right_pos):
                    if self._bimanual_feasible(left_pos, right_pos):
                        valid_pairs.append((left_pos, right_pos))
            else:
                # Free relative pose
                for right_pos in self._sample_positions():
                    if not self._right_arm_reachable(right_pos):
                        continue
                    if self._bimanual_feasible(left_pos, right_pos):
                        valid_pairs.append((left_pos, right_pos))

        return valid_pairs

    def _bimanual_feasible(self,
                           left_pos: np.ndarray,
                           right_pos: np.ndarray) -> bool:
        """Check if bimanual configuration is feasible."""
        # Check collision between arms
        if self._arms_collide(left_pos, right_pos):
            return False

        # Check combined balance
        if not self._combined_balance_ok(left_pos, right_pos):
            return False

        # Check joint limits for both arms simultaneously
        left_ik = self.robot.compute_ik(left_pos, 'left')
        right_ik = self.robot.compute_ik(right_pos, 'right')

        if not left_ik.success or not right_ik.success:
            return False

        return True
```

## Grasp Planning for Humanoids

```
            Grasp Planning Overview

    1. Object Analysis              2. Grasp Synthesis
    ──────────────────              ──────────────────

    ┌─────────────┐                   ━━━━━
    │             │                  ╱     ╲
    │   Object    │  ──────►       ●       ● Contact points
    │    Mesh     │                  ╲     ╱
    └─────────────┘                   ━━━━━
          │
          ▼
    • Surface normals               3. Grasp Evaluation
    • Friction properties           ──────────────────
    • Mass distribution
    • Symmetries                    ┌──────────────────┐
                                    │ Force Closure?   │
                                    │ Quality Score    │
                                    │ Kinematic Check  │
                                    │ Balance Check    │
                                    └──────────────────┘


            Force Closure Concept
            ─────────────────────

              Good Grasp                 Bad Grasp
              ──────────                 ─────────

              ─────●─────               ─────●─────
             │     │     │                   │
         ───►│  ●  │◄────              ───►  ●  ◄───
             │     │     │                   │
              ─────●─────

        Forces can resist             Cannot resist
        any external wrench           vertical forces
```

### Grasp Planning Implementation

```python
"""
Grasp Planning Module

Implements grasp synthesis and evaluation for humanoid hands.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class GraspType(Enum):
    """Common grasp types."""
    POWER = "power"           # Full hand wrap
    PRECISION = "precision"   # Fingertip grasp
    LATERAL = "lateral"       # Side pinch (key grasp)
    TRIPOD = "tripod"         # Thumb + 2 fingers
    SPHERICAL = "spherical"   # Curved hand around sphere
    CYLINDRICAL = "cylindrical"  # Wrap around cylinder


@dataclass
class Contact:
    """A contact point for grasping."""
    position: np.ndarray      # 3D position on object surface
    normal: np.ndarray        # Surface normal (into object)
    friction: float           # Friction coefficient
    finger: str               # Which finger makes this contact


@dataclass
class Grasp:
    """A complete grasp configuration."""
    contacts: List[Contact]
    grasp_type: GraspType
    approach_direction: np.ndarray
    hand_pose: np.ndarray     # 4x4 transform of hand frame
    finger_positions: np.ndarray  # Joint angles for fingers
    quality: float            # Grasp quality score


class GraspPlanner:
    """
    Plans grasps for humanoid manipulation.

    Considers hand kinematics, object geometry,
    and task requirements.
    """

    def __init__(self, hand_model):
        """
        Initialize grasp planner.

        Args:
            hand_model: Kinematic model of the hand
        """
        self.hand = hand_model
        self.min_contacts = 3  # Minimum for force closure

    def plan_grasp(self,
                   object_mesh,
                   task_type: str = 'pick',
                   preferred_type: GraspType = None) -> Optional[Grasp]:
        """
        Plan grasp for an object.

        Args:
            object_mesh: Object geometry
            task_type: 'pick', 'use', 'handover', etc.
            preferred_type: Preferred grasp type (optional)

        Returns:
            Best grasp configuration or None
        """
        # Step 1: Analyze object
        object_info = self._analyze_object(object_mesh)

        # Step 2: Generate grasp candidates
        candidates = self._generate_candidates(object_info, preferred_type)

        # Step 3: Evaluate and rank
        evaluated = []
        for candidate in candidates:
            quality = self._evaluate_grasp(candidate, object_info, task_type)
            if quality > 0:
                candidate.quality = quality
                evaluated.append(candidate)

        if not evaluated:
            return None

        # Return best grasp
        evaluated.sort(key=lambda g: g.quality, reverse=True)
        return evaluated[0]

    def _analyze_object(self, mesh) -> dict:
        """Analyze object geometry for grasp planning."""
        return {
            'centroid': self._compute_centroid(mesh),
            'principal_axes': self._compute_principal_axes(mesh),
            'bounding_box': self._compute_bounding_box(mesh),
            'surface_normals': self._sample_surface_normals(mesh),
            'curvature': self._estimate_curvature(mesh),
            'symmetry': self._detect_symmetry(mesh)
        }

    def _generate_candidates(self,
                             object_info: dict,
                             preferred_type: GraspType) -> List[Grasp]:
        """Generate grasp candidates based on object analysis."""
        candidates = []

        # Determine applicable grasp types
        if preferred_type:
            grasp_types = [preferred_type]
        else:
            grasp_types = self._select_grasp_types(object_info)

        for grasp_type in grasp_types:
            # Generate approach directions
            approaches = self._generate_approaches(object_info, grasp_type)

            for approach in approaches:
                # Sample contact configurations
                contacts = self._sample_contacts(
                    object_info, grasp_type, approach
                )

                for contact_set in contacts:
                    # Check hand kinematics
                    hand_config = self._solve_hand_ik(contact_set)

                    if hand_config is not None:
                        grasp = Grasp(
                            contacts=contact_set,
                            grasp_type=grasp_type,
                            approach_direction=approach,
                            hand_pose=hand_config['pose'],
                            finger_positions=hand_config['fingers'],
                            quality=0.0
                        )
                        candidates.append(grasp)

        return candidates

    def _evaluate_grasp(self,
                        grasp: Grasp,
                        object_info: dict,
                        task_type: str) -> float:
        """
        Evaluate grasp quality.

        Combines multiple quality metrics.
        """
        scores = {}

        # Force closure quality
        scores['force_closure'] = self._compute_force_closure_quality(
            grasp.contacts
        )

        if scores['force_closure'] < 0.1:
            return 0.0  # Not force closure

        # Manipulability
        scores['manipulability'] = self._compute_grasp_manipulability(grasp)

        # Task suitability
        scores['task_fit'] = self._evaluate_task_suitability(
            grasp, task_type
        )

        # Robustness to uncertainty
        scores['robustness'] = self._evaluate_robustness(grasp)

        # Weighted combination
        weights = {
            'force_closure': 0.4,
            'manipulability': 0.2,
            'task_fit': 0.25,
            'robustness': 0.15
        }

        quality = sum(w * scores[k] for k, w in weights.items())
        return quality

    def _compute_force_closure_quality(self,
                                        contacts: List[Contact]) -> float:
        """
        Compute force closure quality using grasp wrench space.

        Force closure means the grasp can resist arbitrary
        external wrenches.
        """
        if len(contacts) < self.min_contacts:
            return 0.0

        # Build grasp matrix
        G = self._build_grasp_matrix(contacts)

        # Compute wrench space
        W = self._compute_wrench_space(G, contacts)

        # Check if origin is inside convex hull of wrenches
        if not self._origin_in_convex_hull(W):
            return 0.0

        # Quality = radius of largest inscribed ball
        quality = self._largest_inscribed_ball_radius(W)

        return quality

    def _build_grasp_matrix(self, contacts: List[Contact]) -> np.ndarray:
        """
        Build the grasp matrix G.

        Maps contact forces to object wrench.
        """
        n_contacts = len(contacts)
        G = np.zeros((6, 3 * n_contacts))

        for i, contact in enumerate(contacts):
            # Force contribution
            G[:3, 3*i:3*i+3] = np.eye(3)

            # Torque contribution (r × f)
            r = contact.position  # Relative to object center
            G[3:6, 3*i:3*i+3] = self._skew_symmetric(r)

        return G

    def _skew_symmetric(self, v: np.ndarray) -> np.ndarray:
        """Create skew-symmetric matrix for cross product."""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])


class ForceClosureAnalyzer:
    """
    Detailed force closure analysis for grasps.
    """

    def __init__(self, friction_model: str = 'coulomb'):
        """
        Initialize analyzer.

        Args:
            friction_model: 'coulomb' or 'soft_finger'
        """
        self.friction_model = friction_model
        self.cone_edges = 8  # Discretization of friction cone

    def compute_wrench_space(self, grasp: Grasp) -> np.ndarray:
        """
        Compute the grasp wrench space.

        The wrench space is the set of all wrenches that
        can be applied to the object through the grasp.
        """
        wrenches = []

        for contact in grasp.contacts:
            # Get friction cone for this contact
            cone = self._discretize_friction_cone(contact)

            for force in cone:
                # Compute wrench from this force
                wrench = self._force_to_wrench(force, contact.position)
                wrenches.append(wrench)

        return np.array(wrenches)

    def _discretize_friction_cone(self, contact: Contact) -> List[np.ndarray]:
        """
        Discretize the friction cone into edge vectors.

        Friction cone: f_tangent <= μ * f_normal
        """
        mu = contact.friction
        normal = contact.normal

        # Create tangent basis
        if abs(normal[2]) < 0.9:
            t1 = np.cross(normal, np.array([0, 0, 1]))
        else:
            t1 = np.cross(normal, np.array([1, 0, 0]))
        t1 = t1 / np.linalg.norm(t1)
        t2 = np.cross(normal, t1)

        # Sample cone edges
        cone_edges = []
        for i in range(self.cone_edges):
            angle = 2 * np.pi * i / self.cone_edges
            tangent = np.cos(angle) * t1 + np.sin(angle) * t2

            # Edge of friction cone
            edge = normal + mu * tangent
            edge = edge / np.linalg.norm(edge)
            cone_edges.append(edge)

        return cone_edges

    def is_force_closure(self, grasp: Grasp) -> bool:
        """
        Check if grasp achieves force closure.

        Force closure iff origin is strictly inside
        the convex hull of the wrench space.
        """
        W = self.compute_wrench_space(grasp)

        # Use linear programming to check
        # If we can express 0 as a positive combination of wrenches
        return self._origin_in_convex_hull(W)

    def compute_quality_metrics(self, grasp: Grasp) -> dict:
        """
        Compute various grasp quality metrics.
        """
        W = self.compute_wrench_space(grasp)

        return {
            # Largest inscribed ball radius (force closure margin)
            'epsilon': self._largest_ball_radius(W),

            # Volume of wrench space
            'volume': self._wrench_space_volume(W),

            # Isotropy (uniformity of wrench capability)
            'isotropy': self._grasp_isotropy(W),

            # Minimum singular value of grasp matrix
            'min_svd': self._min_singular_value(grasp)
        }
```

## Dual-Arm Coordination

Humanoid robots can perform complex tasks using both arms together.

```
              Dual-Arm Coordination Modes

    1. Independent                  2. Symmetric
    ──────────────                  ────────────

        ○                               ○
       ╱│╲                             ╱│╲
      ╱ │ ╲                           ╱ │ ╲
     ●  │  ●                         ●  │  ●
    ↙   │   ↘                       ↑   │   ↑
   A    │    B                      Same motion mirrored
        │
    Different tasks               Mirror (e.g., lifting)


    3. Coordinated                 4. Leader-Follower
    ──────────────                 ─────────────────

        ○                               ○
       ╱│╲                             ╱│╲
      ╱ │ ╲                           ╱ │ ╲
     ●──┼──●                         ●──┼──●
      ╲ │ ╱                          L  │  F
       ╲│╱                              │
        ▼                               │
    Fixed relative                 Right follows left
    pose (carrying)                (teaching mode)
```

### Dual-Arm Controller

```python
"""
Dual-Arm Coordination Module

Implements various coordination strategies for bimanual manipulation.
"""

import numpy as np
from typing import Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class CoordinationMode(Enum):
    """Dual-arm coordination modes."""
    INDEPENDENT = "independent"       # Arms move independently
    SYMMETRIC = "symmetric"           # Mirror motion
    RELATIVE = "relative"             # Fixed relative pose
    LEADER_FOLLOWER = "leader_follower"  # One arm leads
    COOPERATIVE = "cooperative"       # Shared object manipulation


@dataclass
class DualArmCommand:
    """Command for both arms."""
    left_pose: np.ndarray         # 4x4 target pose for left
    right_pose: np.ndarray        # 4x4 target pose for right
    left_wrench: np.ndarray       # Force/torque for left
    right_wrench: np.ndarray      # Force/torque for right
    mode: CoordinationMode


class DualArmController:
    """
    Controls coordinated motion of both arms.

    Handles various coordination modes and constraint
    satisfaction between the arms.
    """

    def __init__(self,
                 left_arm,
                 right_arm,
                 robot_model):
        """
        Initialize dual-arm controller.

        Args:
            left_arm: Left arm controller
            right_arm: Right arm controller
            robot_model: Full robot model for coordination
        """
        self.left = left_arm
        self.right = right_arm
        self.robot = robot_model
        self.mode = CoordinationMode.INDEPENDENT

        # Relative pose constraint (for RELATIVE mode)
        self.relative_pose = np.eye(4)

        # Synchronization tolerance
        self.sync_tolerance = 0.01  # seconds

    def set_mode(self, mode: CoordinationMode):
        """Set coordination mode."""
        self.mode = mode

        if mode == CoordinationMode.RELATIVE:
            # Compute current relative pose
            left_pose = self.left.get_pose()
            right_pose = self.right.get_pose()
            self.relative_pose = np.linalg.inv(left_pose) @ right_pose

    def compute_command(self,
                        target: np.ndarray,
                        mode: CoordinationMode = None) -> DualArmCommand:
        """
        Compute dual-arm command to reach target.

        Args:
            target: Target specification (depends on mode)
            mode: Coordination mode (uses current if None)

        Returns:
            Command for both arms
        """
        if mode is None:
            mode = self.mode

        if mode == CoordinationMode.INDEPENDENT:
            return self._independent_command(target)

        elif mode == CoordinationMode.SYMMETRIC:
            return self._symmetric_command(target)

        elif mode == CoordinationMode.RELATIVE:
            return self._relative_command(target)

        elif mode == CoordinationMode.LEADER_FOLLOWER:
            return self._leader_follower_command(target)

        elif mode == CoordinationMode.COOPERATIVE:
            return self._cooperative_command(target)

    def _independent_command(self,
                             targets: Tuple[np.ndarray, np.ndarray]
                             ) -> DualArmCommand:
        """Independent motion for each arm."""
        left_target, right_target = targets

        return DualArmCommand(
            left_pose=left_target,
            right_pose=right_target,
            left_wrench=np.zeros(6),
            right_wrench=np.zeros(6),
            mode=CoordinationMode.INDEPENDENT
        )

    def _symmetric_command(self,
                           left_target: np.ndarray) -> DualArmCommand:
        """
        Symmetric (mirrored) motion.

        Right arm mirrors left arm about the sagittal plane.
        """
        # Mirror matrix (flip Y axis)
        mirror = np.diag([1, -1, 1, 1])

        # Compute mirrored right target
        right_target = mirror @ left_target @ mirror

        return DualArmCommand(
            left_pose=left_target,
            right_pose=right_target,
            left_wrench=np.zeros(6),
            right_wrench=np.zeros(6),
            mode=CoordinationMode.SYMMETRIC
        )

    def _relative_command(self,
                          left_target: np.ndarray) -> DualArmCommand:
        """
        Maintain fixed relative pose between hands.

        Used for carrying objects with both hands.
        """
        # Right pose determined by relative constraint
        right_target = left_target @ self.relative_pose

        return DualArmCommand(
            left_pose=left_target,
            right_pose=right_target,
            left_wrench=np.zeros(6),
            right_wrench=np.zeros(6),
            mode=CoordinationMode.RELATIVE
        )

    def _cooperative_command(self,
                             object_target: np.ndarray) -> DualArmCommand:
        """
        Cooperative manipulation of shared object.

        Both arms apply forces to move object to target.
        """
        # Current object pose (from both hand poses)
        object_pose = self._estimate_object_pose()

        # Compute required object wrench
        object_wrench = self._compute_object_wrench(
            object_pose, object_target
        )

        # Distribute wrench between arms
        left_wrench, right_wrench = self._distribute_wrench(object_wrench)

        # Compute arm poses for object target
        left_pose = object_target @ self.left_grasp_offset
        right_pose = object_target @ self.right_grasp_offset

        return DualArmCommand(
            left_pose=left_pose,
            right_pose=right_pose,
            left_wrench=left_wrench,
            right_wrench=right_wrench,
            mode=CoordinationMode.COOPERATIVE
        )

    def _distribute_wrench(self,
                           object_wrench: np.ndarray
                           ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Distribute object wrench between both arms.

        Uses optimization to minimize internal forces.
        """
        # Build grasp matrix for both hands
        G = self._build_dual_arm_grasp_matrix()

        # Solve for minimum norm contact forces
        # G @ f = w, minimize ||f||
        G_pinv = np.linalg.pinv(G)
        f = G_pinv @ object_wrench

        # Split forces between arms
        n = len(f) // 2
        left_wrench = f[:n]
        right_wrench = f[n:]

        return left_wrench, right_wrench


class BimanualManipulation:
    """
    High-level bimanual manipulation primitives.
    """

    def __init__(self, dual_arm_controller: DualArmController):
        self.controller = dual_arm_controller

    def lift_object(self,
                    object_pose: np.ndarray,
                    lift_height: float) -> bool:
        """
        Lift object using both hands.

        Args:
            object_pose: Current object pose
            lift_height: How high to lift [m]

        Returns:
            Success flag
        """
        # Set relative coordination mode
        self.controller.set_mode(CoordinationMode.RELATIVE)

        # Compute target pose
        target_pose = object_pose.copy()
        target_pose[2, 3] += lift_height

        # Execute lift
        command = self.controller.compute_command(target_pose)
        return self._execute_command(command)

    def handover(self,
                 from_arm: str,
                 to_arm: str,
                 object_pose: np.ndarray) -> bool:
        """
        Hand over object between arms.

        Args:
            from_arm: 'left' or 'right'
            to_arm: 'left' or 'right'
            object_pose: Current object pose

        Returns:
            Success flag
        """
        # Step 1: Move receiving arm to grasp position
        grasp_pose = self._compute_handover_grasp(object_pose, to_arm)

        # Step 2: Close receiving hand

        # Step 3: Open giving hand

        # Step 4: Retract giving arm

        return True

    def rotate_object(self,
                      object_pose: np.ndarray,
                      rotation: np.ndarray) -> bool:
        """
        Rotate object in place using regrasping.

        For rotations beyond single-grasp limits.
        """
        # Check if rotation requires regrasping
        rotation_angle = self._rotation_angle(rotation)

        if rotation_angle < np.pi / 4:
            # Small rotation - single grasp
            target = object_pose @ rotation
            command = self.controller.compute_command(target)
            return self._execute_command(command)
        else:
            # Large rotation - regrasp sequence
            return self._execute_regrasp_rotation(object_pose, rotation)
```

## Mobile Manipulation

Humanoid robots must coordinate base motion with arm manipulation.

```
            Mobile Manipulation Workspace Extension

    Fixed Base                      With Base Motion
    ──────────                      ────────────────

         ╭───────╮                     ╭─────────────────────────╮
        ╱         ╲                   ╱                           ╲
       │           │                 │                             │
       │  Arm      │                 │    Extended Workspace       │
       │ Workspace │                 │                             │
       │     ●     │                 │  ╭───────╮    ╭───────╮    │
       │  Robot    │     ────►       │ ╱         ╲  ╱         ╲   │
        ╲         ╱                  │ │   ●───────●   │         │
         ╰───────╯                   │  ╲ Start   ╱  ╲  End   ╱   │
                                      ╲   ╰───────╯    ╰───────╯  ╱
                                       ╰─────────────────────────╯

                                    Base movement enables reaching
                                    targets beyond arm workspace
```

### Mobile Manipulation Controller

```python
"""
Mobile Manipulation Module

Coordinates base locomotion with arm manipulation.
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class MobileManipulationGoal:
    """Goal specification for mobile manipulation."""
    target_pose: np.ndarray       # Desired end-effector pose
    arm: str = 'right'            # Which arm to use
    approach_distance: float = 0.5  # Distance for approach
    base_tolerance: float = 0.1   # Base positioning tolerance


class MobileManipulator:
    """
    Coordinates locomotion and manipulation.

    Decides when to move base vs. when to use arm.
    """

    def __init__(self,
                 robot_model,
                 locomotion_controller,
                 arm_controller,
                 balance_controller):
        """
        Initialize mobile manipulator.

        Args:
            robot_model: Full robot kinematic model
            locomotion_controller: Walking/base motion controller
            arm_controller: Arm manipulation controller
            balance_controller: Balance/whole-body controller
        """
        self.robot = robot_model
        self.locomotion = locomotion_controller
        self.arm = arm_controller
        self.balance = balance_controller

    def reach_target(self,
                     goal: MobileManipulationGoal) -> bool:
        """
        Reach manipulation target, moving base if needed.

        Args:
            goal: Target specification

        Returns:
            Success flag
        """
        # Check if target is reachable without base motion
        if self._is_reachable_from_current(goal):
            return self._execute_arm_motion(goal)

        # Need to move base
        # Step 1: Plan base position
        base_pose = self._plan_base_position(goal)

        if base_pose is None:
            return False  # Cannot find valid base position

        # Step 2: Move base
        success = self._move_base(base_pose)

        if not success:
            return False

        # Step 3: Execute arm motion
        return self._execute_arm_motion(goal)

    def _is_reachable_from_current(self,
                                    goal: MobileManipulationGoal) -> bool:
        """Check if target is reachable without base motion."""
        # Get current base pose
        base_pose = self.robot.get_base_pose()

        # Transform target to base frame
        target_in_base = np.linalg.inv(base_pose) @ goal.target_pose

        # Check arm workspace
        arm_reach = self.robot.get_arm_reach(goal.arm)
        target_distance = np.linalg.norm(target_in_base[:3, 3])

        if target_distance > arm_reach:
            return False

        # Check IK solution
        ik_result = self.robot.compute_arm_ik(
            goal.target_pose, goal.arm
        )

        return ik_result.success

    def _plan_base_position(self,
                            goal: MobileManipulationGoal
                            ) -> Optional[np.ndarray]:
        """
        Plan base position for manipulation.

        Finds a base pose that puts target in arm workspace.
        """
        target_pos = goal.target_pose[:3, 3]

        # Optimal distance from target
        optimal_distance = self.robot.get_arm_reach(goal.arm) * 0.7

        # Sample base positions around target
        candidates = []

        for angle in np.linspace(0, 2*np.pi, 36):
            # Position base at optimal distance
            offset = optimal_distance * np.array([
                np.cos(angle), np.sin(angle), 0
            ])
            base_pos = target_pos - offset

            # Orient base toward target
            base_yaw = angle + np.pi
            base_pose = self._pose_from_position_yaw(base_pos, base_yaw)

            # Check validity
            if self._is_valid_base_pose(base_pose):
                # Score by manipulation quality
                score = self._score_base_pose(base_pose, goal)
                candidates.append((score, base_pose))

        if not candidates:
            return None

        # Return best candidate
        candidates.sort(reverse=True)
        return candidates[0][1]

    def _score_base_pose(self,
                         base_pose: np.ndarray,
                         goal: MobileManipulationGoal) -> float:
        """
        Score a base pose for manipulation task.

        Higher score = better base position.
        """
        # Transform target to base frame
        target_in_base = np.linalg.inv(base_pose) @ goal.target_pose
        target_pos = target_in_base[:3, 3]

        scores = {}

        # Distance from optimal reach
        distance = np.linalg.norm(target_pos[:2])
        optimal = self.robot.get_arm_reach(goal.arm) * 0.7
        scores['distance'] = 1.0 - abs(distance - optimal) / optimal

        # Target in front of robot (preferred)
        forward_component = target_pos[0]
        scores['forward'] = max(0, forward_component / distance)

        # Manipulability at this configuration
        scores['manipulability'] = self._compute_manipulability(
            base_pose, goal.target_pose, goal.arm
        )

        # Balance margin
        scores['balance'] = self._compute_balance_margin(base_pose)

        # Weighted sum
        weights = {
            'distance': 0.3,
            'forward': 0.2,
            'manipulability': 0.3,
            'balance': 0.2
        }

        return sum(w * scores[k] for k, w in weights.items())

    def _move_base(self, target_pose: np.ndarray) -> bool:
        """Move base to target pose using locomotion."""
        # Plan footsteps
        current_pose = self.robot.get_base_pose()
        footsteps = self.locomotion.plan_footsteps(
            current_pose, target_pose
        )

        if footsteps is None:
            return False

        # Execute walking
        for step in footsteps:
            success = self.locomotion.execute_step(step)
            if not success:
                return False

        return True


class WholeBodyManipulation:
    """
    Whole-body manipulation using all robot DOFs.

    Coordinates legs, torso, and arms for manipulation
    tasks beyond arm-only capabilities.
    """

    def __init__(self, robot_model, balance_controller):
        self.robot = robot_model
        self.balance = balance_controller

    def compute_whole_body_ik(self,
                              ee_target: np.ndarray,
                              arm: str,
                              constraints: dict = None) -> Optional[np.ndarray]:
        """
        Compute whole-body IK for manipulation.

        Uses torso and legs to extend arm workspace
        while maintaining balance.

        Args:
            ee_target: End-effector target pose
            arm: Which arm
            constraints: Additional constraints

        Returns:
            Full robot configuration or None
        """
        # Initialize with current configuration
        q = self.robot.get_configuration()

        # Iterative IK with prioritized tasks
        for iteration in range(100):
            # Task 1 (highest): Balance constraint
            balance_task = self._compute_balance_task(q)

            # Task 2: End-effector pose
            ee_task = self._compute_ee_task(q, ee_target, arm)

            # Task 3 (lowest): Posture regularization
            posture_task = self._compute_posture_task(q)

            # Hierarchical task execution
            dq = self._hierarchical_ik(
                q, [balance_task, ee_task, posture_task]
            )

            # Update configuration
            q = q + dq

            # Check convergence
            if np.linalg.norm(ee_task.error) < 1e-3:
                return q

        return None  # Failed to converge

    def _hierarchical_ik(self,
                         q: np.ndarray,
                         tasks: List) -> np.ndarray:
        """
        Solve hierarchical IK using null-space projection.

        Higher priority tasks are satisfied first.
        """
        n_dof = len(q)
        dq = np.zeros(n_dof)

        # Null-space projector (starts as identity)
        N = np.eye(n_dof)

        for task in tasks:
            # Jacobian for this task
            J = task.jacobian

            # Project Jacobian into null space of higher tasks
            J_proj = J @ N

            # Solve for this task
            J_pinv = np.linalg.pinv(J_proj)
            dq_task = J_pinv @ task.error

            # Add contribution
            dq = dq + N @ dq_task

            # Update null-space projector
            N = N @ (np.eye(n_dof) - np.linalg.pinv(J_proj) @ J_proj)

        return dq
```

## Force-Controlled Manipulation

Humanoid manipulation often requires precise force control for contact tasks.

```
            Force Control in Manipulation

    Position Control               Force Control
    ────────────────               ─────────────

        Target                         Target
           ↓                              ↓
    ┌──────────────┐              ┌──────────────┐
    │   Position   │              │    Force     │
    │  Controller  │              │  Controller  │
    └──────┬───────┘              └──────┬───────┘
           │                             │
           ▼                             ▼
      ┌────────┐                    ┌────────┐
      │ Robot  │                    │ Robot  │
      └────┬───┘                    └────┬───┘
           │                             │
           ▼                             ▼
    ━━━━━━━━━━━━━━━━              ━━━━━━━━━━━━━━━━

    Collision = High force         Controlled contact
    (bad!)                         (good!)


          Hybrid Position/Force Control
          ──────────────────────────────

              ┌─────────────────────────┐
              │     Task Frame T        │
              │                         │
              │    z ↑                  │
              │      │    y             │
              │      │   ╱              │
              │      │  ╱               │
              │      │ ╱                │
              │      └──────► x         │
              │                         │
              │  x: Force control       │
              │  y: Position control    │
              │  z: Force control       │
              └─────────────────────────┘

    Example: Wiping a surface
    - Control force into surface (z)
    - Control position along surface (x, y)
```

### Hybrid Force/Position Controller

```python
"""
Force-Controlled Manipulation Module

Implements hybrid position/force control for contact tasks.
"""

import numpy as np
from typing import Optional, Callable
from dataclasses import dataclass


@dataclass
class HybridControlConfig:
    """Configuration for hybrid controller."""

    # Selection matrix (1 = force control, 0 = position control)
    selection: np.ndarray   # 6x6 diagonal matrix

    # Position control gains
    kp_pos: np.ndarray      # 6D proportional gains
    kd_pos: np.ndarray      # 6D derivative gains

    # Force control gains
    kp_force: np.ndarray    # 6D proportional gains
    ki_force: np.ndarray    # 6D integral gains

    # Limits
    max_force: float = 50.0
    max_velocity: float = 0.5


class HybridForcePositionController:
    """
    Hybrid force/position controller for manipulation.

    Controls position in some task-space directions
    and force in others.
    """

    def __init__(self,
                 robot_model,
                 config: HybridControlConfig):
        """
        Initialize hybrid controller.

        Args:
            robot_model: Robot kinematic model
            config: Controller configuration
        """
        self.robot = robot_model
        self.config = config

        # Selection matrices
        self.S_f = config.selection  # Force-controlled directions
        self.S_p = np.eye(6) - config.selection  # Position-controlled

        # Integral term for force control
        self.force_integral = np.zeros(6)

    def compute_command(self,
                        current_pose: np.ndarray,
                        current_velocity: np.ndarray,
                        current_force: np.ndarray,
                        target_pose: np.ndarray,
                        target_force: np.ndarray) -> np.ndarray:
        """
        Compute control command.

        Args:
            current_pose: Current end-effector pose (4x4)
            current_velocity: Current velocity (6D twist)
            current_force: Measured force/torque (6D wrench)
            target_pose: Desired pose (4x4)
            target_force: Desired force (6D wrench)

        Returns:
            Commanded velocity (6D twist)
        """
        # Position error
        pose_error = self._compute_pose_error(current_pose, target_pose)

        # Force error
        force_error = target_force - current_force

        # Update force integral
        self.force_integral += force_error * self.dt
        self.force_integral = np.clip(
            self.force_integral, -10, 10
        )  # Anti-windup

        # Position control component
        v_position = (
            self.config.kp_pos * pose_error -
            self.config.kd_pos * current_velocity
        )

        # Force control component
        v_force = (
            self.config.kp_force * force_error +
            self.config.ki_force * self.force_integral
        )

        # Combine using selection matrices
        v_command = self.S_p @ v_position + self.S_f @ v_force

        # Apply velocity limits
        v_command = self._limit_velocity(v_command)

        return v_command

    def _compute_pose_error(self,
                            current: np.ndarray,
                            target: np.ndarray) -> np.ndarray:
        """Compute pose error as 6D vector."""
        # Position error
        pos_error = target[:3, 3] - current[:3, 3]

        # Orientation error (axis-angle)
        R_error = target[:3, :3] @ current[:3, :3].T
        angle = np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1, 1))

        if angle < 1e-6:
            rot_error = np.zeros(3)
        else:
            axis = np.array([
                R_error[2, 1] - R_error[1, 2],
                R_error[0, 2] - R_error[2, 0],
                R_error[1, 0] - R_error[0, 1]
            ]) / (2 * np.sin(angle))
            rot_error = angle * axis

        return np.concatenate([pos_error, rot_error])


class ImpedanceController:
    """
    Impedance controller for compliant manipulation.

    Makes the robot behave like a mass-spring-damper system.
    """

    def __init__(self,
                 mass: np.ndarray,
                 damping: np.ndarray,
                 stiffness: np.ndarray):
        """
        Initialize impedance controller.

        Implements: M*a + D*v + K*x = F_ext

        Args:
            mass: Desired inertia (6x6)
            damping: Desired damping (6x6)
            stiffness: Desired stiffness (6x6)
        """
        self.M = mass
        self.D = damping
        self.K = stiffness

    def compute_command(self,
                        current_pose: np.ndarray,
                        current_velocity: np.ndarray,
                        target_pose: np.ndarray,
                        external_force: np.ndarray) -> np.ndarray:
        """
        Compute torque command for impedance behavior.

        Args:
            current_pose: Current end-effector pose
            current_velocity: Current velocity
            target_pose: Equilibrium pose
            external_force: Measured external force

        Returns:
            Joint torques
        """
        # Pose error from equilibrium
        x_error = self._compute_pose_error(current_pose, target_pose)

        # Desired acceleration from impedance model
        # M*a = F_ext - D*v - K*x
        # a = M^(-1) * (F_ext - D*v - K*x)

        a_desired = np.linalg.solve(
            self.M,
            external_force - self.D @ current_velocity - self.K @ x_error
        )

        # Convert to joint torques using dynamics
        tau = self._operational_space_control(a_desired)

        return tau

    def set_stiffness(self, stiffness: np.ndarray):
        """Update stiffness (for variable impedance)."""
        self.K = stiffness

        # Update damping for critical damping
        # D = 2 * sqrt(M * K)
        self.D = 2 * np.sqrt(self.M @ stiffness)


class AdmittanceController:
    """
    Admittance controller for force-guided motion.

    Converts measured forces to motion commands.
    F → x (opposite of impedance)
    """

    def __init__(self,
                 virtual_mass: np.ndarray,
                 virtual_damping: np.ndarray):
        """
        Initialize admittance controller.

        Args:
            virtual_mass: Virtual inertia
            virtual_damping: Virtual damping
        """
        self.M = virtual_mass
        self.D = virtual_damping
        self.velocity = np.zeros(6)

    def compute_velocity(self,
                         measured_force: np.ndarray,
                         dt: float) -> np.ndarray:
        """
        Compute velocity from measured force.

        Simulates: M*a + D*v = F

        Args:
            measured_force: Measured external force
            dt: Time step

        Returns:
            Commanded velocity
        """
        # Compute acceleration
        acceleration = np.linalg.solve(
            self.M,
            measured_force - self.D @ self.velocity
        )

        # Integrate to get velocity
        self.velocity += acceleration * dt

        return self.velocity
```

## Manipulation Skills

Pre-defined manipulation skills enable rapid task execution.

```python
"""
Manipulation Skills Module

Library of reusable manipulation primitives.
"""

import numpy as np
from typing import Optional, List, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod


class ManipulationSkill(ABC):
    """Base class for manipulation skills."""

    @abstractmethod
    def preconditions(self, context) -> bool:
        """Check if skill can be executed."""
        pass

    @abstractmethod
    def execute(self, context) -> bool:
        """Execute the skill."""
        pass

    @abstractmethod
    def postconditions(self, context) -> bool:
        """Check if skill succeeded."""
        pass


class PickSkill(ManipulationSkill):
    """Skill for picking up objects."""

    def __init__(self,
                 arm_controller,
                 grasp_planner,
                 hand_controller):
        self.arm = arm_controller
        self.grasp_planner = grasp_planner
        self.hand = hand_controller

    def preconditions(self, context) -> bool:
        """Check picking preconditions."""
        # Object must be detected
        if context.target_object is None:
            return False

        # Hand must be empty
        if context.held_object is not None:
            return False

        # Object must be reachable
        if not self._is_reachable(context.target_object.pose):
            return False

        return True

    def execute(self, context) -> bool:
        """Execute pick sequence."""
        object_pose = context.target_object.pose
        object_mesh = context.target_object.mesh

        # Step 1: Plan grasp
        grasp = self.grasp_planner.plan_grasp(object_mesh)
        if grasp is None:
            return False

        # Step 2: Compute approach pose
        approach_pose = self._compute_approach(object_pose, grasp)

        # Step 3: Move to approach pose
        success = self.arm.move_to(approach_pose)
        if not success:
            return False

        # Step 4: Open hand
        self.hand.open()

        # Step 5: Move to grasp pose
        grasp_pose = object_pose @ grasp.hand_pose
        success = self.arm.move_to(grasp_pose)
        if not success:
            return False

        # Step 6: Close hand
        self.hand.close(grasp.finger_positions)

        # Step 7: Verify grasp
        if not self.hand.is_grasping():
            return False

        # Step 8: Lift
        lift_pose = grasp_pose.copy()
        lift_pose[2, 3] += 0.1  # Lift 10cm
        success = self.arm.move_to(lift_pose)

        return success

    def postconditions(self, context) -> bool:
        """Verify pick succeeded."""
        return self.hand.is_grasping()


class PlaceSkill(ManipulationSkill):
    """Skill for placing objects."""

    def __init__(self, arm_controller, hand_controller):
        self.arm = arm_controller
        self.hand = hand_controller

    def preconditions(self, context) -> bool:
        # Must be holding something
        if not self.hand.is_grasping():
            return False

        # Target location must be reachable
        if not self._is_reachable(context.place_target):
            return False

        # Target must be clear
        if context.is_occupied(context.place_target):
            return False

        return True

    def execute(self, context) -> bool:
        """Execute place sequence."""
        target = context.place_target

        # Step 1: Compute approach from above
        approach = target.copy()
        approach[2, 3] += 0.15  # 15cm above

        # Step 2: Move to approach
        success = self.arm.move_to(approach)
        if not success:
            return False

        # Step 3: Lower to place pose
        success = self.arm.move_to(target)
        if not success:
            return False

        # Step 4: Open hand
        self.hand.open()

        # Step 5: Retract
        success = self.arm.move_to(approach)

        return success


class WipeSkill(ManipulationSkill):
    """Skill for wiping surfaces (force-controlled)."""

    def __init__(self,
                 arm_controller,
                 force_controller):
        self.arm = arm_controller
        self.force = force_controller

    def execute(self, context) -> bool:
        """Execute wiping motion with force control."""
        surface = context.surface
        pattern = context.wipe_pattern  # List of waypoints
        target_force = context.target_force  # Normal force

        # Configure hybrid controller for wiping
        # Force control normal to surface, position along surface
        self.force.configure_for_surface(surface.normal)

        # Move to start with approach
        start = pattern[0]
        approach = start - 0.05 * surface.normal
        self.arm.move_to(approach)

        # Engage surface with force control
        self.force.set_target_force(target_force * surface.normal)

        # Follow wiping pattern
        for waypoint in pattern:
            self.force.set_target_position(waypoint)
            self.force.wait_until_reached()

        # Disengage
        self.force.set_target_force(np.zeros(6))
        self.arm.move_to(approach)

        return True


class SkillLibrary:
    """
    Library of manipulation skills.

    Provides skill selection and sequencing.
    """

    def __init__(self):
        self.skills = {}

    def register(self, name: str, skill: ManipulationSkill):
        """Register a skill."""
        self.skills[name] = skill

    def get_skill(self, name: str) -> Optional[ManipulationSkill]:
        """Get skill by name."""
        return self.skills.get(name)

    def find_applicable_skills(self, context) -> List[str]:
        """Find all skills whose preconditions are satisfied."""
        applicable = []

        for name, skill in self.skills.items():
            if skill.preconditions(context):
                applicable.append(name)

        return applicable
```

## Summary

```
┌────────────────────────────────────────────────────────────────────┐
│                 Humanoid Manipulation Recap                        │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Key Differences from Fixed-Base                                   │
│  ───────────────────────────────                                   │
│  • Floating base requires balance coordination                     │
│  • Workspace depends on stance configuration                       │
│  • Reaction forces affect whole-body stability                     │
│  • Lower precision but greater flexibility                         │
│                                                                    │
│  Workspace Analysis                                                │
│  ──────────────────                                                │
│  • Optimal zone: High manipulability + good balance                │
│  • Extended zone: Reachable but less dexterous                     │
│  • Bimanual workspace: Both arms can reach                         │
│                                                                    │
│  Grasp Planning                                                    │
│  ──────────────                                                    │
│  • Force closure ensures stable grasps                             │
│  • Quality metrics: ε (margin), volume, isotropy                   │
│  • Consider task requirements in grasp selection                   │
│                                                                    │
│  Dual-Arm Coordination                                             │
│  ─────────────────────                                             │
│  • Independent: Different tasks each arm                           │
│  • Symmetric: Mirrored motion                                      │
│  • Relative: Fixed pose constraint (carrying)                      │
│  • Cooperative: Shared object manipulation                         │
│                                                                    │
│  Mobile Manipulation                                               │
│  ───────────────────                                               │
│  • Extends workspace through base motion                           │
│  • Plan base position for manipulation quality                     │
│  • Whole-body IK uses all DOFs                                     │
│                                                                    │
│  Force Control                                                     │
│  ─────────────                                                     │
│  • Hybrid: Position + force in different directions                │
│  • Impedance: Robot acts as spring-damper                          │
│  • Admittance: Force guides motion                                 │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### Key Equations Reference

| Concept | Equation |
|---------|----------|
| Manipulability | w = √det(J·Jᵀ) |
| Force closure | G·f = w, f ≥ 0 |
| Grasp quality | ε = min distance to wrench space boundary |
| Impedance | M·ẍ + D·ẋ + K·x = Fext |
| Hybrid control | v = Sp·vpos + Sf·vforce |

### Implementation Checklist

- [ ] Implement workspace analysis considering balance
- [ ] Build grasp planner with force closure evaluation
- [ ] Create dual-arm coordination modes
- [ ] Integrate mobile manipulation planning
- [ ] Implement hybrid force/position controller
- [ ] Develop manipulation skill library
- [ ] Test in simulation before hardware
- [ ] Add safety checks for contact tasks

## Further Reading

- Murray, R.M. "A Mathematical Introduction to Robotic Manipulation" (1994)
- Siciliano, B. "Robotics: Modelling, Planning and Control" (2010)
- Mason, M.T. "Mechanics of Robotic Manipulation" (2001)
- Prattichizzo, D. "Grasping" in Springer Handbook of Robotics (2016)
- Khatib, O. "A Unified Approach for Motion and Force Control" (1987)
