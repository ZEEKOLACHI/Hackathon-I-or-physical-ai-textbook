---
id: ch-7-19
title: System Integration
sidebar_position: 1
difficulty: advanced
estimated_time: 60
prerequisites: [ch-6-16, ch-6-17, ch-6-18]
---

# System Integration

> "The whole is greater than the sum of its parts."
> — Aristotle

Building a working humanoid robot requires integrating perception, planning, control, and learning into a cohesive system. Each subsystem we have studied is sophisticated, but the real challenge lies in making them work together reliably at runtime. This chapter explores the architecture patterns, communication strategies, and engineering practices that enable successful Physical AI systems.

## The Integration Challenge

```
              Physical AI System Architecture

    ┌─────────────────────────────────────────────────────────────────┐
    │                     High-Level Intelligence                      │
    │  ┌─────────────────────────────────────────────────────────────┐│
    │  │ Task Planning │ Language Understanding │ World Model        ││
    │  └─────────────────────────────────────────────────────────────┘│
    │                              │                                   │
    │                              ▼                                   │
    │  ┌─────────────────────────────────────────────────────────────┐│
    │  │                    Motion Planning                           ││
    │  │  Path Planning │ Trajectory Optimization │ Collision Avoid  ││
    │  └─────────────────────────────────────────────────────────────┘│
    │                              │                                   │
    │                              ▼                                   │
    │  ┌─────────────────────────────────────────────────────────────┐│
    │  │                   Real-Time Control                          ││
    │  │  Whole-Body │ Balance │ Force Control │ Joint Servos        ││
    │  └─────────────────────────────────────────────────────────────┘│
    │                              │                                   │
    └──────────────────────────────┼───────────────────────────────────┘
                                   │
    ┌──────────────────────────────┼───────────────────────────────────┐
    │                              ▼                                   │
    │  ┌─────────────────────────────────────────────────────────────┐│
    │  │                      Hardware                                ││
    │  │  Motors │ Sensors │ Computers │ Power │ Communication       ││
    │  └─────────────────────────────────────────────────────────────┘│
    └─────────────────────────────────────────────────────────────────┘


              Integration Challenges
              ──────────────────────

    1. Timing          Different loops run at different rates
       ───────         Task: seconds, Motion: 100ms, Control: 1ms

    2. Data Flow       Information must flow between subsystems
       ─────────       Sensors → Perception → Planning → Control

    3. Failures        Any component can fail at any time
       ────────        Need graceful degradation and recovery

    4. Resources       Limited compute, memory, bandwidth, power
       ─────────       Must prioritize and allocate efficiently

    5. Latency         End-to-end delay affects performance
       ───────         Camera → Action must be fast enough
```

## Architecture Patterns

### Hierarchical Control Architecture

The most common pattern organizes control into layers by timescale.

```
              Hierarchical Architecture

    Timescale          Layer                 Responsibility
    ─────────          ─────                 ──────────────

    Minutes to         ┌─────────────────┐
    Hours              │ Task Planning   │   What to do
                       │ (Strategic)     │   Goal decomposition
                       └────────┬────────┘
                                │
    Seconds to                  ▼
    Minutes            ┌─────────────────┐
                       │ Motion Planning │   How to move
                       │ (Tactical)      │   Trajectory generation
                       └────────┬────────┘
                                │
    Milliseconds                ▼
                       ┌─────────────────┐
                       │ Servo Control   │   Execute motion
                       │ (Operational)   │   Track trajectories
                       └────────┬────────┘
                                │
    Microseconds                ▼
                       ┌─────────────────┐
                       │ Hardware/Drivers│   Motor commands
                       │ (Physical)      │   Sensor reading
                       └─────────────────┘


    Key Principle: Each layer operates at its natural timescale
                   Higher layers set goals for lower layers
                   Lower layers provide feedback to higher layers
```

```python
"""
Hierarchical Control Architecture Module

Implements a multi-level control architecture for humanoid robots.
"""

import time
import threading
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue, Empty
import numpy as np


class ControlLevel(Enum):
    """Levels in the control hierarchy."""
    STRATEGIC = "strategic"      # Task planning (1-10 Hz)
    TACTICAL = "tactical"        # Motion planning (10-100 Hz)
    OPERATIONAL = "operational"  # Real-time control (100-1000 Hz)
    PHYSICAL = "physical"        # Hardware interface (1000+ Hz)


@dataclass
class Goal:
    """A goal to be achieved by a control level."""
    type: str
    target: Any
    priority: int = 0
    deadline: Optional[float] = None
    constraints: Dict = field(default_factory=dict)


@dataclass
class Feedback:
    """Feedback from a lower level to a higher level."""
    status: str           # 'executing', 'completed', 'failed', 'blocked'
    progress: float       # 0.0 to 1.0
    error: Optional[str] = None
    data: Dict = field(default_factory=dict)


class ControlLayer(ABC):
    """Abstract base class for control layers."""

    def __init__(self, name: str, rate_hz: float):
        """
        Initialize control layer.

        Args:
            name: Layer name for logging
            rate_hz: Target update rate
        """
        self.name = name
        self.rate_hz = rate_hz
        self.period = 1.0 / rate_hz

        self.running = False
        self.current_goal: Optional[Goal] = None
        self.feedback: Optional[Feedback] = None

    @abstractmethod
    def update(self, state: Dict) -> Any:
        """
        Process one update cycle.

        Args:
            state: Current system state

        Returns:
            Output to lower level (or hardware)
        """
        pass

    @abstractmethod
    def set_goal(self, goal: Goal):
        """Set a new goal for this layer."""
        pass

    @abstractmethod
    def get_feedback(self) -> Feedback:
        """Get current feedback for higher level."""
        pass


class TaskPlanner(ControlLayer):
    """
    Strategic level: Task planning.

    Decomposes high-level goals into subtasks.
    Runs at 1-10 Hz.
    """

    def __init__(self, world_model):
        super().__init__("TaskPlanner", rate_hz=2.0)
        self.world_model = world_model
        self.task_queue: List[Goal] = []
        self.current_subtask = None

    def update(self, state: Dict) -> Optional[Goal]:
        """
        Plan and sequence tasks.

        Returns next subtask goal for motion planner.
        """
        # Update world model with current observations
        self.world_model.update(state.get('observations', {}))

        # Check if current subtask is complete
        if self.current_subtask:
            motion_feedback = state.get('motion_feedback')
            if motion_feedback and motion_feedback.status == 'completed':
                self._on_subtask_complete()
            elif motion_feedback and motion_feedback.status == 'failed':
                self._on_subtask_failed(motion_feedback.error)

        # Get next subtask if needed
        if self.current_subtask is None and self.current_goal:
            self.current_subtask = self._plan_next_subtask()

        return self.current_subtask

    def set_goal(self, goal: Goal):
        """Set high-level task goal."""
        self.current_goal = goal
        self.task_queue = self._decompose_task(goal)
        self.current_subtask = None
        self.feedback = Feedback(status='executing', progress=0.0)

    def _decompose_task(self, goal: Goal) -> List[Goal]:
        """Decompose task into subtasks."""
        subtasks = []

        if goal.type == 'pick_and_place':
            # Decompose into: approach, grasp, lift, move, place, release
            subtasks = [
                Goal(type='move_to', target=goal.constraints['pick_approach']),
                Goal(type='grasp', target=goal.constraints['object']),
                Goal(type='lift', target={'height': 0.1}),
                Goal(type='move_to', target=goal.constraints['place_approach']),
                Goal(type='place', target=goal.target),
                Goal(type='release', target=None),
            ]

        elif goal.type == 'navigate_to':
            # May need to decompose if obstacles
            waypoints = self.world_model.plan_path(
                state['position'], goal.target
            )
            for wp in waypoints:
                subtasks.append(Goal(type='walk_to', target=wp))

        return subtasks

    def _plan_next_subtask(self) -> Optional[Goal]:
        """Get next subtask from queue."""
        if self.task_queue:
            return self.task_queue.pop(0)
        return None

    def _on_subtask_complete(self):
        """Handle subtask completion."""
        completed = len(self.current_goal.constraints.get('subtasks', [])) - len(self.task_queue)
        total = len(self.current_goal.constraints.get('subtasks', [])) + completed
        self.feedback.progress = completed / max(total, 1)

        self.current_subtask = None

        if not self.task_queue:
            self.feedback.status = 'completed'
            self.feedback.progress = 1.0

    def get_feedback(self) -> Feedback:
        return self.feedback or Feedback(status='idle', progress=0.0)


class MotionPlanner(ControlLayer):
    """
    Tactical level: Motion planning.

    Plans trajectories to achieve subtask goals.
    Runs at 10-100 Hz.
    """

    def __init__(self, robot_model, planning_scene):
        super().__init__("MotionPlanner", rate_hz=50.0)
        self.robot = robot_model
        self.scene = planning_scene
        self.current_trajectory = None
        self.trajectory_index = 0

    def update(self, state: Dict) -> Optional[np.ndarray]:
        """
        Generate trajectory waypoints.

        Returns next waypoint for real-time controller.
        """
        if self.current_trajectory is None:
            return None

        # Get next waypoint
        if self.trajectory_index < len(self.current_trajectory):
            waypoint = self.current_trajectory[self.trajectory_index]
            self.trajectory_index += 1

            # Update progress
            progress = self.trajectory_index / len(self.current_trajectory)
            self.feedback = Feedback(status='executing', progress=progress)

            return waypoint

        # Trajectory complete
        self.feedback = Feedback(status='completed', progress=1.0)
        return None

    def set_goal(self, goal: Goal):
        """Plan trajectory for subtask goal."""
        self.current_goal = goal
        self.feedback = Feedback(status='planning', progress=0.0)

        if goal.type == 'move_to':
            self.current_trajectory = self._plan_arm_trajectory(goal.target)
        elif goal.type == 'walk_to':
            self.current_trajectory = self._plan_walking_trajectory(goal.target)
        elif goal.type == 'grasp':
            self.current_trajectory = self._plan_grasp_trajectory(goal.target)
        else:
            self.current_trajectory = None

        self.trajectory_index = 0

        if self.current_trajectory is not None:
            self.feedback = Feedback(status='executing', progress=0.0)
        else:
            self.feedback = Feedback(status='failed', progress=0.0,
                                      error='Planning failed')

    def _plan_arm_trajectory(self, target) -> Optional[List[np.ndarray]]:
        """Plan collision-free arm trajectory."""
        current = self.robot.get_arm_configuration()

        # Use RRT or optimization-based planner
        path = self._rrt_plan(current, target)

        if path is None:
            return None

        # Smooth and time-parameterize
        trajectory = self._smooth_trajectory(path)
        return trajectory

    def get_feedback(self) -> Feedback:
        return self.feedback or Feedback(status='idle', progress=0.0)


class RealtimeController(ControlLayer):
    """
    Operational level: Real-time control.

    Executes trajectories with feedback control.
    Runs at 100-1000 Hz (hard real-time).
    """

    def __init__(self, robot_model, control_gains):
        super().__init__("RealtimeController", rate_hz=1000.0)
        self.robot = robot_model
        self.gains = control_gains
        self.target_waypoint = None

    def update(self, state: Dict) -> np.ndarray:
        """
        Compute control command.

        Returns joint torques or velocities.
        """
        if self.target_waypoint is None:
            # Hold current position
            return self._compute_hold_command(state)

        # Track waypoint
        current = state['joint_positions']
        velocity = state['joint_velocities']

        error = self.target_waypoint - current
        command = self._pd_control(error, velocity)

        # Add feedforward, gravity compensation, etc.
        command += self._compute_feedforward(self.target_waypoint)
        command += self._compute_gravity_compensation(current)

        return command

    def set_goal(self, goal: Goal):
        """Set waypoint to track."""
        if isinstance(goal.target, np.ndarray):
            self.target_waypoint = goal.target
        self.feedback = Feedback(status='tracking', progress=0.0)

    def _pd_control(self, error: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """Simple PD control."""
        return self.gains['kp'] * error - self.gains['kd'] * velocity

    def _compute_gravity_compensation(self, q: np.ndarray) -> np.ndarray:
        """Compute gravity compensation torques."""
        return self.robot.gravity_vector(q)

    def get_feedback(self) -> Feedback:
        return self.feedback or Feedback(status='idle', progress=0.0)


class HierarchicalController:
    """
    Complete hierarchical control system.

    Manages all control layers and data flow between them.
    """

    def __init__(self, config: Dict):
        """
        Initialize hierarchical controller.

        Args:
            config: Configuration for all layers
        """
        # Create layers
        self.task_planner = TaskPlanner(config['world_model'])
        self.motion_planner = MotionPlanner(
            config['robot_model'],
            config['planning_scene']
        )
        self.controller = RealtimeController(
            config['robot_model'],
            config['control_gains']
        )

        # Layer threads
        self.threads: Dict[str, threading.Thread] = {}
        self.running = False

        # Communication queues
        self.task_to_motion = Queue(maxsize=10)
        self.motion_to_control = Queue(maxsize=100)
        self.sensor_data = Queue(maxsize=10)

    def start(self):
        """Start all control layers."""
        self.running = True

        # Start each layer in its own thread
        self.threads['task'] = threading.Thread(
            target=self._task_loop,
            daemon=True
        )
        self.threads['motion'] = threading.Thread(
            target=self._motion_loop,
            daemon=True
        )
        self.threads['control'] = threading.Thread(
            target=self._control_loop,
            daemon=True
        )

        for thread in self.threads.values():
            thread.start()

    def stop(self):
        """Stop all control layers."""
        self.running = False
        for thread in self.threads.values():
            thread.join(timeout=1.0)

    def _task_loop(self):
        """Task planning loop (slow)."""
        while self.running:
            start_time = time.time()

            # Get latest state
            state = self._get_state()

            # Update task planner
            subtask = self.task_planner.update(state)

            if subtask:
                self.task_to_motion.put(subtask)

            # Sleep to maintain rate
            elapsed = time.time() - start_time
            sleep_time = self.task_planner.period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _motion_loop(self):
        """Motion planning loop (medium)."""
        while self.running:
            start_time = time.time()

            # Check for new subtask from task planner
            try:
                subtask = self.task_to_motion.get_nowait()
                self.motion_planner.set_goal(subtask)
            except Empty:
                pass

            # Get latest state
            state = self._get_state()

            # Update motion planner
            waypoint = self.motion_planner.update(state)

            if waypoint is not None:
                goal = Goal(type='waypoint', target=waypoint)
                try:
                    self.motion_to_control.put_nowait(goal)
                except:
                    pass  # Queue full, skip this waypoint

            # Sleep to maintain rate
            elapsed = time.time() - start_time
            sleep_time = self.motion_planner.period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _control_loop(self):
        """Real-time control loop (fast)."""
        while self.running:
            start_time = time.time()

            # Check for new waypoint from motion planner
            try:
                goal = self.motion_to_control.get_nowait()
                self.controller.set_goal(goal)
            except Empty:
                pass

            # Get latest state
            state = self._get_state()

            # Compute and send control command
            command = self.controller.update(state)
            self._send_command(command)

            # Sleep to maintain rate (or use real-time scheduling)
            elapsed = time.time() - start_time
            sleep_time = self.controller.period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def set_task(self, task: Goal):
        """Set high-level task goal."""
        self.task_planner.set_goal(task)

    def _get_state(self) -> Dict:
        """Get current system state."""
        # This would read from actual sensors/state estimator
        return {
            'joint_positions': np.zeros(7),
            'joint_velocities': np.zeros(7),
            'observations': {}
        }

    def _send_command(self, command: np.ndarray):
        """Send command to hardware."""
        # This would write to actual motor drivers
        pass
```

### Behavior-Based Architecture

An alternative pattern inspired by biological systems.

```
              Behavior-Based Architecture (Subsumption)

    ┌─────────────────────────────────────────────────────────────────┐
    │ Layer 3: Explore                                                │
    │ ─────────────────                                               │
    │ Wander, seek interesting areas                                  │
    │                                                     ┌───────┐   │
    │ ┌─────────┐    ┌─────────┐    ┌─────────┐         │Output │   │
    │ │ Explore │───►│ Heading │───►│   S     │────────►│       │   │
    │ └─────────┘    └─────────┘    └────┬────┘         └───┬───┘   │
    │                                    │ Suppress           │       │
    ├────────────────────────────────────┼───────────────────┼───────┤
    │ Layer 2: Avoid                     │                   │       │
    │ ─────────────────                  ▼                   │       │
    │ Avoid obstacles                ┌───────┐               │       │
    │                                │   S   │               │       │
    │ ┌─────────┐    ┌─────────┐    └───┬───┘               │       │
    │ │ Sonar   │───►│ Avoid   │────────┴──────────────────►│       │
    │ └─────────┘    └─────────┘                             │       │
    │                                     │ Suppress         │       │
    ├─────────────────────────────────────┼─────────────────┼───────┤
    │ Layer 1: Escape                     ▼                 │       │
    │ ────────────────                ┌───────┐             │       │
    │ Run from danger                 │   S   │             │       │
    │                                 └───┬───┘             │       │
    │ ┌─────────┐    ┌─────────┐         │                  │       │
    │ │ Threat  │───►│ Escape  │─────────┴─────────────────►│       │
    │ └─────────┘    └─────────┘                             │       │
    │                                                        ▼       │
    └───────────────────────────────────────────────────────────────┘
                                                         To Motors


    Key Principle: Lower layers (survival) suppress higher layers
                   No central planner - emergent behavior
                   Reactive and robust to failures
```

```python
"""
Behavior-Based Architecture Module

Implements subsumption-style behavior layering.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
from enum import Enum


@dataclass
class BehaviorOutput:
    """Output from a behavior module."""
    velocity: np.ndarray          # Desired velocity command
    valid: bool                   # Whether this output is active
    priority: int                 # Higher = more urgent
    confidence: float = 1.0       # Confidence in this command


class Behavior(ABC):
    """Base class for reactive behaviors."""

    def __init__(self, name: str, priority: int):
        """
        Initialize behavior.

        Args:
            name: Behavior name for debugging
            priority: Priority level (higher = more important)
        """
        self.name = name
        self.priority = priority

    @abstractmethod
    def compute(self, state: Dict) -> BehaviorOutput:
        """
        Compute behavior output given current state.

        Args:
            state: Current sensor/state data

        Returns:
            Behavior output (may be invalid if not applicable)
        """
        pass


class EmergencyStopBehavior(Behavior):
    """
    Highest priority: Emergency stop.

    Activates when critical safety conditions are violated.
    """

    def __init__(self):
        super().__init__("EmergencyStop", priority=100)

    def compute(self, state: Dict) -> BehaviorOutput:
        # Check critical conditions
        if state.get('e_stop_pressed', False):
            return BehaviorOutput(
                velocity=np.zeros(6),
                valid=True,
                priority=self.priority
            )

        if state.get('joint_limit_violation', False):
            return BehaviorOutput(
                velocity=np.zeros(6),
                valid=True,
                priority=self.priority
            )

        # Not active
        return BehaviorOutput(
            velocity=np.zeros(6),
            valid=False,
            priority=self.priority
        )


class CollisionAvoidanceBehavior(Behavior):
    """
    High priority: Avoid collisions.

    Uses proximity sensors to steer away from obstacles.
    """

    def __init__(self, safety_distance: float = 0.5):
        super().__init__("CollisionAvoidance", priority=80)
        self.safety_distance = safety_distance

    def compute(self, state: Dict) -> BehaviorOutput:
        obstacles = state.get('obstacles', [])

        if not obstacles:
            return BehaviorOutput(
                velocity=np.zeros(6),
                valid=False,
                priority=self.priority
            )

        # Find nearest obstacle
        min_distance = float('inf')
        nearest_direction = None

        for obs in obstacles:
            distance = obs['distance']
            if distance < min_distance:
                min_distance = distance
                nearest_direction = obs['direction']

        # Check if too close
        if min_distance < self.safety_distance:
            # Compute avoidance velocity (move away from obstacle)
            avoid_direction = -nearest_direction / np.linalg.norm(nearest_direction)

            # Strength increases as we get closer
            strength = (self.safety_distance - min_distance) / self.safety_distance
            velocity = np.zeros(6)
            velocity[:3] = avoid_direction * strength * 0.5  # Max 0.5 m/s

            return BehaviorOutput(
                velocity=velocity,
                valid=True,
                priority=self.priority,
                confidence=strength
            )

        return BehaviorOutput(
            velocity=np.zeros(6),
            valid=False,
            priority=self.priority
        )


class BalanceMaintenanceBehavior(Behavior):
    """
    High priority: Maintain balance.

    Adjusts posture to keep CoM over support polygon.
    """

    def __init__(self, stability_margin: float = 0.05):
        super().__init__("BalanceMaintenance", priority=75)
        self.stability_margin = stability_margin

    def compute(self, state: Dict) -> BehaviorOutput:
        com = state.get('com_position', np.zeros(3))
        support = state.get('support_polygon', [])

        if len(support) < 3:
            return BehaviorOutput(
                velocity=np.zeros(6),
                valid=False,
                priority=self.priority
            )

        # Compute distance to support edge
        margin = self._compute_margin(com[:2], support)

        if margin < self.stability_margin:
            # Need to correct balance
            # Move CoM toward center of support
            center = np.mean(support, axis=0)
            correction = center - com[:2]

            velocity = np.zeros(6)
            velocity[:2] = correction * 2.0  # Gain

            return BehaviorOutput(
                velocity=velocity,
                valid=True,
                priority=self.priority,
                confidence=1.0 - margin / self.stability_margin
            )

        return BehaviorOutput(
            velocity=np.zeros(6),
            valid=False,
            priority=self.priority
        )

    def _compute_margin(self, point: np.ndarray, polygon: List) -> float:
        """Compute distance from point to polygon edge."""
        # Simplified - proper implementation uses computational geometry
        center = np.mean(polygon, axis=0)
        return np.linalg.norm(point - center)


class GoalSeekingBehavior(Behavior):
    """
    Low priority: Move toward goal.

    Basic attractive field toward target.
    """

    def __init__(self, goal: np.ndarray = None):
        super().__init__("GoalSeeking", priority=20)
        self.goal = goal

    def set_goal(self, goal: np.ndarray):
        """Set new goal position."""
        self.goal = goal

    def compute(self, state: Dict) -> BehaviorOutput:
        if self.goal is None:
            return BehaviorOutput(
                velocity=np.zeros(6),
                valid=False,
                priority=self.priority
            )

        position = state.get('position', np.zeros(3))

        # Direction to goal
        to_goal = self.goal - position
        distance = np.linalg.norm(to_goal)

        if distance < 0.1:  # Close enough
            return BehaviorOutput(
                velocity=np.zeros(6),
                valid=False,
                priority=self.priority
            )

        # Move toward goal (saturated velocity)
        direction = to_goal / distance
        speed = min(distance, 0.5)  # Max 0.5 m/s

        velocity = np.zeros(6)
        velocity[:3] = direction * speed

        return BehaviorOutput(
            velocity=velocity,
            valid=True,
            priority=self.priority,
            confidence=1.0
        )


class BehaviorArbitrator:
    """
    Combines outputs from multiple behaviors.

    Uses priority-based arbitration (subsumption).
    """

    def __init__(self):
        self.behaviors: List[Behavior] = []

    def add_behavior(self, behavior: Behavior):
        """Add a behavior to the system."""
        self.behaviors.append(behavior)
        # Keep sorted by priority (highest first)
        self.behaviors.sort(key=lambda b: b.priority, reverse=True)

    def compute(self, state: Dict) -> np.ndarray:
        """
        Compute final velocity command.

        Uses winner-take-all based on priority.
        """
        for behavior in self.behaviors:
            output = behavior.compute(state)

            if output.valid:
                # This behavior wants control
                return output.velocity

        # No behavior active - stop
        return np.zeros(6)


class BlendedArbitrator:
    """
    Alternative: Blend behavior outputs by confidence.

    Smoother but less predictable than winner-take-all.
    """

    def __init__(self):
        self.behaviors: List[Behavior] = []

    def add_behavior(self, behavior: Behavior):
        self.behaviors.append(behavior)

    def compute(self, state: Dict) -> np.ndarray:
        """
        Compute blended velocity command.

        Weighted average based on priority and confidence.
        """
        total_weight = 0.0
        blended_velocity = np.zeros(6)

        for behavior in self.behaviors:
            output = behavior.compute(state)

            if output.valid:
                weight = output.priority * output.confidence
                blended_velocity += weight * output.velocity
                total_weight += weight

        if total_weight > 0:
            return blended_velocity / total_weight

        return np.zeros(6)


class BehaviorBasedController:
    """
    Complete behavior-based control system.
    """

    def __init__(self, arbitration: str = 'priority'):
        """
        Initialize behavior-based controller.

        Args:
            arbitration: 'priority' or 'blended'
        """
        if arbitration == 'priority':
            self.arbitrator = BehaviorArbitrator()
        else:
            self.arbitrator = BlendedArbitrator()

        # Add default behaviors
        self.arbitrator.add_behavior(EmergencyStopBehavior())
        self.arbitrator.add_behavior(CollisionAvoidanceBehavior())
        self.arbitrator.add_behavior(BalanceMaintenanceBehavior())
        self.arbitrator.add_behavior(GoalSeekingBehavior())

    def update(self, state: Dict) -> np.ndarray:
        """
        Compute control command from behaviors.

        Args:
            state: Current system state

        Returns:
            Velocity command
        """
        return self.arbitrator.compute(state)

    def set_goal(self, goal: np.ndarray):
        """Set navigation goal."""
        for behavior in self.arbitrator.behaviors:
            if isinstance(behavior, GoalSeekingBehavior):
                behavior.set_goal(goal)
```

## ROS 2 System Integration

ROS 2 (Robot Operating System 2) provides a robust middleware for building integrated robot systems.

```
              ROS 2 Architecture for Humanoid

    ┌─────────────────────────────────────────────────────────────────┐
    │                        ROS 2 Graph                              │
    │                                                                 │
    │  ┌─────────────┐    /camera/rgb    ┌─────────────┐             │
    │  │   Camera    │─────────────────►│  Detection  │             │
    │  │   Driver    │                   │    Node     │             │
    │  └─────────────┘                   └──────┬──────┘             │
    │                                           │                     │
    │  ┌─────────────┐    /imu/data            │ /detections         │
    │  │    IMU      │─────────────┐           │                     │
    │  │   Driver    │             │           ▼                     │
    │  └─────────────┘             │    ┌─────────────┐             │
    │                              ├───►│   State     │             │
    │  ┌─────────────┐  /joint_states   │  Estimator  │             │
    │  │   Joint     │─────────────┴───►│    Node     │             │
    │  │  Encoders   │                   └──────┬──────┘             │
    │  └─────────────┘                          │                     │
    │                                           │ /robot_state        │
    │                                           ▼                     │
    │  ┌─────────────┐    /cmd_vel      ┌─────────────┐             │
    │  │  Motion     │◄─────────────────│  Planning   │             │
    │  │ Controller  │                   │    Node     │◄── /goal   │
    │  └──────┬──────┘                   └─────────────┘             │
    │         │                                                       │
    │         │ /joint_commands                                       │
    │         ▼                                                       │
    │  ┌─────────────┐                                               │
    │  │   Motor     │                                               │
    │  │   Drivers   │                                               │
    │  └─────────────┘                                               │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
```

### ROS 2 Node Implementation

```python
"""
ROS 2 Humanoid Control Nodes

Production-quality ROS 2 nodes for humanoid control.
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time

from sensor_msgs.msg import Image, Imu, JointState
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Bool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import numpy as np
from typing import Optional
from threading import Lock


class PerceptionNode(Node):
    """
    Perception node for object detection and tracking.
    """

    def __init__(self):
        super().__init__('perception_node')

        # QoS for sensor data (best effort for real-time)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # QoS for detection output (reliable)
        detect_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribers
        self.camera_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.camera_callback,
            sensor_qos
        )

        self.depth_sub = self.create_subscription(
            Image,
            '/camera/depth/image_rect',
            self.depth_callback,
            sensor_qos
        )

        # Publishers
        self.detection_pub = self.create_publisher(
            DetectionArray,  # Custom message
            '/detections',
            detect_qos
        )

        # State
        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_depth: Optional[np.ndarray] = None
        self.lock = Lock()

        # Detector (initialized later)
        self.detector = None

        # Processing timer (10 Hz detection)
        self.create_timer(0.1, self.process_callback)

        self.get_logger().info('Perception node started')

    def camera_callback(self, msg: Image):
        """Handle RGB image."""
        with self.lock:
            self.latest_rgb = self._msg_to_numpy(msg)

    def depth_callback(self, msg: Image):
        """Handle depth image."""
        with self.lock:
            self.latest_depth = self._msg_to_numpy(msg)

    def process_callback(self):
        """Run detection on latest images."""
        with self.lock:
            rgb = self.latest_rgb
            depth = self.latest_depth

        if rgb is None:
            return

        # Run detection
        detections = self.detector.detect(rgb, depth)

        # Publish results
        msg = self._detections_to_msg(detections)
        self.detection_pub.publish(msg)


class StateEstimatorNode(Node):
    """
    State estimation node.

    Fuses sensor data to estimate robot state.
    """

    def __init__(self):
        super().__init__('state_estimator_node')

        # Callback groups for concurrent execution
        self.sensor_cb_group = ReentrantCallbackGroup()
        self.timer_cb_group = MutuallyExclusiveCallbackGroup()

        # Subscribers (different callback groups allow parallel execution)
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10,
            callback_group=self.sensor_cb_group
        )

        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10,
            callback_group=self.sensor_cb_group
        )

        # Publisher
        self.state_pub = self.create_publisher(
            RobotState,  # Custom message
            '/robot_state',
            10
        )

        # State
        self.lock = Lock()
        self.latest_imu: Optional[Imu] = None
        self.latest_joints: Optional[JointState] = None

        # State estimator
        self.estimator = None  # EKF or similar

        # High-rate state estimation (100 Hz)
        self.create_timer(
            0.01,
            self.estimate_callback,
            callback_group=self.timer_cb_group
        )

    def imu_callback(self, msg: Imu):
        """Handle IMU data."""
        with self.lock:
            self.latest_imu = msg

    def joint_callback(self, msg: JointState):
        """Handle joint state data."""
        with self.lock:
            self.latest_joints = msg

    def estimate_callback(self):
        """Run state estimation."""
        with self.lock:
            imu = self.latest_imu
            joints = self.latest_joints

        if imu is None or joints is None:
            return

        # Run estimator
        state = self.estimator.update(imu, joints)

        # Publish
        msg = self._state_to_msg(state)
        self.state_pub.publish(msg)


class MotionControllerNode(Node):
    """
    Real-time motion controller node.

    Runs at 1kHz to track trajectories.
    """

    def __init__(self):
        super().__init__('motion_controller_node')

        # Declare parameters
        self.declare_parameter('control_rate', 1000.0)
        self.declare_parameter('kp', [100.0] * 7)
        self.declare_parameter('kd', [10.0] * 7)

        control_rate = self.get_parameter('control_rate').value
        self.kp = np.array(self.get_parameter('kp').value)
        self.kd = np.array(self.get_parameter('kd').value)

        # Subscribers
        self.state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.state_callback,
            10
        )

        self.trajectory_sub = self.create_subscription(
            JointTrajectory,
            '/trajectory',
            self.trajectory_callback,
            10
        )

        # Publisher
        self.command_pub = self.create_publisher(
            JointState,  # Commands to motor drivers
            '/joint_commands',
            10
        )

        # State
        self.current_state: Optional[JointState] = None
        self.current_trajectory: Optional[JointTrajectory] = None
        self.trajectory_start_time: Optional[Time] = None
        self.lock = Lock()

        # Control timer (1 kHz)
        self.create_timer(1.0 / control_rate, self.control_callback)

        self.get_logger().info(f'Motion controller started at {control_rate} Hz')

    def state_callback(self, msg: JointState):
        """Handle joint state feedback."""
        with self.lock:
            self.current_state = msg

    def trajectory_callback(self, msg: JointTrajectory):
        """Handle new trajectory."""
        with self.lock:
            self.current_trajectory = msg
            self.trajectory_start_time = self.get_clock().now()

        self.get_logger().info(f'New trajectory with {len(msg.points)} points')

    def control_callback(self):
        """Real-time control loop."""
        with self.lock:
            state = self.current_state
            trajectory = self.current_trajectory
            start_time = self.trajectory_start_time

        if state is None:
            return

        # Get current state
        q = np.array(state.position)
        qd = np.array(state.velocity)

        # Get desired state from trajectory
        if trajectory is not None and start_time is not None:
            elapsed = (self.get_clock().now() - start_time).nanoseconds / 1e9
            q_des, qd_des = self._interpolate_trajectory(trajectory, elapsed)
        else:
            # Hold current position
            q_des = q
            qd_des = np.zeros_like(qd)

        # PD control
        tau = self.kp * (q_des - q) + self.kd * (qd_des - qd)

        # Publish command
        cmd = JointState()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.name = state.name
        cmd.effort = tau.tolist()

        self.command_pub.publish(cmd)

    def _interpolate_trajectory(self,
                                 trajectory: JointTrajectory,
                                 t: float) -> tuple:
        """Interpolate trajectory at time t."""
        # Find bracketing points
        for i, point in enumerate(trajectory.points):
            point_time = point.time_from_start.sec + \
                        point.time_from_start.nanosec / 1e9

            if point_time >= t:
                if i == 0:
                    return np.array(point.positions), np.array(point.velocities)

                # Linear interpolation
                prev_point = trajectory.points[i - 1]
                prev_time = prev_point.time_from_start.sec + \
                           prev_point.time_from_start.nanosec / 1e9

                alpha = (t - prev_time) / (point_time - prev_time)

                q = (1 - alpha) * np.array(prev_point.positions) + \
                    alpha * np.array(point.positions)
                qd = (1 - alpha) * np.array(prev_point.velocities) + \
                     alpha * np.array(point.velocities)

                return q, qd

        # Past end of trajectory
        last = trajectory.points[-1]
        return np.array(last.positions), np.zeros(len(last.positions))


def main():
    """Main entry point."""
    rclpy.init()

    # Create executor for parallel node execution
    executor = MultiThreadedExecutor(num_threads=4)

    # Create nodes
    perception = PerceptionNode()
    state_estimator = StateEstimatorNode()
    controller = MotionControllerNode()

    # Add nodes to executor
    executor.add_node(perception)
    executor.add_node(state_estimator)
    executor.add_node(controller)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Launch System

```python
"""
ROS 2 Launch File for Humanoid Robot

Launches all nodes with proper configuration.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, ComposableNodeContainer, LoadComposableNode
from launch_ros.descriptions import ComposableNode
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare arguments
    use_sim_arg = DeclareLaunchArgument(
        'use_sim',
        default_value='false',
        description='Use simulation instead of real robot'
    )

    robot_name_arg = DeclareLaunchArgument(
        'robot_name',
        default_value='humanoid',
        description='Robot name for namespacing'
    )

    # Get configuration file paths
    config_path = PathJoinSubstitution([
        FindPackageShare('humanoid_bringup'),
        'config',
        'default_params.yaml'
    ])

    # Perception container (composable nodes for efficiency)
    perception_container = ComposableNodeContainer(
        name='perception_container',
        namespace=LaunchConfiguration('robot_name'),
        package='rclcpp_components',
        executable='component_container_mt',  # Multi-threaded
        composable_node_descriptions=[
            ComposableNode(
                package='humanoid_perception',
                plugin='humanoid::CameraDriverNode',
                name='camera_driver',
                parameters=[config_path],
            ),
            ComposableNode(
                package='humanoid_perception',
                plugin='humanoid::ObjectDetectorNode',
                name='object_detector',
                parameters=[config_path],
            ),
            ComposableNode(
                package='humanoid_perception',
                plugin='humanoid::PointCloudProcessorNode',
                name='pointcloud_processor',
                parameters=[config_path],
            ),
        ],
        output='screen',
    )

    # State estimation node (separate for isolation)
    state_estimator = Node(
        package='humanoid_state_estimation',
        executable='state_estimator_node',
        name='state_estimator',
        namespace=LaunchConfiguration('robot_name'),
        parameters=[config_path],
        output='screen',
    )

    # Motion planning node
    motion_planner = Node(
        package='humanoid_planning',
        executable='motion_planner_node',
        name='motion_planner',
        namespace=LaunchConfiguration('robot_name'),
        parameters=[
            config_path,
            {'planning_time': 1.0},
            {'planner_id': 'RRTConnect'},
        ],
        output='screen',
    )

    # Real-time controller (high priority)
    controller = Node(
        package='humanoid_control',
        executable='motion_controller_node',
        name='motion_controller',
        namespace=LaunchConfiguration('robot_name'),
        parameters=[
            config_path,
            {'control_rate': 1000.0},
        ],
        output='screen',
        # Request real-time scheduling
        prefix='chrt -f 90',
    )

    # Safety monitor
    safety_monitor = Node(
        package='humanoid_safety',
        executable='safety_monitor_node',
        name='safety_monitor',
        namespace=LaunchConfiguration('robot_name'),
        parameters=[config_path],
        output='screen',
    )

    return LaunchDescription([
        use_sim_arg,
        robot_name_arg,
        perception_container,
        state_estimator,
        motion_planner,
        controller,
        safety_monitor,
    ])
```

## State Machine Integration

State machines manage the high-level operating modes of the robot.

```
              Robot State Machine

                    ┌─────────┐
                    │  INIT   │
                    └────┬────┘
                         │ Hardware ready
                         ▼
                    ┌─────────┐
          ┌────────│  IDLE   │◄───────────────┐
          │        └────┬────┘                │
          │             │ Receive task        │ Task complete
          │             ▼                     │
          │        ┌─────────┐                │
          │   ┌───►│PLANNING │                │
          │   │    └────┬────┘                │
          │   │         │ Plan ready          │
          │   │ Replan  ▼                     │
          │   │    ┌─────────┐                │
          │   └────│EXECUTING│────────────────┘
          │        └────┬────┘
          │             │ Error detected
          │             ▼
          │        ┌─────────┐
          │        │RECOVERY │────────────────┐
          │        └────┬────┘                │
          │             │ Recovery failed     │ Recovery success
          │             ▼                     │
          │        ┌─────────┐                │
          └───────►│ ERROR   │                │
                   └────┬────┘                │
                        │ E-stop              │
                        ▼                     │
                   ┌─────────┐                │
                   │EMERGENCY│◄───────────────┘
                   └─────────┘                  (any state)
```

```python
"""
State Machine Module

Implements hierarchical state machine for robot operation.
"""

from enum import Enum, auto
from typing import Dict, Callable, Optional, List
from dataclasses import dataclass, field
import time


class RobotState(Enum):
    """High-level robot states."""
    INIT = auto()
    IDLE = auto()
    PLANNING = auto()
    EXECUTING = auto()
    RECOVERY = auto()
    ERROR = auto()
    EMERGENCY = auto()


@dataclass
class Transition:
    """State machine transition."""
    from_state: RobotState
    to_state: RobotState
    condition: Callable[[], bool]
    action: Optional[Callable[[], None]] = None


@dataclass
class StateConfig:
    """Configuration for a state."""
    on_enter: Optional[Callable[[], None]] = None
    on_exit: Optional[Callable[[], None]] = None
    on_update: Optional[Callable[[], None]] = None
    timeout: Optional[float] = None
    timeout_state: Optional[RobotState] = None


class StateMachine:
    """
    Hierarchical state machine for robot control.
    """

    def __init__(self):
        self.current_state = RobotState.INIT
        self.previous_state = None
        self.state_start_time = time.time()

        self.transitions: List[Transition] = []
        self.state_configs: Dict[RobotState, StateConfig] = {}

        self._setup_default_transitions()

    def _setup_default_transitions(self):
        """Set up default state machine transitions."""

        # INIT → IDLE when hardware ready
        self.add_transition(
            RobotState.INIT, RobotState.IDLE,
            condition=lambda: self._hardware_ready(),
            action=self._on_init_complete
        )

        # IDLE → PLANNING when task received
        self.add_transition(
            RobotState.IDLE, RobotState.PLANNING,
            condition=lambda: self._has_pending_task(),
            action=self._start_planning
        )

        # PLANNING → EXECUTING when plan ready
        self.add_transition(
            RobotState.PLANNING, RobotState.EXECUTING,
            condition=lambda: self._plan_ready(),
            action=self._start_execution
        )

        # EXECUTING → IDLE when task complete
        self.add_transition(
            RobotState.EXECUTING, RobotState.IDLE,
            condition=lambda: self._task_complete(),
            action=self._on_task_complete
        )

        # EXECUTING → RECOVERY when error detected
        self.add_transition(
            RobotState.EXECUTING, RobotState.RECOVERY,
            condition=lambda: self._error_detected(),
            action=self._start_recovery
        )

        # RECOVERY → IDLE when recovered
        self.add_transition(
            RobotState.RECOVERY, RobotState.IDLE,
            condition=lambda: self._recovery_complete(),
            action=self._on_recovery_complete
        )

        # RECOVERY → ERROR when recovery fails
        self.add_transition(
            RobotState.RECOVERY, RobotState.ERROR,
            condition=lambda: self._recovery_failed(),
            action=self._on_recovery_failed
        )

        # Any state → EMERGENCY on e-stop
        for state in RobotState:
            if state != RobotState.EMERGENCY:
                self.add_transition(
                    state, RobotState.EMERGENCY,
                    condition=lambda: self._emergency_triggered(),
                    action=self._emergency_stop
                )

        # Configure state behaviors
        self.state_configs[RobotState.PLANNING] = StateConfig(
            timeout=30.0,
            timeout_state=RobotState.ERROR,
            on_update=self._update_planning
        )

        self.state_configs[RobotState.RECOVERY] = StateConfig(
            timeout=10.0,
            timeout_state=RobotState.ERROR,
            on_update=self._update_recovery
        )

    def add_transition(self,
                       from_state: RobotState,
                       to_state: RobotState,
                       condition: Callable[[], bool],
                       action: Optional[Callable[[], None]] = None):
        """Add a state transition."""
        self.transitions.append(Transition(
            from_state=from_state,
            to_state=to_state,
            condition=condition,
            action=action
        ))

    def update(self):
        """Update state machine (call in main loop)."""
        # Check for transitions from current state
        for transition in self.transitions:
            if transition.from_state == self.current_state:
                if transition.condition():
                    self._do_transition(transition)
                    return

        # Check for timeout
        config = self.state_configs.get(self.current_state)
        if config and config.timeout:
            elapsed = time.time() - self.state_start_time
            if elapsed > config.timeout:
                self._do_timeout_transition(config)
                return

        # Run state update
        if config and config.on_update:
            config.on_update()

    def _do_transition(self, transition: Transition):
        """Execute a state transition."""
        # Exit current state
        old_config = self.state_configs.get(self.current_state)
        if old_config and old_config.on_exit:
            old_config.on_exit()

        # Execute transition action
        if transition.action:
            transition.action()

        # Update state
        self.previous_state = self.current_state
        self.current_state = transition.to_state
        self.state_start_time = time.time()

        # Enter new state
        new_config = self.state_configs.get(self.current_state)
        if new_config and new_config.on_enter:
            new_config.on_enter()

    def _do_timeout_transition(self, config: StateConfig):
        """Handle state timeout."""
        if config.timeout_state:
            # Create temporary transition
            timeout_transition = Transition(
                from_state=self.current_state,
                to_state=config.timeout_state,
                condition=lambda: True,
                action=lambda: self._on_timeout()
            )
            self._do_transition(timeout_transition)

    # Condition methods (implement based on your system)
    def _hardware_ready(self) -> bool:
        return True  # Check actual hardware

    def _has_pending_task(self) -> bool:
        return False  # Check task queue

    def _plan_ready(self) -> bool:
        return False  # Check planning status

    def _task_complete(self) -> bool:
        return False  # Check execution status

    def _error_detected(self) -> bool:
        return False  # Check for errors

    def _recovery_complete(self) -> bool:
        return False  # Check recovery status

    def _recovery_failed(self) -> bool:
        return False  # Check recovery status

    def _emergency_triggered(self) -> bool:
        return False  # Check e-stop

    # Action methods
    def _on_init_complete(self):
        print("Initialization complete")

    def _start_planning(self):
        print("Starting task planning")

    def _start_execution(self):
        print("Starting execution")

    def _on_task_complete(self):
        print("Task completed")

    def _start_recovery(self):
        print("Starting recovery")

    def _on_recovery_complete(self):
        print("Recovery successful")

    def _on_recovery_failed(self):
        print("Recovery failed")

    def _emergency_stop(self):
        print("EMERGENCY STOP")
        # Immediately stop all motion
        # Engage brakes
        # Safe shutdown

    def _on_timeout(self):
        print(f"State {self.current_state} timed out")

    def _update_planning(self):
        pass  # Planning progress

    def _update_recovery(self):
        pass  # Recovery progress
```

## Real-Time Communication

Reliable communication between components is critical for system integration.

```python
"""
Real-Time Communication Module

Lock-free data structures for real-time communication.
"""

import numpy as np
from typing import Optional, Generic, TypeVar
from dataclasses import dataclass
import threading
import time


T = TypeVar('T')


class TripleBuffer(Generic[T]):
    """
    Triple buffer for lock-free producer-consumer.

    Allows writer to always write without blocking,
    and reader to always get latest complete data.
    """

    def __init__(self):
        self._buffers = [None, None, None]
        self._write_idx = 0
        self._read_idx = 1
        self._ready_idx = 2
        self._lock = threading.Lock()

    def write(self, data: T):
        """
        Write new data (producer side).

        Never blocks - always succeeds.
        """
        # Write to current write buffer
        self._buffers[self._write_idx] = data

        # Swap write and ready buffers
        with self._lock:
            self._write_idx, self._ready_idx = self._ready_idx, self._write_idx

    def read(self) -> Optional[T]:
        """
        Read latest data (consumer side).

        Never blocks - returns latest complete data.
        """
        # Swap read and ready buffers
        with self._lock:
            self._read_idx, self._ready_idx = self._ready_idx, self._read_idx

        return self._buffers[self._read_idx]


@dataclass
class TimestampedData(Generic[T]):
    """Data with timestamp for freshness checking."""
    data: T
    timestamp: float


class DataBus:
    """
    Central data bus for system-wide state sharing.

    Provides named channels with freshness tracking.
    """

    def __init__(self):
        self._channels: dict = {}
        self._lock = threading.Lock()

    def create_channel(self, name: str):
        """Create a new data channel."""
        with self._lock:
            if name not in self._channels:
                self._channels[name] = TripleBuffer()

    def write(self, channel: str, data, timestamp: float = None):
        """Write data to channel."""
        if timestamp is None:
            timestamp = time.time()

        if channel not in self._channels:
            self.create_channel(channel)

        self._channels[channel].write(
            TimestampedData(data=data, timestamp=timestamp)
        )

    def read(self, channel: str, max_age: float = None) -> Optional:
        """
        Read latest data from channel.

        Args:
            channel: Channel name
            max_age: Maximum age in seconds (None = any age)

        Returns:
            Data if available and fresh, None otherwise
        """
        if channel not in self._channels:
            return None

        result = self._channels[channel].read()

        if result is None:
            return None

        if max_age is not None:
            age = time.time() - result.timestamp
            if age > max_age:
                return None  # Data too old

        return result.data


class RealtimePipeline:
    """
    Real-time processing pipeline.

    Manages data flow through processing stages.
    """

    def __init__(self, stages: list):
        """
        Initialize pipeline.

        Args:
            stages: List of processing functions
        """
        self.stages = stages
        self.buffers = [TripleBuffer() for _ in range(len(stages) + 1)]

    def push(self, data):
        """Push new input data."""
        self.buffers[0].write(data)

    def process(self):
        """Process through all stages."""
        for i, stage in enumerate(self.stages):
            input_data = self.buffers[i].read()

            if input_data is not None:
                output_data = stage(input_data)
                self.buffers[i + 1].write(output_data)

    def get_output(self):
        """Get final output."""
        return self.buffers[-1].read()
```

## Summary

```
┌────────────────────────────────────────────────────────────────────┐
│                   System Integration Recap                         │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Architecture Patterns                                             │
│  ─────────────────────                                             │
│  • Hierarchical: Strategic → Tactical → Operational                │
│  • Behavior-based: Reactive layers with priority arbitration       │
│  • Hybrid: Combine deliberative planning with reactive control     │
│                                                                    │
│  ROS 2 Integration                                                 │
│  ─────────────────                                                 │
│  • Nodes for modular functionality                                 │
│  • Topics for sensor/command data flow                             │
│  • Services for synchronous operations                             │
│  • Actions for long-running tasks                                  │
│  • QoS profiles for reliability/latency tradeoffs                  │
│                                                                    │
│  State Management                                                  │
│  ────────────────                                                  │
│  • State machines for operating modes                              │
│  • Clear transitions with conditions and actions                   │
│  • Timeout handling for stuck states                               │
│  • Emergency stop from any state                                   │
│                                                                    │
│  Real-Time Communication                                           │
│  ────────────────────────                                          │
│  • Lock-free data structures (triple buffer)                       │
│  • Timestamped data with freshness checking                        │
│  • Data bus for system-wide state sharing                          │
│  • Pipeline processing for sensor data                             │
│                                                                    │
│  Key Principles                                                    │
│  ──────────────                                                    │
│  • Separation of concerns by timescale                             │
│  • Graceful degradation on component failure                       │
│  • Real-time guarantees for critical paths                         │
│  • Comprehensive monitoring and logging                            │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### Implementation Checklist

- [ ] Define control architecture (hierarchical/behavior-based/hybrid)
- [ ] Implement ROS 2 nodes for each subsystem
- [ ] Create launch files with proper configuration
- [ ] Build state machine for operating modes
- [ ] Implement lock-free communication primitives
- [ ] Add comprehensive logging and monitoring
- [ ] Write integration tests for data flow
- [ ] Test failure recovery scenarios
- [ ] Profile and optimize latency-critical paths
- [ ] Document system architecture and interfaces

## Further Reading

- Quigley, M. "Programming Robots with ROS 2" (2022)
- Kortenkamp, D. "Building Intelligent Robots" (1998)
- Brooks, R. "A Robust Layered Control System for a Mobile Robot" (1986)
- ROS 2 Design Documentation: https://design.ros2.org
- Real-Time Linux: https://wiki.linuxfoundation.org/realtime
