---
id: ch-3-07
title: Motion Planning
sidebar_position: 1
difficulty: intermediate
estimated_time: 95
prerequisites: [ch-2-06]
---

# Motion Planning: Finding Paths Through Possibility

> *"The essence of motion planning is this: given where you are and where you want to be, find a way to get there without bumping into anything. Simple to state, remarkably subtle to solve."*
> — Steven LaValle

When you reach for a cup on a cluttered desk, your brain instantly computes a path that avoids the lamp, weaves around the monitor, and arrives at the target without conscious effort. For robots, this seemingly trivial task requires sophisticated algorithms that search through spaces of unimaginable dimensionality—a 6-DOF arm has a configuration space with over a billion possible positions even at coarse discretization.

## The Motion Planning Problem

### From Workspace to Configuration Space

The key insight of motion planning is the transformation from workspace (physical space) to configuration space (C-space)—the space of all possible robot configurations.

```
Workspace vs. Configuration Space
=================================

    WORKSPACE (Physical):            CONFIGURATION SPACE (Abstract):

    ┌─────────────────────┐          ┌─────────────────────┐
    │                     │          │  ░░░░░░░░░░░░░░░░░ │
    │   ┌───┐    [Robot]  │          │ ░░░░░██████░░░░░░░ │
    │   │Obs│      ◇      │          │ ░░░░████████░░░░░ │
    │   └───┘      │      │          │ ░░░░░██████░░░░░░ │
    │              │      │          │  ░░░░░░░░░░░●━━━━━● │
    │         Start●      │          │  ░░░░░░░░░░░Start Goal│
    │              │      │          │                     │
    │         Goal ●      │          │  θ₁ ────────────▶  │
    └─────────────────────┘          └─────────────────────┘

    3D physical space               2D joint angle space
    Obstacle has fixed shape        Obstacle "grows" by robot size
    Robot is a complex shape        Robot is a single POINT!

    Key insight: Planning for a complex robot in workspace
                 = Planning for a point in C-space
```

**Why C-space Transforms the Problem:**

| Aspect | Workspace | Configuration Space |
|--------|-----------|---------------------|
| Robot shape | Complex geometry | Single point |
| Obstacle shape | Fixed | Expanded by robot envelope |
| Dimension | 3D (position only) | N-DOF (all joint angles) |
| Collision check | Geometry intersection | Point containment |
| Path | Complex curve | Curve in joint space |

### The Piano Mover's Problem

The classic formulation asks: can a piano be moved from one room to another through a house with narrow corridors and doorways? This captures the essence of motion planning.

```
The Piano Mover's Problem
========================

    Room 1                 Corridor                 Room 2
    ┌────────────┬──────┬────────────┬──────┬────────────┐
    │            │      │            │      │            │
    │   ╔════╗   │      │            │      │            │
    │   ║    ║   │ Door │            │ Door │   Goal     │
    │   ║Piano   │  ┃   │            │  ┃   │     ★      │
    │   ╚════╝   │  ┃   │            │  ┃   │            │
    │    Start   │      │            │      │            │
    └────────────┴──────┴────────────┴──────┴────────────┘

    Piano must rotate, translate, and carefully navigate
    through doorways. Some paths exist; most do not.

    C-space: (x, y, θ) - 3 dimensions
    Obstacles in C-space: Complex shapes based on
                          piano geometry at each angle
```

**Motion Planning Problem Variants:**

| Problem Type | Configuration | Constraints | Example |
|--------------|---------------|-------------|---------|
| **Point robot** | (x, y) or (x, y, z) | Avoid obstacles | Mobile robot navigation |
| **Rigid body** | (x, y, z, roll, pitch, yaw) | Avoid obstacles | Drone, floating object |
| **Articulated arm** | (θ₁, θ₂, ..., θₙ) | Joint limits, collisions | Robot manipulator |
| **Humanoid** | 30+ DOF | Balance, self-collision | Walking robot |
| **Multi-robot** | All robot configs | Avoid each other + obstacles | Warehouse fleet |

## Configuration Space Fundamentals

### Representing Robot State

The configuration of a robot is the minimum information needed to specify the position of every point on the robot.

```
Configuration Examples
=====================

    2-DOF Planar Arm:              6-DOF Spatial Arm:

         ● Joint 2                      ● Wrist (3 DOF)
        ╱                              ╱
       ╱  L₂                          ╱
      ● Joint 1                      ● Elbow
     ╱                              ╱
    ╱  L₁                          ╱
   ●═══════                       ● Shoulder (3 DOF)
   Base                           ╲
                                   ╲
                                    ● Base

   Config: q = (θ₁, θ₂)            Config: q = (θ₁, θ₂, θ₃, θ₄, θ₅, θ₆)
   C-space: ℝ² (2D plane)          C-space: ℝ⁶ (6D hypercube)
```

```python
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class JointLimits:
    """Joint angle limits in radians."""
    lower: float
    upper: float

class ConfigurationSpace:
    """
    Represents the configuration space of a robot manipulator.

    C-space is the space of all possible joint configurations.
    Points in C-space are configuration vectors q = (θ₁, ..., θₙ).
    """

    def __init__(self,
                 num_joints: int,
                 joint_limits: List[JointLimits],
                 collision_checker):
        """
        Initialize configuration space.

        Parameters:
        - num_joints: Degrees of freedom
        - joint_limits: Min/max angles for each joint
        - collision_checker: Function(config) -> bool for collision check
        """
        self.ndim = num_joints
        self.limits = joint_limits
        self.collision_checker = collision_checker

    def sample_random(self) -> np.ndarray:
        """
        Sample a random configuration uniformly in C-space.

        Returns configuration vector within joint limits.
        """
        config = np.zeros(self.ndim)
        for i, limit in enumerate(self.limits):
            config[i] = np.random.uniform(limit.lower, limit.upper)
        return config

    def is_valid(self, config: np.ndarray) -> bool:
        """
        Check if configuration is valid (within limits, collision-free).
        """
        # Check joint limits
        for i, (value, limit) in enumerate(zip(config, self.limits)):
            if value < limit.lower or value > limit.upper:
                return False

        # Check collisions
        return not self.collision_checker(config)

    def distance(self, q1: np.ndarray, q2: np.ndarray) -> float:
        """
        Compute distance between configurations.

        For joint angles, weighted Euclidean distance is common.
        Could also use geodesic distance for orientations.
        """
        return np.linalg.norm(q2 - q1)

    def interpolate(self, q1: np.ndarray, q2: np.ndarray,
                    t: float) -> np.ndarray:
        """
        Linearly interpolate between configurations.

        Parameter t in [0, 1]: t=0 gives q1, t=1 gives q2.
        """
        return q1 + t * (q2 - q1)
```

### Obstacle Representation in C-space

Obstacles in workspace create C-space obstacles (C-obstacles) that depend on the robot's shape.

```
C-obstacle Construction
======================

    Workspace:                     C-space (2D for point robot):

    ┌─────────────────────┐        ┌─────────────────────┐
    │                     │        │                     │
    │    ┌─────────┐      │        │    ┌─────────┐      │
    │    │         │      │        │    │ C_obs   │      │
    │    │ Obstacle│      │        │    │(expanded│      │
    │    │         │      │        │    │ by robot│      │
    │    └─────────┘      │        │    │  radius)│      │
    │         ◯           │        │    └─────────┘      │
    │       robot         │        │         ●           │
    │                     │        │    robot = point    │
    └─────────────────────┘        └─────────────────────┘

    For a disk robot of radius r, obstacles expand by r.
    For complex robots, C-obstacles have complex shapes.
```

```python
def check_path_collision(cspace: ConfigurationSpace,
                         q_start: np.ndarray,
                         q_end: np.ndarray,
                         num_checks: int = 10) -> bool:
    """
    Check if the straight-line path between configurations is collision-free.

    Discretizes the path and checks intermediate points.
    Returns True if collision found, False if path is clear.
    """
    for i in range(num_checks + 1):
        t = i / num_checks
        q_interp = cspace.interpolate(q_start, q_end, t)
        if not cspace.is_valid(q_interp):
            return True  # Collision found
    return False  # Path is clear
```

## Sampling-Based Planning

### The Curse of Dimensionality

Grid-based methods that discretize C-space become intractable in high dimensions. A 6-DOF arm with 360 discrete angles per joint would have 360⁶ ≈ 2.2 × 10¹⁵ cells—impossible to store or search exhaustively.

```
The Dimensionality Curse
=======================

    Dimension    Grid Cells (10 divisions)    Storage (8 bytes each)
    ─────────    ─────────────────────────    ──────────────────────
        2             100                          800 bytes
        3           1,000                            8 KB
        4          10,000                           80 KB
        5         100,000                          800 KB
        6       1,000,000                            8 MB
        7      10,000,000                           80 MB
       10    10,000,000,000                         80 GB
       30    10³⁰                                   IMPOSSIBLE

    Humanoids have 30+ DOF → Grid methods fail completely
    Solution: Sample the space, don't enumerate it
```

### RRT: Rapidly-exploring Random Trees

RRT builds a tree by randomly sampling configurations and extending the nearest tree node toward the sample. Its key property is rapid exploration of the configuration space.

```
RRT Algorithm Visualization
==========================

    Start with tree at q_start:

    Step 1: Sample random q_rand      Step 2: Find nearest q_near
    ┌─────────────────────┐           ┌─────────────────────┐
    │                     │           │                     │
    │              ○      │           │              ○      │
    │          q_rand     │           │          q_rand     │
    │                     │           │             ↑       │
    │    ●────●           │           │    ●────●───┘       │
    │  start              │           │  start  q_near      │
    │                     │           │                     │
    └─────────────────────┘           └─────────────────────┘

    Step 3: Extend toward q_rand      Step 4: Add q_new if valid
    ┌─────────────────────┐           ┌─────────────────────┐
    │                     │           │                     │
    │              ○      │           │              ○      │
    │          q_rand     │           │          q_rand     │
    │           ↗         │           │                     │
    │    ●────●──●        │           │    ●────●──●        │
    │  start    q_new     │           │  start    q_new     │
    │                     │           │      (added!)       │
    └─────────────────────┘           └─────────────────────┘

    Repeat until tree reaches goal region
```

```python
from typing import Optional, Dict, List
import numpy as np

class RRT:
    """
    Rapidly-exploring Random Tree planner.

    Builds a tree by iteratively:
    1. Sampling random configurations
    2. Finding the nearest tree node
    3. Extending toward the sample
    4. Adding new node if collision-free
    """

    def __init__(self,
                 cspace: ConfigurationSpace,
                 start: np.ndarray,
                 goal: np.ndarray,
                 step_size: float = 0.1,
                 goal_bias: float = 0.05):
        """
        Initialize RRT planner.

        Parameters:
        - cspace: Configuration space with collision checking
        - start: Starting configuration
        - goal: Goal configuration
        - step_size: Maximum extension distance per step
        - goal_bias: Probability of sampling goal directly
        """
        self.cspace = cspace
        self.start = start
        self.goal = goal
        self.step_size = step_size
        self.goal_bias = goal_bias

        # Tree structure: node -> parent
        self.tree: Dict[tuple, Optional[tuple]] = {tuple(start): None}
        self.nodes: List[np.ndarray] = [start.copy()]

    def sample(self) -> np.ndarray:
        """
        Sample a random configuration.

        With small probability, sample the goal directly
        to bias growth toward the target.
        """
        if np.random.random() < self.goal_bias:
            return self.goal.copy()
        return self.cspace.sample_random()

    def find_nearest(self, q: np.ndarray) -> np.ndarray:
        """
        Find the nearest node in the tree to configuration q.
        """
        distances = [self.cspace.distance(q, node) for node in self.nodes]
        nearest_idx = np.argmin(distances)
        return self.nodes[nearest_idx]

    def steer(self, q_near: np.ndarray, q_rand: np.ndarray) -> np.ndarray:
        """
        Steer from q_near toward q_rand by step_size.

        Returns new configuration at most step_size away from q_near.
        """
        direction = q_rand - q_near
        distance = np.linalg.norm(direction)

        if distance <= self.step_size:
            return q_rand.copy()

        # Normalize and scale by step size
        direction = direction / distance * self.step_size
        return q_near + direction

    def plan(self, max_iterations: int = 10000,
             goal_tolerance: float = 0.1) -> Optional[List[np.ndarray]]:
        """
        Plan a path from start to goal.

        Returns list of configurations forming the path,
        or None if no path found within iteration limit.
        """
        for iteration in range(max_iterations):
            # Sample random configuration
            q_rand = self.sample()

            # Find nearest node in tree
            q_near = self.find_nearest(q_rand)

            # Steer toward sample
            q_new = self.steer(q_near, q_rand)

            # Check if new configuration is valid
            if not self.cspace.is_valid(q_new):
                continue

            # Check if edge is collision-free
            if check_path_collision(self.cspace, q_near, q_new):
                continue

            # Add node to tree
            self.tree[tuple(q_new)] = tuple(q_near)
            self.nodes.append(q_new.copy())

            # Check if goal reached
            if self.cspace.distance(q_new, self.goal) < goal_tolerance:
                return self._extract_path(q_new)

            # Progress reporting
            if iteration % 1000 == 0:
                print(f"Iteration {iteration}, nodes: {len(self.nodes)}")

        print(f"Failed to find path after {max_iterations} iterations")
        return None

    def _extract_path(self, q_goal: np.ndarray) -> List[np.ndarray]:
        """
        Extract path from start to goal by following parent pointers.
        """
        path = [q_goal.copy()]
        current = tuple(q_goal)

        while self.tree[current] is not None:
            current = self.tree[current]
            path.append(np.array(current))

        path.reverse()
        return path
```

### RRT*: Optimal RRT

RRT provides paths quickly but they are often suboptimal. RRT* adds two key improvements: choosing the best parent among nearby nodes, and rewiring the tree when better paths are found.

```
RRT* vs RRT Path Quality
=======================

    RRT Path (first found):         RRT* Path (optimized):

    ┌─────────────────────┐         ┌─────────────────────┐
    │                     │         │                     │
    │    Goal ●           │         │    Goal ●           │
    │        ╱            │         │        │            │
    │       ╱             │         │        │            │
    │      ╱──╮           │         │        │            │
    │     ╱   │           │         │        │            │
    │    ╱    │           │         │        │            │
    │   ●─────╯           │         │   ●────╯            │
    │   Start             │         │   Start             │
    └─────────────────────┘         └─────────────────────┘

    Path has unnecessary                Path approaches
    turns and detours                   optimal straight line

    RRT*: Asymptotically optimal
    As iterations → ∞, path cost → optimal cost
```

```python
class RRTStar(RRT):
    """
    RRT* (Optimal RRT) planner.

    Improvements over RRT:
    1. Choose best parent among nearby nodes
    2. Rewire tree when new node provides shorter paths
    3. Asymptotically converges to optimal path
    """

    def __init__(self, *args, rewire_radius: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.rewire_radius = rewire_radius
        self.costs: Dict[tuple, float] = {tuple(self.start): 0.0}

    def find_near(self, q: np.ndarray, radius: float) -> List[np.ndarray]:
        """
        Find all nodes within radius of configuration q.
        """
        near_nodes = []
        for node in self.nodes:
            if self.cspace.distance(q, node) <= radius:
                near_nodes.append(node)
        return near_nodes

    def cost(self, q: np.ndarray) -> float:
        """Get cost to reach configuration q from start."""
        return self.costs.get(tuple(q), float('inf'))

    def edge_cost(self, q1: np.ndarray, q2: np.ndarray) -> float:
        """Cost of edge between two configurations."""
        return self.cspace.distance(q1, q2)

    def choose_best_parent(self, q_new: np.ndarray,
                           q_near: np.ndarray,
                           near_nodes: List[np.ndarray]) -> np.ndarray:
        """
        Choose the parent that minimizes cost to q_new.

        Among all nearby nodes, pick the one that gives
        the lowest total cost from start to q_new.
        """
        best_parent = q_near
        best_cost = self.cost(q_near) + self.edge_cost(q_near, q_new)

        for node in near_nodes:
            if check_path_collision(self.cspace, node, q_new):
                continue

            candidate_cost = self.cost(node) + self.edge_cost(node, q_new)
            if candidate_cost < best_cost:
                best_parent = node
                best_cost = candidate_cost

        return best_parent

    def rewire(self, q_new: np.ndarray, near_nodes: List[np.ndarray]) -> None:
        """
        Rewire tree: check if q_new provides better paths to nearby nodes.

        If reaching a neighbor through q_new is cheaper than
        its current path, update the tree.
        """
        new_cost = self.cost(q_new)

        for node in near_nodes:
            if np.array_equal(node, q_new):
                continue

            if check_path_collision(self.cspace, q_new, node):
                continue

            candidate_cost = new_cost + self.edge_cost(q_new, node)
            if candidate_cost < self.cost(node):
                # Rewire: node's new parent is q_new
                self.tree[tuple(node)] = tuple(q_new)
                self.costs[tuple(node)] = candidate_cost

    def plan(self, max_iterations: int = 10000,
             goal_tolerance: float = 0.1) -> Optional[List[np.ndarray]]:
        """
        Plan an optimized path from start to goal.

        Continues optimizing even after finding initial solution.
        """
        best_path = None
        best_cost = float('inf')

        for iteration in range(max_iterations):
            # Sample random configuration
            q_rand = self.sample()

            # Find nearest node
            q_near = self.find_nearest(q_rand)

            # Steer toward sample
            q_new = self.steer(q_near, q_rand)

            if not self.cspace.is_valid(q_new):
                continue

            # Find nearby nodes for rewiring
            near_nodes = self.find_near(q_new, self.rewire_radius)

            # Choose best parent
            q_parent = self.choose_best_parent(q_new, q_near, near_nodes)

            if check_path_collision(self.cspace, q_parent, q_new):
                continue

            # Add node with best parent
            self.tree[tuple(q_new)] = tuple(q_parent)
            self.costs[tuple(q_new)] = (
                self.cost(q_parent) + self.edge_cost(q_parent, q_new)
            )
            self.nodes.append(q_new.copy())

            # Rewire nearby nodes
            self.rewire(q_new, near_nodes)

            # Check if goal reached with better cost
            if self.cspace.distance(q_new, self.goal) < goal_tolerance:
                path = self._extract_path(q_new)
                path_cost = self.cost(q_new)
                if path_cost < best_cost:
                    best_path = path
                    best_cost = path_cost
                    print(f"Iteration {iteration}: New best cost = {best_cost:.3f}")

        return best_path
```

### Informed RRT*: Focusing the Search

Once an initial solution is found, search can be focused on the ellipsoidal region that could contain better solutions.

```
Informed RRT* Search Focus
=========================

    Before solution:                After solution found:

    ┌─────────────────────┐         ┌─────────────────────┐
    │ . . . . . . . . . . │         │         .....       │
    │ . . . . . . . . . . │         │       ...●●●...     │
    │ . . . ● . . . . . . │         │      ..●●●●●●..     │
    │ . . . Goal. . . . . │         │     ...●●●●●...     │
    │ . . . . . . . . . . │         │      ...●●●●..      │
    │ . . . . . . . . . . │         │       ..●●●..       │
    │ . . . . . . . . . . │         │        ..●..        │
    │ ● . . . . . . . . . │         │        Start        │
    │ Start               │         │                     │
    └─────────────────────┘         └─────────────────────┘

    Sample everywhere                Sample only in ellipse
    (uniform random)                 (can improve solution)

    Ellipse foci: start and goal
    Ellipse size: based on current best path length
```

## Graph-Based Planning: A* and Variants

### A* Search Algorithm

A* is optimal for graph search, combining the best of uniform cost search (optimality) and greedy best-first search (efficiency).

```
A* Algorithm Intuition
=====================

    f(n) = g(n) + h(n)

    ┌──────────────────────────────────────────────────────┐
    │                                                      │
    │   Start ●────────────────●────────────────● Goal     │
    │         ◄─── g(n) ───► n ◄─── h(n) ───►             │
    │                                                      │
    │   g(n): Actual cost from start to n (known)         │
    │   h(n): Estimated cost from n to goal (heuristic)   │
    │   f(n): Estimated total path cost through n         │
    │                                                      │
    └──────────────────────────────────────────────────────┘

    A* always expands the node with minimum f(n)

    Key property: If h(n) never overestimates true cost,
                  A* is guaranteed to find optimal path.
```

**Heuristic Properties:**

| Heuristic Type | Property | Example |
|----------------|----------|---------|
| **Admissible** | h(n) ≤ true cost | Straight-line distance |
| **Consistent** | h(n) ≤ cost(n,m) + h(m) | Most geometric heuristics |
| **Inadmissible** | h(n) > true cost (sometimes) | Weighted A* (faster, suboptimal) |

```python
import heapq
from typing import Dict, List, Tuple, Optional, Callable

def astar_search(start: Tuple,
                 goal: Tuple,
                 get_neighbors: Callable[[Tuple], List[Tuple]],
                 heuristic: Callable[[Tuple, Tuple], float],
                 edge_cost: Callable[[Tuple, Tuple], float] = lambda a, b: 1
                 ) -> Optional[List[Tuple]]:
    """
    A* path planning algorithm.

    Parameters:
    - start: Starting node
    - goal: Goal node
    - get_neighbors: Function returning list of neighbors for a node
    - heuristic: Function h(node, goal) estimating cost to goal
    - edge_cost: Function returning cost between adjacent nodes

    Returns path as list of nodes, or None if no path exists.
    """
    # Priority queue: (f_score, node)
    open_set = [(heuristic(start, goal), start)]
    heapq.heapify(open_set)

    # Track visited nodes and best paths
    came_from: Dict[Tuple, Tuple] = {}
    g_score: Dict[Tuple, float] = {start: 0}
    f_score: Dict[Tuple, float] = {start: heuristic(start, goal)}

    # Track nodes in open set for efficient membership check
    open_set_hash = {start}

    while open_set:
        # Pop node with lowest f_score
        _, current = heapq.heappop(open_set)
        open_set_hash.discard(current)

        # Goal reached
        if current == goal:
            return _reconstruct_path(came_from, current)

        # Explore neighbors
        for neighbor in get_neighbors(current):
            tentative_g = g_score[current] + edge_cost(current, neighbor)

            # Found better path to neighbor
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)

                if neighbor not in open_set_hash:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    open_set_hash.add(neighbor)

    return None  # No path found

def _reconstruct_path(came_from: Dict[Tuple, Tuple],
                      current: Tuple) -> List[Tuple]:
    """Reconstruct path by following came_from pointers."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path
```

### Grid-Based A* for Mobile Robots

For 2D navigation, A* on a discretized grid is highly effective.

```
Grid-Based Path Planning
=======================

    Occupancy Grid:                  A* Path Found:

    ┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┐           ┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┐
    │S│ │ │ │ │█│ │ │ │G│           │●│→│→│→│↓│█│←│←│←│●│
    ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤           ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
    │ │ │█│█│ │█│ │█│ │ │           │ │ │█│█│↓│█│↑│█│ │ │
    ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤           ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
    │ │ │█│ │ │ │ │█│ │ │           │ │ │█│ │↓│→│↑│█│ │ │
    ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤           ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
    │ │ │ │ │█│ │ │ │ │ │           │ │ │ │ │█│→│↑│ │ │ │
    ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤           ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
    │ │ │ │ │█│ │ │ │ │ │           │ │ │ │ │█│→│↑│ │ │ │
    └─┴─┴─┴─┴─┴─┴─┴─┴─┴─┘           └─┴─┴─┴─┴─┴─┴─┴─┴─┴─┘

    S = Start, G = Goal               Arrows show path
    █ = Obstacle                      from S to G
```

```python
from typing import Tuple, List

def grid_astar(grid: np.ndarray,
               start: Tuple[int, int],
               goal: Tuple[int, int],
               allow_diagonal: bool = True) -> Optional[List[Tuple[int, int]]]:
    """
    A* path planning on a 2D occupancy grid.

    Parameters:
    - grid: 2D array where 0 = free, 1 = obstacle
    - start: Starting cell (row, col)
    - goal: Goal cell (row, col)
    - allow_diagonal: Whether to allow diagonal movement

    Returns path as list of cells, or None if no path exists.
    """
    rows, cols = grid.shape

    def get_neighbors(cell: Tuple[int, int]) -> List[Tuple[int, int]]:
        r, c = cell
        neighbors = []

        # 4-connected moves
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        # Add diagonal moves if allowed
        if allow_diagonal:
            moves += [(1, 1), (1, -1), (-1, 1), (-1, -1)]

        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if grid[nr, nc] == 0:  # Free cell
                    neighbors.append((nr, nc))

        return neighbors

    def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        # Euclidean distance (admissible for 8-connected)
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def edge_cost(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        # Diagonal moves cost √2, cardinal moves cost 1
        if a[0] != b[0] and a[1] != b[1]:
            return np.sqrt(2)
        return 1.0

    return astar_search(start, goal, get_neighbors, heuristic, edge_cost)
```

### D* Lite: Replanning for Dynamic Environments

In dynamic environments where obstacles appear or the map changes, D* Lite efficiently replans by reusing computation from previous searches.

```
D* Lite for Dynamic Replanning
=============================

    Initial Path:                    Obstacle Appears:
    ┌─────────────────────┐          ┌─────────────────────┐
    │                     │          │                     │
    │    ●────────────●   │          │    ●─────┬────×     │
    │ Start          Goal │          │ Start    │    Goal  │
    │                     │          │          │   ▓▓▓    │
    │                     │          │          │   NEW    │
    └─────────────────────┘          └─────────────────────┘

    Path found                       Path blocked!

    D* Lite Replan:                  Key Advantage:
    ┌─────────────────────┐          ┌─────────────────────┐
    │                     │          │                     │
    │    ●──┬──────────●  │          │ Only recomputes     │
    │ Start │         Goal│          │ affected region     │
    │       └─────────────│          │                     │
    │           ▓▓▓       │          │ Much faster than    │
    └─────────────────────┘          │ replanning from     │
                                     │ scratch!            │
    New path found quickly           └─────────────────────┘
```

## Path Smoothing and Optimization

### Raw Paths Need Refinement

Paths from sampling-based planners are often jagged and inefficient. Smoothing improves path quality for execution.

```
Path Smoothing Process
=====================

    Raw RRT Path:                    After Smoothing:

    ●───┐                            ●
    Start                           Start
        └───┐                           ╲
            │                            ╲
            └───┐                         ╲
                │                          ╲
                └───┐                       ╲
                    │                        ╲
                    └───● Goal                ● Goal

    Many unnecessary turns            Direct path where possible
```

```python
def shortcut_path(cspace: ConfigurationSpace,
                  path: List[np.ndarray],
                  max_iterations: int = 100) -> List[np.ndarray]:
    """
    Shorten path by removing unnecessary waypoints.

    Repeatedly tries to connect non-adjacent waypoints directly.
    If collision-free, removes intermediate waypoints.
    """
    if len(path) <= 2:
        return path

    smoothed = path.copy()

    for _ in range(max_iterations):
        if len(smoothed) <= 2:
            break

        # Pick two random non-adjacent points
        i = np.random.randint(0, len(smoothed) - 2)
        j = np.random.randint(i + 2, len(smoothed))

        # Check if direct connection is collision-free
        if not check_path_collision(cspace, smoothed[i], smoothed[j]):
            # Remove intermediate waypoints
            smoothed = smoothed[:i+1] + smoothed[j:]

    return smoothed

def interpolate_path(path: List[np.ndarray],
                     max_segment_length: float = 0.05) -> List[np.ndarray]:
    """
    Densify path by adding intermediate waypoints.

    Ensures no segment exceeds max_segment_length.
    Required before executing path on real robot.
    """
    dense_path = [path[0]]

    for i in range(len(path) - 1):
        start = path[i]
        end = path[i + 1]
        distance = np.linalg.norm(end - start)

        if distance > max_segment_length:
            num_segments = int(np.ceil(distance / max_segment_length))
            for j in range(1, num_segments):
                t = j / num_segments
                intermediate = start + t * (end - start)
                dense_path.append(intermediate)

        dense_path.append(end)

    return dense_path
```

## MoveIt: Industrial-Strength Motion Planning

### MoveIt Architecture

MoveIt is the standard motion planning framework in ROS, integrating planning, kinematics, collision checking, and execution.

```
MoveIt System Architecture
=========================

    ┌─────────────────────────────────────────────────────────┐
    │                      User Request                        │
    │                  "Move arm to pose X"                    │
    └─────────────────────────┬───────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────┐
    │                     Move Group                           │
    │            (High-level planning interface)               │
    └───────┬─────────────────┬─────────────────┬─────────────┘
            │                 │                 │
            ▼                 ▼                 ▼
    ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
    │   Planning    │ │   Collision   │ │   Kinematics  │
    │   Pipeline    │ │   Checking    │ │   (IK/FK)     │
    │               │ │               │ │               │
    │  OMPL, STOMP  │ │  FCL, Bullet  │ │  KDL, ikfast  │
    └───────┬───────┘ └───────────────┘ └───────────────┘
            │
            ▼
    ┌─────────────────────────────────────────────────────────┐
    │                  Trajectory Controller                   │
    │              (Execute on real robot)                     │
    └─────────────────────────────────────────────────────────┘
```

### Using MoveIt in Python

```python
import moveit_commander
import geometry_msgs.msg
from typing import List, Optional

class ArmMotionPlanner:
    """
    High-level interface for arm motion planning using MoveIt.
    """

    def __init__(self, group_name: str = "arm"):
        """
        Initialize MoveIt interface.

        Parameters:
        - group_name: Name of the move group (from SRDF)
        """
        moveit_commander.roscpp_initialize([])
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.move_group = moveit_commander.MoveGroupCommander(group_name)

        # Planning parameters
        self.move_group.set_planning_time(10.0)
        self.move_group.set_num_planning_attempts(5)
        self.move_group.set_max_velocity_scaling_factor(0.5)
        self.move_group.set_max_acceleration_scaling_factor(0.5)

    def plan_to_pose(self, target_pose: geometry_msgs.msg.Pose
                     ) -> Optional[moveit_commander.RobotTrajectory]:
        """
        Plan motion to reach target end-effector pose.

        Returns trajectory if planning succeeds, None otherwise.
        """
        self.move_group.set_pose_target(target_pose)

        # Plan (returns tuple: success, trajectory, planning_time, error_code)
        success, trajectory, _, _ = self.move_group.plan()

        if success:
            return trajectory
        return None

    def plan_to_joint_values(self, joint_values: List[float]
                             ) -> Optional[moveit_commander.RobotTrajectory]:
        """
        Plan motion to reach target joint configuration.
        """
        self.move_group.set_joint_value_target(joint_values)
        success, trajectory, _, _ = self.move_group.plan()

        if success:
            return trajectory
        return None

    def plan_cartesian_path(self, waypoints: List[geometry_msgs.msg.Pose],
                            eef_step: float = 0.01,
                            jump_threshold: float = 0.0
                            ) -> Optional[moveit_commander.RobotTrajectory]:
        """
        Plan a Cartesian path through waypoints.

        The end-effector follows a straight line between waypoints.
        Useful for tasks requiring specific end-effector trajectories.
        """
        trajectory, fraction = self.move_group.compute_cartesian_path(
            waypoints,
            eef_step,        # Resolution of path
            jump_threshold   # Disable jump detection
        )

        if fraction >= 0.99:  # 99%+ of path achieved
            return trajectory
        return None

    def execute(self, trajectory: moveit_commander.RobotTrajectory,
                wait: bool = True) -> bool:
        """
        Execute a planned trajectory.

        Parameters:
        - trajectory: Planned trajectory from plan_* methods
        - wait: If True, block until execution completes
        """
        return self.move_group.execute(trajectory, wait=wait)

    def move_to_named_target(self, name: str) -> bool:
        """
        Move to a predefined pose from SRDF.

        Common names: "home", "ready", "extended"
        """
        self.move_group.set_named_target(name)
        return self.move_group.go(wait=True)

    def add_collision_object(self, name: str,
                             pose: geometry_msgs.msg.PoseStamped,
                             size: List[float]) -> None:
        """
        Add a box obstacle to the planning scene.
        """
        self.scene.add_box(name, pose, size)

    def remove_collision_object(self, name: str) -> None:
        """
        Remove an obstacle from the planning scene.
        """
        self.scene.remove_world_object(name)
```

## Industry Perspectives: Motion Planning in Practice

### Warehouse Automation

Amazon Robotics and similar systems plan paths for thousands of robots simultaneously:

```
Warehouse Multi-Robot Planning
=============================

    ┌──────────────────────────────────────────────────────┐
    │  ● → → ●     Shelf    ● → → → → → ●                 │
    │  ↓       ↓            │           ↑                  │
    │  ●   Shelf   ● → → → ●   Shelf   ●                  │
    │  ↓           ↑                    ↓                  │
    │  ● → → → → → ●   Shelf   ● ← ← ← ●                  │
    │                          ↓                           │
    │  Pick Station    ● ← ← ← ●    Drop Station          │
    └──────────────────────────────────────────────────────┘

    Challenges:
    - Thousands of robots moving simultaneously
    - Dynamic environment (humans, forklifts)
    - Throughput optimization (not just collision avoidance)
    - Deadlock prevention
```

### Surgical Robotics

The da Vinci surgical system performs motion planning with extreme precision:

| Requirement | Solution |
|-------------|----------|
| Submillimeter accuracy | High-resolution encoders, redundant sensing |
| Constrained workspace | RCM (Remote Center of Motion) constraints |
| Collision avoidance | Real-time distance monitoring |
| Smooth motion | B-spline trajectory generation |

### Humanoid Motion

Humanoid robots face unique challenges combining locomotion and manipulation:

| Challenge | Planning Approach |
|-----------|-------------------|
| High DOF (30+) | Hierarchical planning, posture optimization |
| Dynamic balance | Preview control, ZMP constraints |
| Multi-contact | Contact planning before motion planning |
| Whole-body | Prioritized task-space control |

## Summary: The Motion Planning Toolbox

Motion planning is the bridge between high-level goals and robot action. The key concepts we've covered:

**Key Takeaways:**

1. **Configuration space transforms the problem**: Planning for a complex robot becomes planning for a point in C-space.

2. **Sampling beats enumeration**: In high dimensions, random sampling explores efficiently where grids fail.

3. **RRT explores, RRT\* optimizes**: Use RRT for feasibility, RRT* when path quality matters.

4. **A* is optimal for graphs**: When you can discretize, A* with admissible heuristics guarantees optimal paths.

5. **Smoothing improves execution**: Raw planned paths need refinement for efficient robot motion.

6. **MoveIt integrates everything**: Production systems benefit from integrated planning, collision checking, and execution.

```
Choosing a Planning Algorithm
============================

                        High Dimensional?
                              │
                    ┌─────────┴─────────┐
                   Yes                  No
                    │                    │
              Sampling-Based        Graph-Based
                    │                    │
           ┌───────┴───────┐     ┌──────┴──────┐
          Any            Optimal  Grid      Graph
         Solution        Path    Search    Search
           │              │        │          │
          RRT          RRT*      A*      Dijkstra
                               D* Lite
```

---

## Further Reading

**Foundational Texts:**
- LaValle, S. "Planning Algorithms" (2006) - Free online, comprehensive
- Choset et al., "Principles of Robot Motion" - Theory and practice
- Latombe, J.C., "Robot Motion Planning" - Classical treatment

**Key Papers:**
- LaValle & Kuffner, "RRT: Rapidly-exploring Random Trees" (1998)
- Karaman & Frazzoli, "RRT*: Optimal Motion Planning" (2011)
- Hart, Nilsson & Raphael, "A* Algorithm" (1968)

**Software Resources:**
- [OMPL (Open Motion Planning Library)](https://ompl.kavrakilab.org/)
- [MoveIt Documentation](https://moveit.ros.org/)
- [nav2 (ROS 2 Navigation)](https://navigation.ros.org/)

**Online Courses:**
- Stanford CS326 - Motion Planning
- MIT 6.881 - Robotic Manipulation (motion planning chapters)
