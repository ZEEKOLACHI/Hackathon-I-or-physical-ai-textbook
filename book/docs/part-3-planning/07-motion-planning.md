---
id: ch-3-07
title: Motion Planning
sidebar_position: 1
difficulty: intermediate
estimated_time: 40
prerequisites: [ch-2-06]
---

# Motion Planning

Motion planning finds collision-free paths for robots to move from start to goal configurations.

## Configuration Space

The configuration space (C-space) represents all possible robot configurations:

```python
import numpy as np

class ConfigurationSpace:
    def __init__(self, robot, obstacles):
        self.robot = robot
        self.obstacles = obstacles

    def is_collision_free(self, config):
        """Check if configuration is collision-free."""
        self.robot.set_config(config)
        for obstacle in self.obstacles:
            if self.robot.collides_with(obstacle):
                return False
        return True
```

## Sampling-Based Planning

### RRT (Rapidly-exploring Random Trees)

```python
class RRT:
    def __init__(self, start, goal, config_space, step_size=0.1):
        self.tree = {tuple(start): None}  # node -> parent
        self.goal = goal
        self.cspace = config_space
        self.step_size = step_size

    def plan(self, max_iterations=1000):
        for _ in range(max_iterations):
            # Sample random configuration
            q_rand = self.sample_random()

            # Find nearest node in tree
            q_near = self.find_nearest(q_rand)

            # Extend toward random sample
            q_new = self.extend(q_near, q_rand)

            if q_new and self.cspace.is_collision_free(q_new):
                self.tree[tuple(q_new)] = tuple(q_near)

                if np.linalg.norm(q_new - self.goal) < self.step_size:
                    return self.extract_path(q_new)

        return None

    def extend(self, q_near, q_rand):
        direction = q_rand - q_near
        distance = np.linalg.norm(direction)
        if distance > self.step_size:
            direction = direction / distance * self.step_size
        return q_near + direction
```

### RRT* (Optimal RRT)

```python
class RRTStar(RRT):
    def __init__(self, *args, rewire_radius=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.rewire_radius = rewire_radius
        self.costs = {}

    def plan(self, max_iterations=1000):
        for _ in range(max_iterations):
            q_rand = self.sample_random()
            q_near = self.find_nearest(q_rand)
            q_new = self.extend(q_near, q_rand)

            if q_new and self.cspace.is_collision_free(q_new):
                # Find best parent
                neighbors = self.find_neighbors(q_new, self.rewire_radius)
                q_min = self.choose_best_parent(q_new, neighbors)

                self.tree[tuple(q_new)] = tuple(q_min)
                self.costs[tuple(q_new)] = self.costs.get(tuple(q_min), 0) + \
                    np.linalg.norm(q_new - q_min)

                # Rewire tree
                self.rewire(q_new, neighbors)

        return self.extract_path(self.find_nearest(self.goal))
```

## A* Search

```python
import heapq

def astar(start, goal, grid, heuristic):
    """A* path planning on a grid."""
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in get_neighbors(current, grid):
            tentative_g = g_score[current] + 1

            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, neighbor))

    return None
```

## MoveIt Integration

```python
import moveit_commander

def plan_arm_motion(target_pose):
    """Plan arm motion using MoveIt."""
    move_group = moveit_commander.MoveGroupCommander("arm")

    move_group.set_pose_target(target_pose)
    plan = move_group.plan()

    if plan[0]:
        move_group.execute(plan[1], wait=True)
        return True
    return False
```

## Summary

- Configuration space abstracts robot motion planning
- RRT efficiently explores high-dimensional spaces
- RRT* provides asymptotically optimal paths
- A* is optimal for grid-based planning
- MoveIt provides planning for robot arms

## Further Reading

- LaValle, S. "Planning Algorithms"
- MoveIt Documentation
