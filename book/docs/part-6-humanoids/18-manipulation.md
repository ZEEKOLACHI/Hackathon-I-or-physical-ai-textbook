---
id: ch-6-18
title: Humanoid Manipulation
sidebar_position: 3
difficulty: advanced
estimated_time: 40
prerequisites: [ch-6-16, ch-6-17]
---

# Humanoid Manipulation

Humanoid manipulation combines whole-body control with dexterous object handling.

## Dual-Arm Coordination

```python
class DualArmController:
    def __init__(self, left_arm, right_arm):
        self.left = left_arm
        self.right = right_arm

    def coordinated_motion(self, left_target, right_target, constraint='relative'):
        """
        Plan coordinated motion for both arms.
        """
        if constraint == 'relative':
            # Maintain relative pose between end-effectors
            relative_pose = self.compute_relative_pose(left_target, right_target)
            return self.plan_with_constraint(left_target, right_target, relative_pose)

        elif constraint == 'symmetric':
            # Mirror motion
            return self.plan_symmetric(left_target)

        else:
            # Independent motion
            left_plan = self.left.plan(left_target)
            right_plan = self.right.plan(right_target)
            return left_plan, right_plan
```

## Grasp Planning

```python
class GraspPlanner:
    def __init__(self, hand_model):
        self.hand = hand_model

    def plan_grasp(self, object_mesh, approach='top'):
        """Plan grasp configuration for object."""
        # Sample grasp candidates
        candidates = self.sample_grasps(object_mesh, approach)

        # Evaluate grasps
        scored_grasps = []
        for grasp in candidates:
            score = self.evaluate_grasp(grasp, object_mesh)
            scored_grasps.append((score, grasp))

        # Return best grasp
        scored_grasps.sort(reverse=True)
        return scored_grasps[0][1]

    def evaluate_grasp(self, grasp, object_mesh):
        """Score grasp quality using force closure."""
        contacts = self.compute_contacts(grasp, object_mesh)
        return compute_force_closure(contacts)
```

## Mobile Manipulation

```python
class MobileManipulator:
    def __init__(self, base, arm):
        self.base = base
        self.arm = arm

    def reach_target(self, target_pose):
        """Coordinate base and arm motion."""
        # Check if target is reachable from current base pose
        if self.arm.is_reachable(target_pose):
            return self.arm.plan(target_pose)

        # Find base pose that makes target reachable
        base_pose = self.find_base_pose(target_pose)

        # Plan base motion, then arm motion
        base_plan = self.base.plan(base_pose)
        arm_plan = self.arm.plan(target_pose, base_pose)

        return base_plan, arm_plan
```

## Summary

- Dual-arm coordination enables complex tasks
- Grasp planning considers force closure
- Mobile manipulation extends workspace
- Whole-body coordination is essential

## Further Reading

- Murray, R.M. "A Mathematical Introduction to Robotic Manipulation"
