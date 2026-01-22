---
id: ch-3-08
title: Task Planning
sidebar_position: 2
difficulty: intermediate
estimated_time: 35
prerequisites: [ch-3-07]
---

# Task Planning

Task planning determines the sequence of high-level actions to achieve complex goals.

## PDDL (Planning Domain Definition Language)

### Domain Definition

```lisp
(define (domain robot-manipulation)
  (:requirements :strips :typing)

  (:types
    object location gripper
  )

  (:predicates
    (at ?obj - object ?loc - location)
    (holding ?g - gripper ?obj - object)
    (gripper-empty ?g - gripper)
    (clear ?loc - location)
  )

  (:action pick
    :parameters (?obj - object ?loc - location ?g - gripper)
    :precondition (and
      (at ?obj ?loc)
      (gripper-empty ?g)
    )
    :effect (and
      (holding ?g ?obj)
      (not (at ?obj ?loc))
      (not (gripper-empty ?g))
      (clear ?loc)
    )
  )

  (:action place
    :parameters (?obj - object ?loc - location ?g - gripper)
    :precondition (and
      (holding ?g ?obj)
      (clear ?loc)
    )
    :effect (and
      (at ?obj ?loc)
      (gripper-empty ?g)
      (not (holding ?g ?obj))
      (not (clear ?loc))
    )
  )
)
```

### Problem Definition

```lisp
(define (problem sort-objects)
  (:domain robot-manipulation)

  (:objects
    cube sphere - object
    table bin - location
    gripper1 - gripper
  )

  (:init
    (at cube table)
    (at sphere table)
    (gripper-empty gripper1)
    (clear bin)
  )

  (:goal (and
    (at cube bin)
    (at sphere bin)
  ))
)
```

## Hierarchical Task Networks (HTN)

```python
class HTNPlanner:
    def __init__(self, domain):
        self.domain = domain

    def plan(self, task, state):
        """Decompose task into primitive actions."""
        if self.is_primitive(task):
            if self.applicable(task, state):
                return [task]
            return None

        for method in self.domain.methods[task.name]:
            if method.preconditions_met(state):
                subtasks = method.decompose(task)
                plan = []
                current_state = state.copy()

                for subtask in subtasks:
                    subplan = self.plan(subtask, current_state)
                    if subplan is None:
                        break
                    plan.extend(subplan)
                    current_state = self.apply_actions(subplan, current_state)
                else:
                    return plan

        return None
```

## Integration with Motion Planning

```python
class TaskMotionPlanner:
    def __init__(self, task_planner, motion_planner):
        self.task_planner = task_planner
        self.motion_planner = motion_planner

    def plan(self, goal, initial_state):
        # Get symbolic task plan
        task_plan = self.task_planner.plan(goal, initial_state)

        if task_plan is None:
            return None

        # Generate motion plans for each action
        full_plan = []
        for action in task_plan:
            motion = self.motion_planner.plan(
                action.start_config,
                action.goal_config
            )
            if motion is None:
                # Backtrack or replan
                return self.replan(action, task_plan)
            full_plan.append((action, motion))

        return full_plan
```

## Summary

- PDDL is a standard language for task planning
- HTN decomposes complex tasks hierarchically
- Task and motion planning must be integrated
- Symbolic reasoning guides geometric execution

## Further Reading

- Ghallab, M. "Automated Planning: Theory and Practice"
- ROSPlan Documentation
