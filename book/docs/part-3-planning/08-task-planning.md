---
id: ch-3-08
title: Task Planning
sidebar_position: 2
difficulty: intermediate
estimated_time: 85
prerequisites: [ch-3-07]
---

# Task Planning: Reasoning About Actions and Goals

> *"Before you can move, you must decide where to go. Before you can grasp, you must decide what to grasp. Task planning is the art of deciding what to do—motion planning is merely the art of doing it."*

A robot tasked with making coffee faces a challenge that motion planning alone cannot solve. It must reason about the sequence of actions: fill the reservoir, add grounds, insert filter, press the button, wait, pour into cup. Each action has preconditions (the reservoir must be filled before brewing) and effects (pressing the button starts the brewing process). Task planning addresses this symbolic reasoning layer that sits above geometric motion planning.

## The Task Planning Problem

### From Goals to Actions

Task planning bridges the gap between what we want (goals) and how to achieve it (action sequences). Unlike motion planning which operates in continuous configuration space, task planning operates in discrete state space with symbolic actions.

```
Task Planning vs. Motion Planning
================================

    TASK PLANNING (Symbolic)          MOTION PLANNING (Geometric)

    Goal: "Coffee in cup"             Goal: Arm at position (x,y,z)

    State: {water_in_reservoir,       State: Joint angles (θ₁...θ₆)
            grounds_added,
            cup_placed, ...}

    Actions: FILL, ADD_GROUNDS,       Actions: Continuous trajectories
             BREW, POUR                        through C-space

    ┌─────────────────────┐           ┌─────────────────────┐
    │ State₀              │           │                     │
    │   ↓ action₁         │           │  Start ●            │
    │ State₁              │           │         ╲           │
    │   ↓ action₂         │           │          ╲          │
    │ State₂              │           │           ╲         │
    │   ↓ action₃         │           │            ● Goal   │
    │ Goal State          │           │                     │
    └─────────────────────┘           └─────────────────────┘

    Discrete state transitions        Continuous path in space
```

**Key Differences:**

| Aspect | Task Planning | Motion Planning |
|--------|---------------|-----------------|
| State space | Discrete, symbolic | Continuous, geometric |
| Actions | Named operators with pre/post conditions | Trajectories through C-space |
| Search | Graph/tree search in state space | Sampling or graph search in C-space |
| Representation | Logic, predicates | Configurations, transforms |
| Challenge | Combinatorial explosion | Curse of dimensionality |

### The Classical Planning Model

Classical planning makes several simplifying assumptions that enable tractable reasoning:

```
Classical Planning Assumptions
=============================

    1. FINITE STATE SPACE
       ┌─────────────────────────────────────┐
       │  S₀ ──▶ S₁ ──▶ S₂ ──▶ ... ──▶ Sₙ   │
       │  Discrete, enumerable states        │
       └─────────────────────────────────────┘

    2. FULLY OBSERVABLE
       ┌─────────────────────────────────────┐
       │  Robot knows exact current state    │
       │  No hidden information              │
       └─────────────────────────────────────┘

    3. DETERMINISTIC
       ┌─────────────────────────────────────┐
       │  Action a in state s always gives   │
       │  the same result state s'           │
       │  s ──a──▶ s' (no uncertainty)       │
       └─────────────────────────────────────┘

    4. STATIC ENVIRONMENT
       ┌─────────────────────────────────────┐
       │  World only changes due to robot    │
       │  actions (no external events)       │
       └─────────────────────────────────────┘

    5. GOAL-BASED
       ┌─────────────────────────────────────┐
       │  Success = reaching goal state      │
       │  No preferences among goal states   │
       └─────────────────────────────────────┘
```

## PDDL: The Language of Planning

### Understanding PDDL Structure

PDDL (Planning Domain Definition Language) is the standard language for expressing planning problems. It separates the domain (general rules) from the problem (specific instance).

```
PDDL Structure
=============

    ┌─────────────────────────────────────────────────────────┐
    │                     DOMAIN FILE                          │
    │                                                          │
    │  Defines WHAT IS POSSIBLE in this world:                │
    │  - Types of objects                                      │
    │  - Predicates (properties and relations)                 │
    │  - Actions (operators with preconditions/effects)        │
    │                                                          │
    │  Example: "Robots can pick and place objects"            │
    └─────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────┐
    │                    PROBLEM FILE                          │
    │                                                          │
    │  Defines WHAT WE WANT in this specific case:            │
    │  - Specific objects                                      │
    │  - Initial state                                         │
    │  - Goal state                                            │
    │                                                          │
    │  Example: "Move cube from table to bin"                  │
    └─────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────┐
    │                       PLANNER                            │
    │                                                          │
    │  Searches for sequence of actions that transforms        │
    │  initial state into goal state                           │
    │                                                          │
    │  Output: [pick(cube), move(table,bin), place(cube)]     │
    └─────────────────────────────────────────────────────────┘
```

### Domain Definition

The domain defines the types, predicates, and actions available in a planning problem.

```lisp
;; Domain: Robot manipulation in a warehouse environment
;; This defines what the robot CAN do (not what it WILL do)

(define (domain warehouse-robot)
  ;; Required PDDL features
  (:requirements :strips :typing :negative-preconditions)

  ;; Type hierarchy
  (:types
    physical - object           ; Physical things in the world
    location - object           ; Places where things can be
    robot - physical            ; The robot itself
    item - physical             ; Things the robot can manipulate
    container - location        ; Storage locations (bins, shelves)
    surface - location          ; Work surfaces (tables)
  )

  ;; Predicates define the state of the world
  ;; Each predicate is either true or false for given arguments
  (:predicates
    ;; Location predicates
    (at ?thing - physical ?loc - location)      ; Thing is at location
    (robot-at ?r - robot ?loc - location)       ; Robot position

    ;; Gripper predicates
    (holding ?r - robot ?item - item)           ; Robot holds item
    (gripper-empty ?r - robot)                  ; Gripper is free

    ;; Location state predicates
    (clear ?loc - location)                     ; Nothing at location
    (accessible ?loc - location)                ; Robot can reach it

    ;; Item properties
    (fragile ?item - item)                      ; Item is fragile
    (heavy ?item - item)                        ; Item is heavy
  )

  ;; Actions define how the robot changes the world
  ;; Each action has parameters, preconditions, and effects

  (:action move
    :parameters (?r - robot ?from - location ?to - location)
    :precondition (and
      (robot-at ?r ?from)
      (accessible ?to)
    )
    :effect (and
      (robot-at ?r ?to)
      (not (robot-at ?r ?from))
    )
  )

  (:action pick
    :parameters (?r - robot ?item - item ?loc - location)
    :precondition (and
      (robot-at ?r ?loc)
      (at ?item ?loc)
      (gripper-empty ?r)
      (not (heavy ?item))        ; Can't pick heavy items directly
    )
    :effect (and
      (holding ?r ?item)
      (not (at ?item ?loc))
      (not (gripper-empty ?r))
      (clear ?loc)
    )
  )

  (:action place
    :parameters (?r - robot ?item - item ?loc - location)
    :precondition (and
      (robot-at ?r ?loc)
      (holding ?r ?item)
      (clear ?loc)
    )
    :effect (and
      (at ?item ?loc)
      (gripper-empty ?r)
      (not (holding ?r ?item))
      (not (clear ?loc))
    )
  )

  (:action place-careful
    :parameters (?r - robot ?item - item ?loc - location)
    :precondition (and
      (robot-at ?r ?loc)
      (holding ?r ?item)
      (clear ?loc)
      (fragile ?item)            ; Only for fragile items
    )
    :effect (and
      (at ?item ?loc)
      (gripper-empty ?r)
      (not (holding ?r ?item))
      (not (clear ?loc))
    )
  )
)
```

### Problem Definition

The problem specifies a concrete instance with specific objects, initial state, and goals.

```lisp
;; Problem: Sort items into appropriate bins
;; Uses the warehouse-robot domain defined above

(define (problem sort-warehouse-items)
  (:domain warehouse-robot)

  ;; Declare specific objects in this problem
  (:objects
    ;; The robot
    kuka - robot

    ;; Items to sort
    red-cube blue-sphere glass-vase - item

    ;; Locations
    conveyor - surface
    bin-a bin-b fragile-bin - container
  )

  ;; Initial state: what is true at the start
  (:init
    ;; Robot starts at conveyor with empty gripper
    (robot-at kuka conveyor)
    (gripper-empty kuka)

    ;; All items on the conveyor
    (at red-cube conveyor)
    (at blue-sphere conveyor)
    (at glass-vase conveyor)

    ;; The glass vase is fragile
    (fragile glass-vase)

    ;; Bins are initially empty and accessible
    (clear bin-a)
    (clear bin-b)
    (clear fragile-bin)
    (accessible bin-a)
    (accessible bin-b)
    (accessible fragile-bin)
    (accessible conveyor)
  )

  ;; Goal: items sorted into appropriate bins
  (:goal (and
    (at red-cube bin-a)
    (at blue-sphere bin-b)
    (at glass-vase fragile-bin)
  ))
)
```

### PDDL Extensions

PDDL has evolved with various extensions for more expressive power:

| Extension | Feature | Example Use |
|-----------|---------|-------------|
| **:strips** | Basic add/delete effects | Core manipulation |
| **:typing** | Type hierarchy | Distinguish robots from objects |
| **:negative-preconditions** | Negation in preconditions | "not holding anything" |
| **:disjunctive-preconditions** | OR in preconditions | "at A or at B" |
| **:equality** | Object equality tests | "?x != ?y" |
| **:numeric-fluents** | Numeric state variables | Battery level, weight |
| **:durative-actions** | Actions with duration | Timed operations |
| **:conditional-effects** | Conditional outcomes | Different effects by context |

## Planning Algorithms

### Forward State-Space Search

The most intuitive approach: start from initial state, apply actions, search for goal.

```
Forward Search
=============

    Initial State
         │
         ▼
    ┌─────────────────────────────────────────────┐
    │            SEARCH TREE                       │
    │                                              │
    │              S₀                              │
    │           ╱  │  ╲                            │
    │         a₁  a₂  a₃   (applicable actions)   │
    │        ╱    │    ╲                           │
    │      S₁    S₂    S₃                          │
    │     ╱│╲   ╱│╲   ╱│╲                          │
    │    ...  ...  ...                             │
    │                    ╲                         │
    │                     Sgoal ← FOUND!           │
    └─────────────────────────────────────────────┘

    Search strategies:
    - BFS: Finds shortest plan (optimal)
    - DFS: Memory efficient, may find long plans
    - A*: Uses heuristic, optimal with admissible h
    - Greedy: Fast but not optimal
```

```python
from typing import List, Set, Optional, Dict
from dataclasses import dataclass
from collections import deque

@dataclass(frozen=True)
class State:
    """Immutable state representation as a frozenset of predicates."""
    predicates: frozenset

    def satisfies(self, goal: Set[str]) -> bool:
        """Check if state satisfies goal predicates."""
        return goal.issubset(self.predicates)

@dataclass
class Action:
    """Planning action with preconditions and effects."""
    name: str
    parameters: tuple
    preconditions: Set[str]      # Required predicates
    add_effects: Set[str]        # Predicates to add
    delete_effects: Set[str]     # Predicates to remove

    def applicable(self, state: State) -> bool:
        """Check if action can be applied in state."""
        return self.preconditions.issubset(state.predicates)

    def apply(self, state: State) -> State:
        """Apply action to state, returning new state."""
        new_predicates = (state.predicates - self.delete_effects) | self.add_effects
        return State(frozenset(new_predicates))

    def __repr__(self):
        return f"{self.name}({', '.join(self.parameters)})"

def forward_search_bfs(initial_state: State,
                       goal: Set[str],
                       get_applicable_actions) -> Optional[List[Action]]:
    """
    Breadth-first forward search for planning.

    Returns shortest plan (optimal in terms of action count).
    """
    if initial_state.satisfies(goal):
        return []

    # Queue of (state, plan) pairs
    queue = deque([(initial_state, [])])
    visited = {initial_state}

    while queue:
        state, plan = queue.popleft()

        # Try all applicable actions
        for action in get_applicable_actions(state):
            new_state = action.apply(state)

            if new_state in visited:
                continue
            visited.add(new_state)

            new_plan = plan + [action]

            if new_state.satisfies(goal):
                return new_plan

            queue.append((new_state, new_plan))

    return None  # No plan found
```

### Backward Search (Regression)

Start from goal, regress through actions, search for initial state.

```
Backward Search (Regression)
===========================

    Goal State
         ▲
         │
    ┌─────────────────────────────────────────────┐
    │            SEARCH TREE                       │
    │                                              │
    │              G                               │
    │           ╱  │  ╲                            │
    │         a₁⁻¹ a₂⁻¹ a₃⁻¹  (actions achieving G)│
    │        ╱    │    ╲                           │
    │      G₁    G₂    G₃    (subgoals)            │
    │     ╱│╲   ╱│╲   ╱│╲                          │
    │    ...  ...  ...                             │
    │   ╱                                          │
    │  S₀ ← Initial state reached!                 │
    └─────────────────────────────────────────────┘

    Regression: Given goal G and action a,
    what must be true BEFORE a to achieve G?

    Advantage: Often more focused than forward search
    Disadvantage: Regression can be complex
```

```python
def regress_goal(goal: Set[str], action: Action) -> Optional[Set[str]]:
    """
    Regress goal through action.

    Returns the subgoal that must be true before action
    to achieve goal after action.
    """
    # Action must contribute to goal (add something we need)
    if not action.add_effects.intersection(goal):
        return None  # Action doesn't help

    # Action must not delete anything in goal
    if action.delete_effects.intersection(goal):
        return None  # Action destroys goal

    # Subgoal = (goal - effects of action) + preconditions
    subgoal = (goal - action.add_effects) | action.preconditions
    return subgoal
```

### Heuristic Search

Good heuristics dramatically improve planning efficiency. The key is estimating how "far" a state is from the goal.

```
Planning Heuristics
==================

    Goal: {on(A,B), on(B,C), on(C,table)}

    Heuristic Ideas:

    1. GOAL COUNT: Count unsatisfied goals
       h(s) = |goals not satisfied in s|
       Simple but weak guidance

    2. DELETE RELAXATION: Ignore delete effects
       Solve "relaxed" problem (only additions)
       Much easier to solve, provides admissible h

       ┌─────────────────────────────────────┐
       │ Real Problem:                       │
       │   pick(A) adds holding(A)           │
       │           deletes on(A,table)       │
       │                                     │
       │ Relaxed Problem:                    │
       │   pick(A) adds holding(A)           │
       │           (deletes nothing)         │
       │                                     │
       │ Relaxed solution length ≤ real      │
       │ → Admissible heuristic!             │
       └─────────────────────────────────────┘

    3. LANDMARK HEURISTICS:
       Identify facts that MUST be achieved
       Count unachieved landmarks
```

```python
import heapq

def astar_planning(initial_state: State,
                   goal: Set[str],
                   get_applicable_actions,
                   heuristic) -> Optional[List[Action]]:
    """
    A* search for planning with heuristic guidance.
    """
    if initial_state.satisfies(goal):
        return []

    # Priority queue: (f_score, state_id, state, plan)
    counter = 0
    open_set = [(heuristic(initial_state, goal), counter, initial_state, [])]
    g_scores = {initial_state: 0}

    while open_set:
        _, _, state, plan = heapq.heappop(open_set)

        if state.satisfies(goal):
            return plan

        for action in get_applicable_actions(state):
            new_state = action.apply(state)
            new_g = g_scores[state] + 1  # Unit action cost

            if new_g < g_scores.get(new_state, float('inf')):
                g_scores[new_state] = new_g
                f_score = new_g + heuristic(new_state, goal)
                counter += 1
                new_plan = plan + [action]
                heapq.heappush(open_set, (f_score, counter, new_state, new_plan))

    return None

def goal_count_heuristic(state: State, goal: Set[str]) -> int:
    """Simple heuristic: count unsatisfied goals."""
    return len(goal - state.predicates)
```

## Hierarchical Task Networks (HTN)

### The Power of Abstraction

HTN planning uses domain knowledge about how to decompose tasks, often leading to much faster planning than classical approaches.

```
HTN Decomposition
================

    HIGH-LEVEL TASK: "Make coffee"
                │
                ▼
    ┌─────────────────────────────────────────┐
    │           DECOMPOSITION                  │
    │                                          │
    │  Method: make-coffee-method              │
    │  Precondition: have(coffee-maker)        │
    │  Subtasks:                               │
    │    1. prepare-machine                    │
    │    2. add-ingredients                    │
    │    3. brew                               │
    │    4. pour-coffee                        │
    └─────────────────────────────────────────┘
                │
                ▼
    ┌─────────────────────────────────────────┐
    │      FURTHER DECOMPOSITION               │
    │                                          │
    │  "add-ingredients" decomposes to:        │
    │    1. fill-water                         │
    │    2. add-filter                         │
    │    3. add-grounds                        │
    └─────────────────────────────────────────┘
                │
                ▼
    ┌─────────────────────────────────────────┐
    │      PRIMITIVE ACTIONS                   │
    │                                          │
    │  Eventually reach executable actions:    │
    │  [open-lid, pour-water, close-lid, ...]  │
    └─────────────────────────────────────────┘
```

**HTN vs Classical Planning:**

| Aspect | Classical Planning | HTN Planning |
|--------|-------------------|--------------|
| Knowledge | Actions only | Actions + methods |
| Search | State space | Task decomposition |
| Guidance | Heuristics | Domain methods |
| Plans | Any valid sequence | Follows decomposition |
| Efficiency | Can be slow | Often very fast |
| Flexibility | Very general | Domain-specific |

### HTN Implementation

```python
from dataclasses import dataclass
from typing import List, Optional, Callable, Dict
from abc import ABC, abstractmethod

@dataclass
class Task(ABC):
    """Base class for tasks in HTN planning."""
    name: str

@dataclass
class PrimitiveTask(Task):
    """A directly executable action."""
    preconditions: Set[str]
    add_effects: Set[str]
    delete_effects: Set[str]

    def applicable(self, state: State) -> bool:
        return self.preconditions.issubset(state.predicates)

    def apply(self, state: State) -> State:
        new_preds = (state.predicates - self.delete_effects) | self.add_effects
        return State(frozenset(new_preds))

@dataclass
class CompoundTask(Task):
    """A task that must be decomposed into subtasks."""
    pass

@dataclass
class Method:
    """A way to decompose a compound task into subtasks."""
    name: str
    task: str                    # Name of task this method decomposes
    preconditions: Set[str]      # When this method applies
    subtasks: List[Task]         # Ordered list of subtasks

    def applicable(self, state: State) -> bool:
        return self.preconditions.issubset(state.predicates)

class HTNPlanner:
    """
    Hierarchical Task Network planner.

    Decomposes compound tasks into primitive actions
    using domain-specific methods.
    """

    def __init__(self):
        self.methods: Dict[str, List[Method]] = {}  # task_name -> methods
        self.primitives: Dict[str, PrimitiveTask] = {}

    def add_method(self, method: Method):
        """Register a decomposition method."""
        if method.task not in self.methods:
            self.methods[method.task] = []
        self.methods[method.task].append(method)

    def add_primitive(self, task: PrimitiveTask):
        """Register a primitive action."""
        self.primitives[task.name] = task

    def plan(self, tasks: List[Task], state: State) -> Optional[List[PrimitiveTask]]:
        """
        Find a plan to accomplish all tasks from given state.

        Returns list of primitive actions, or None if no plan found.
        """
        if not tasks:
            return []  # All tasks done

        task = tasks[0]
        remaining = tasks[1:]

        # Case 1: Primitive task
        if task.name in self.primitives:
            primitive = self.primitives[task.name]
            if not primitive.applicable(state):
                return None  # Can't execute

            new_state = primitive.apply(state)
            rest_plan = self.plan(remaining, new_state)

            if rest_plan is not None:
                return [primitive] + rest_plan
            return None

        # Case 2: Compound task - try each applicable method
        if task.name in self.methods:
            for method in self.methods[task.name]:
                if not method.applicable(state):
                    continue

                # Replace task with method's subtasks
                new_tasks = method.subtasks + remaining
                plan = self.plan(new_tasks, state)

                if plan is not None:
                    return plan

        return None  # No method worked

# Example: Coffee-making domain
def create_coffee_domain() -> HTNPlanner:
    """Create an HTN domain for making coffee."""
    planner = HTNPlanner()

    # Primitive actions
    planner.add_primitive(PrimitiveTask(
        name="fill-water",
        preconditions={"reservoir-empty", "water-available"},
        add_effects={"reservoir-full"},
        delete_effects={"reservoir-empty"}
    ))

    planner.add_primitive(PrimitiveTask(
        name="add-grounds",
        preconditions={"filter-in-place", "have-grounds"},
        add_effects={"grounds-added"},
        delete_effects=set()
    ))

    planner.add_primitive(PrimitiveTask(
        name="insert-filter",
        preconditions={"have-filter"},
        add_effects={"filter-in-place"},
        delete_effects=set()
    ))

    planner.add_primitive(PrimitiveTask(
        name="press-brew",
        preconditions={"reservoir-full", "grounds-added"},
        add_effects={"coffee-ready"},
        delete_effects=set()
    ))

    planner.add_primitive(PrimitiveTask(
        name="pour-coffee",
        preconditions={"coffee-ready", "cup-placed"},
        add_effects={"coffee-in-cup"},
        delete_effects={"coffee-ready"}
    ))

    # Compound task decomposition methods
    planner.add_method(Method(
        name="standard-brew",
        task="make-coffee",
        preconditions={"have-grounds", "water-available"},
        subtasks=[
            CompoundTask("prepare-machine"),
            CompoundTask("brew"),
            CompoundTask("serve")
        ]
    ))

    planner.add_method(Method(
        name="prepare-method",
        task="prepare-machine",
        preconditions=set(),
        subtasks=[
            PrimitiveTask("insert-filter", set(), set(), set()),
            PrimitiveTask("fill-water", set(), set(), set()),
            PrimitiveTask("add-grounds", set(), set(), set())
        ]
    ))

    return planner
```

## Task and Motion Planning (TAMP)

### The Integration Challenge

Real robots need both symbolic task planning and geometric motion planning. TAMP integrates these layers, handling the fact that not all symbolically valid plans are geometrically feasible.

```
TAMP Architecture
================

    Task Level                    Motion Level
    ┌─────────────┐              ┌─────────────┐
    │   PDDL      │              │   MoveIt    │
    │   Planner   │              │   RRT/RRT*  │
    └──────┬──────┘              └──────┬──────┘
           │                            │
           │  Symbolic Plan             │  Trajectories
           │                            │
           ▼                            ▼
    ┌─────────────────────────────────────────────┐
    │              TAMP INTEGRATION                │
    │                                              │
    │  For each symbolic action:                   │
    │    1. Ground action to specific poses        │
    │    2. Check geometric feasibility            │
    │    3. Plan motion trajectory                 │
    │    4. If infeasible → backtrack or replan    │
    │                                              │
    └─────────────────────────────────────────────┘
                        │
                        ▼
              Executable Robot Plan
              [trajectory₁, trajectory₂, ...]
```

### Geometric Grounding

Symbolic actions must be "grounded" to specific geometric configurations.

```
Geometric Grounding
==================

    Symbolic Action: pick(cup, table)

    Geometric Questions:
    ┌─────────────────────────────────────────────┐
    │                                             │
    │  WHERE exactly on the table?                │
    │    → Sample positions on table surface      │
    │                                             │
    │  HOW to grasp the cup?                      │
    │    → Sample grasp poses around cup rim      │
    │                                             │
    │  FROM WHAT configuration?                   │
    │    → Robot arm joint values for grasp       │
    │                                             │
    │  Is the grasp REACHABLE?                    │
    │    → Inverse kinematics must have solution  │
    │                                             │
    │  Is the motion COLLISION-FREE?              │
    │    → Motion planner must find valid path    │
    │                                             │
    └─────────────────────────────────────────────┘
```

```python
from typing import Tuple, Optional
import numpy as np

@dataclass
class GeometricState:
    """Full geometric state including object poses and robot config."""
    robot_config: np.ndarray       # Joint angles
    object_poses: Dict[str, np.ndarray]  # Object name -> SE(3) pose

@dataclass
class GroundedAction:
    """A symbolic action with geometric parameters."""
    symbolic_action: Action
    robot_config: np.ndarray       # Target robot configuration
    grasp_pose: Optional[np.ndarray]  # Grasp pose if manipulation
    trajectory: Optional[List[np.ndarray]]  # Motion plan

class TAMPPlanner:
    """
    Task and Motion Planning integration.
    """

    def __init__(self, task_planner, motion_planner, ik_solver):
        self.task_planner = task_planner
        self.motion_planner = motion_planner
        self.ik_solver = ik_solver

    def plan(self, task_goal: Set[str],
             initial_task_state: State,
             initial_geom_state: GeometricState) -> Optional[List[GroundedAction]]:
        """
        Generate a combined task and motion plan.
        """
        # Step 1: Get symbolic task plan
        task_plan = self.task_planner.plan(initial_task_state, task_goal)
        if task_plan is None:
            return None

        # Step 2: Ground each action geometrically
        grounded_plan = []
        current_geom = initial_geom_state

        for action in task_plan:
            grounded = self.ground_action(action, current_geom)

            if grounded is None:
                # Geometric grounding failed - need to replan
                return self.replan_with_geometric_constraints(
                    action, task_plan, current_geom
                )

            grounded_plan.append(grounded)
            current_geom = self.apply_geometric_effects(grounded, current_geom)

        return grounded_plan

    def ground_action(self, action: Action,
                      geom_state: GeometricState) -> Optional[GroundedAction]:
        """
        Find geometric parameters for a symbolic action.

        Samples grasp poses, computes IK, plans motion.
        """
        if action.name == "pick":
            return self.ground_pick(action, geom_state)
        elif action.name == "place":
            return self.ground_place(action, geom_state)
        elif action.name == "move":
            return self.ground_move(action, geom_state)
        return None

    def ground_pick(self, action: Action,
                    geom_state: GeometricState) -> Optional[GroundedAction]:
        """Ground a pick action with grasp sampling."""
        object_name = action.parameters[0]
        object_pose = geom_state.object_poses[object_name]

        # Sample multiple grasp poses
        grasp_poses = self.sample_grasps(object_pose)

        for grasp_pose in grasp_poses:
            # Check IK feasibility
            ik_solution = self.ik_solver.solve(grasp_pose)
            if ik_solution is None:
                continue

            # Plan motion to grasp configuration
            trajectory = self.motion_planner.plan(
                geom_state.robot_config,
                ik_solution
            )
            if trajectory is None:
                continue

            # Found valid grounding
            return GroundedAction(
                symbolic_action=action,
                robot_config=ik_solution,
                grasp_pose=grasp_pose,
                trajectory=trajectory
            )

        return None  # No valid grounding found

    def sample_grasps(self, object_pose: np.ndarray,
                      num_samples: int = 10) -> List[np.ndarray]:
        """Sample grasp poses around an object."""
        grasps = []
        for _ in range(num_samples):
            # Sample approach direction
            angle = np.random.uniform(0, 2 * np.pi)
            offset = np.array([
                0.1 * np.cos(angle),
                0.1 * np.sin(angle),
                0.05  # Approach from above
            ])

            grasp_pose = object_pose.copy()
            grasp_pose[:3, 3] += offset
            grasps.append(grasp_pose)

        return grasps
```

## Modern Approaches: LLMs for Task Planning

### Language Models as Planners

Large Language Models can generate task plans from natural language, offering flexibility and generalization.

```
LLM-Based Planning
=================

    User Request: "Clean up the living room"
           │
           ▼
    ┌─────────────────────────────────────────────┐
    │              LLM PLANNER                     │
    │                                              │
    │  Prompt: "You are a robot assistant.         │
    │   Generate a step-by-step plan to            │
    │   clean up the living room.                  │
    │   Available actions: pick, place, move,      │
    │   open, close, wipe..."                      │
    │                                              │
    │  LLM Output:                                 │
    │    1. Survey room for items out of place     │
    │    2. Pick up books from floor               │
    │    3. Place books on bookshelf               │
    │    4. Pick up cushions                       │
    │    5. Place cushions on couch                │
    │    6. Move to kitchen                        │
    │    7. Get cleaning cloth                     │
    │    8. Return to living room                  │
    │    9. Wipe coffee table                      │
    │   10. ...                                    │
    └─────────────────────────────────────────────┘
           │
           ▼
    Validate & Execute with Robot
```

**LLM Planning Approaches:**

| Approach | Description | Advantage | Challenge |
|----------|-------------|-----------|-----------|
| **Direct prompting** | LLM outputs action sequence | Simple | May violate constraints |
| **Code-as-policy** | LLM writes executable code | Precise, composable | Requires API design |
| **LLM + verifier** | LLM proposes, verifier checks | More reliable | Slower |
| **LLM + PDDL** | LLM generates PDDL problems | Formal verification | Translation required |

## Industry Perspectives: Task Planning in Practice

### Warehouse Robotics

Kiva/Amazon robotics systems coordinate thousands of robots:

| Challenge | Planning Solution |
|-----------|-------------------|
| Multi-robot coordination | Centralized task assignment + distributed motion |
| Dynamic orders | Continuous replanning |
| Deadlock avoidance | Resource reservation systems |
| Throughput optimization | Task sequencing heuristics |

### Manufacturing Assembly

Automotive and electronics assembly lines:

```
Assembly Task Planning
=====================

    Product: Smartphone

    Bill of Materials → Task Graph:

    ┌─────────┐     ┌─────────┐     ┌─────────┐
    │ Insert  │────▶│ Connect │────▶│ Attach  │
    │ battery │     │ flex    │     │ screen  │
    └─────────┘     │ cable   │     └─────────┘
                    └─────────┘
         │
         │      ┌─────────┐
         └─────▶│ Close   │
                │ case    │
                └─────────┘

    Constraints:
    - Precedence (battery before closing)
    - Resource (one robot per station)
    - Timing (cure times, test durations)
```

### Service Robotics

Home and hospitality robots face open-world planning:

| Challenge | Approach |
|-----------|----------|
| Partial observability | Contingent planning, replanning |
| Human preferences | Learned reward functions |
| Novel situations | LLM-based reasoning |
| Safety constraints | Temporal logic specifications |

## Summary: The Task Planning Landscape

Task planning enables robots to reason about what to do, not just how to do it. The key concepts we've covered:

**Key Takeaways:**

1. **PDDL is the standard language**: Separating domain from problem enables reusable planning knowledge.

2. **Search is the core mechanism**: Forward, backward, or heuristic-guided search explores the state space.

3. **HTN encodes expertise**: Domain-specific methods dramatically speed up planning.

4. **TAMP bridges symbolic and geometric**: Real robot plans need both task reasoning and motion planning.

5. **LLMs offer new possibilities**: Natural language interfaces and flexible reasoning complement formal methods.

6. **Integration is key**: Production systems combine multiple approaches for robustness.

```
Choosing a Planning Approach
===========================

    Task Characteristics
           │
    ┌──────┴──────┐
    │             │
   Known       Novel
   Domain      Situations
    │             │
    ▼             ▼
   HTN          LLM +
   PDDL       Classical
    │             │
    └──────┬──────┘
           │
    Need Geometry?
           │
    ┌──────┴──────┐
   Yes           No
    │             │
    ▼             ▼
   TAMP      Pure Task
             Planning
```

---

## Further Reading

**Foundational Texts:**
- Ghallab, Nau & Traverso, "Automated Planning: Theory and Practice" - The comprehensive reference
- Russell & Norvig, "Artificial Intelligence: A Modern Approach" - Planning chapters

**Key Papers:**
- Fikes & Nilsson, "STRIPS: A New Approach to AI Planning" (1971)
- Erol, Hendler & Nau, "HTN Planning: Complexity and Expressivity" (1994)
- Garrett et al., "Integrated Task and Motion Planning" (2021)

**Software Resources:**
- [Fast Downward Planner](https://www.fast-downward.org/)
- [PDDL Editor and Resources](https://planning.wiki/)
- [PDDLGym for RL](https://github.com/tomsilver/pddlgym)

**Modern Approaches:**
- Huang et al., "Language Models as Zero-Shot Planners" (2022)
- Ahn et al., "SayCan: Grounding Language in Robotic Affordances" (2022)
