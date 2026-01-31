---
id: ch-3-09
title: Behavior Trees
sidebar_position: 3
difficulty: intermediate
estimated_time: 80
prerequisites: [ch-3-08]
---

# Behavior Trees: Composable Robot Intelligence

> *"Behavior trees emerged from video games, where characters needed to seem intelligent without actually being so. For robots, they provide the same gift: complex behavior from simple, composable pieces."*

When a game character patrols an area, investigates sounds, and attacks enemies, its behavior isn't hard-coded as a giant state machine. Instead, it's organized as a behavior tree—a modular, hierarchical structure that composes simple behaviors into complex ones. This architecture, battle-tested in millions of video game characters, has become the standard for organizing robot behavior in ROS 2 and beyond.

## Why Behavior Trees?

### The Finite State Machine Problem

The natural first approach to robot behavior is finite state machines (FSMs). But FSMs suffer from the "state explosion" problem as complexity grows.

```
FSM Complexity Explosion
=======================

    Simple FSM (3 states):          Complex FSM (adds error handling):

    ┌─────────┐                     ┌─────────┐
    │  IDLE   │──trigger──▶┌───────┐│  IDLE   │──trigger──▶┌───────┐
    └────┬────┘            │WORKING│└────┬────┘            │WORKING│
         │                 └───┬───┘     │  │              └───┬───┘
         │◀────done────────────┘         │  │error             │
         │                               │  ▼                  │
         │                          ┌────┴─────┐          ┌────┴─────┐
         │                          │  ERROR   │◀─error───│  ERROR   │
         │                          └──────────┘          └──────────┘
                                         │  retry              │
                                         ▼                     │
    3 states, 2 transitions         ┌──────────┐               │
                                    │ RECOVERY │───────────────┘
                                    └──────────┘

                                    Now 5+ states, 8+ transitions
                                    And we haven't added:
                                    - Battery low handling
                                    - Human interruption
                                    - Multiple error types
                                    - Priority behaviors...

    FSMs grow as O(n²) in complexity
    Adding one concern affects MANY transitions
```

**FSM vs Behavior Tree:**

| Aspect | Finite State Machine | Behavior Tree |
|--------|---------------------|---------------|
| Structure | Graph (states + transitions) | Tree (nodes + children) |
| Transitions | Explicit, between any states | Implicit, via tree traversal |
| Adding behavior | May require many new transitions | Add subtree, minimal impact |
| Reactivity | Must handle in each state | Built-in via tick mechanism |
| Parallelism | Complex to implement | Natural with parallel nodes |
| Debugging | Hard to trace paths | Clear hierarchical structure |

### The Behavior Tree Solution

Behavior trees organize behavior as a tree where:
- **Leaf nodes** execute actions or check conditions
- **Internal nodes** control flow (sequence, selection, parallel)
- The tree is "ticked" regularly, traversing and executing nodes

```
Behavior Tree Structure
======================

    Root                          The tree is ticked repeatedly
      │                           Each tick traverses the tree
      ▼                           Nodes return: SUCCESS, FAILURE, or RUNNING
    ┌───┐
    │ ? │ ← Selector (fallback)
    └─┬─┘
      │
      ├──────────────┬──────────────┐
      ▼              ▼              ▼
    ┌───┐          ┌───┐          ┌───┐
    │ → │          │ → │          │ A │
    └─┬─┘          └─┬─┘          └───┘
      │              │              │
    ┌─┴─┐          ┌─┴─┐          Fallback
    ▼   ▼          ▼   ▼          action
  ┌───┐┌───┐    ┌───┐┌───┐
  │ C ││ A │    │ C ││ A │
  └───┘└───┘    └───┘└───┘

    Legend:
    ? = Selector (try until success)
    → = Sequence (all must succeed)
    C = Condition check
    A = Action
```

## Core Node Types

### The Tick Mechanism

Every behavior tree node responds to "tick" by returning one of three statuses:

```
Node Status Values
=================

    SUCCESS ✓     The node completed its task successfully
                  Example: "Move to position" reached target

    FAILURE ✗     The node could not complete its task
                  Example: "Check battery" found battery low

    RUNNING ⟳     The node is still working
                  Example: "Navigate to goal" is in progress

    ┌─────────────────────────────────────────────────────┐
    │                    TICK CYCLE                        │
    │                                                      │
    │   Main Loop:                                         │
    │     while robot_running:                             │
    │       status = root.tick()    # Traverse tree        │
    │       if status == RUNNING:                          │
    │         wait(tick_period)     # Typically 10-100ms   │
    │       else:                                          │
    │         handle_completion(status)                    │
    │                                                      │
    └─────────────────────────────────────────────────────┘
```

### Leaf Nodes: Actions and Conditions

Leaf nodes interact with the robot and world.

```python
from enum import Enum
from abc import ABC, abstractmethod
from typing import Callable, Any

class NodeStatus(Enum):
    """Return values for behavior tree nodes."""
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RUNNING = "RUNNING"

class BehaviorNode(ABC):
    """Base class for all behavior tree nodes."""

    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def tick(self) -> NodeStatus:
        """Execute one tick of this node."""
        pass

    def reset(self) -> None:
        """Reset node state for fresh execution."""
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"

class ActionNode(BehaviorNode):
    """
    Executes an action in the world.

    Returns SUCCESS when action completes.
    Returns FAILURE if action cannot be performed.
    Returns RUNNING if action is in progress.
    """

    def __init__(self, name: str, action_func: Callable[[], NodeStatus]):
        super().__init__(name)
        self.action = action_func

    def tick(self) -> NodeStatus:
        return self.action()

class ConditionNode(BehaviorNode):
    """
    Checks a condition without side effects.

    Returns SUCCESS if condition is true.
    Returns FAILURE if condition is false.
    Never returns RUNNING (conditions are instantaneous).
    """

    def __init__(self, name: str, condition_func: Callable[[], bool]):
        super().__init__(name)
        self.condition = condition_func

    def tick(self) -> NodeStatus:
        if self.condition():
            return NodeStatus.SUCCESS
        return NodeStatus.FAILURE
```

### Sequence Node: Do All In Order

The sequence node executes children left-to-right, succeeding only if all children succeed.

```
Sequence Node Semantics
======================

    Symbol: →  or  ──▶

    ┌───────────────────────────────────────────────────────┐
    │                                                       │
    │    ┌─────┐                                            │
    │    │  →  │                                            │
    │    └──┬──┘                                            │
    │       │                                               │
    │  ┌────┼────┬────┐                                     │
    │  ▼    ▼    ▼    ▼                                     │
    │ [A]  [B]  [C]  [D]                                    │
    │                                                       │
    │  Execution order: A, B, C, D                          │
    │                                                       │
    │  If any child FAILS → Sequence FAILS                  │
    │  If any child RUNNING → Sequence RUNNING              │
    │  If all children SUCCEED → Sequence SUCCEEDS          │
    │                                                       │
    └───────────────────────────────────────────────────────┘

    Analogy: AND gate for success
             "Do A AND B AND C AND D"
```

```python
class SequenceNode(BehaviorNode):
    """
    Execute children in order until one fails.

    Like a logical AND: all children must succeed.
    Remembers which child is running between ticks.
    """

    def __init__(self, name: str, children: list):
        super().__init__(name)
        self.children = children
        self.current_index = 0

    def tick(self) -> NodeStatus:
        while self.current_index < len(self.children):
            child = self.children[self.current_index]
            status = child.tick()

            if status == NodeStatus.RUNNING:
                # Child still working, return and continue next tick
                return NodeStatus.RUNNING
            elif status == NodeStatus.FAILURE:
                # One child failed, sequence fails
                self.reset()
                return NodeStatus.FAILURE

            # Child succeeded, move to next
            self.current_index += 1

        # All children succeeded
        self.reset()
        return NodeStatus.SUCCESS

    def reset(self):
        self.current_index = 0
        for child in self.children:
            child.reset()
```

### Selector Node: Try Until Success

The selector (fallback) node tries children until one succeeds.

```
Selector Node Semantics
======================

    Symbol: ?  or  ──?

    ┌───────────────────────────────────────────────────────┐
    │                                                       │
    │    ┌─────┐                                            │
    │    │  ?  │                                            │
    │    └──┬──┘                                            │
    │       │                                               │
    │  ┌────┼────┬────┐                                     │
    │  ▼    ▼    ▼    ▼                                     │
    │ [A]  [B]  [C]  [D]                                    │
    │                                                       │
    │  Execution: Try A, if fails try B, if fails try C... │
    │                                                       │
    │  If any child SUCCEEDS → Selector SUCCEEDS            │
    │  If any child RUNNING → Selector RUNNING              │
    │  If all children FAIL → Selector FAILS                │
    │                                                       │
    └───────────────────────────────────────────────────────┘

    Analogy: OR gate for success
             "Do A OR B OR C OR D"
             Fallback / priority list
```

```python
class SelectorNode(BehaviorNode):
    """
    Execute children until one succeeds.

    Like a logical OR: any child success means selector success.
    Often called "Fallback" node.
    Useful for trying alternatives or priority behaviors.
    """

    def __init__(self, name: str, children: list):
        super().__init__(name)
        self.children = children
        self.current_index = 0

    def tick(self) -> NodeStatus:
        while self.current_index < len(self.children):
            child = self.children[self.current_index]
            status = child.tick()

            if status == NodeStatus.RUNNING:
                return NodeStatus.RUNNING
            elif status == NodeStatus.SUCCESS:
                # Found a successful child
                self.reset()
                return NodeStatus.SUCCESS

            # Child failed, try next
            self.current_index += 1

        # All children failed
        self.reset()
        return NodeStatus.FAILURE

    def reset(self):
        self.current_index = 0
        for child in self.children:
            child.reset()
```

### Parallel Node: Do Multiple Things

The parallel node executes all children simultaneously.

```
Parallel Node Semantics
======================

    Symbol: ⇉  or  ═══

    ┌───────────────────────────────────────────────────────┐
    │                                                       │
    │    ┌─────┐                                            │
    │    │  ⇉  │  Policy: SUCCESS_ALL / SUCCESS_ONE        │
    │    └──┬──┘                                            │
    │       │                                               │
    │  ┌────┼────┬────┐                                     │
    │  ▼    ▼    ▼    ▼                                     │
    │ [A]  [B]  [C]  [D]  ← All ticked each cycle          │
    │                                                       │
    │  SUCCESS_ALL:                                         │
    │    All must succeed → Parallel succeeds               │
    │    Any fails → Parallel fails                         │
    │                                                       │
    │  SUCCESS_ONE:                                         │
    │    Any succeeds → Parallel succeeds                   │
    │    All fail → Parallel fails                          │
    │                                                       │
    └───────────────────────────────────────────────────────┘

    Use cases:
    - Move while monitoring sensors
    - Multiple simultaneous checks
    - Timeout with action
```

```python
class ParallelPolicy(Enum):
    """Policy for parallel node success/failure."""
    SUCCESS_ALL = "SUCCESS_ALL"    # All children must succeed
    SUCCESS_ONE = "SUCCESS_ONE"    # One child success is enough

class ParallelNode(BehaviorNode):
    """
    Execute all children simultaneously.

    All children are ticked each cycle.
    Policy determines success/failure criteria.
    """

    def __init__(self, name: str, children: list,
                 policy: ParallelPolicy = ParallelPolicy.SUCCESS_ALL):
        super().__init__(name)
        self.children = children
        self.policy = policy

    def tick(self) -> NodeStatus:
        success_count = 0
        failure_count = 0
        running_count = 0

        # Tick all children
        for child in self.children:
            status = child.tick()

            if status == NodeStatus.SUCCESS:
                success_count += 1
            elif status == NodeStatus.FAILURE:
                failure_count += 1
            else:
                running_count += 1

        # Apply policy
        if self.policy == ParallelPolicy.SUCCESS_ALL:
            if failure_count > 0:
                return NodeStatus.FAILURE
            if success_count == len(self.children):
                return NodeStatus.SUCCESS
            return NodeStatus.RUNNING

        else:  # SUCCESS_ONE
            if success_count > 0:
                return NodeStatus.SUCCESS
            if failure_count == len(self.children):
                return NodeStatus.FAILURE
            return NodeStatus.RUNNING
```

## Decorator Nodes

Decorators modify the behavior of a single child node.

```
Common Decorators
================

    INVERTER (!)           REPEAT (×n)           RETRY (↻)
    ┌───────┐              ┌───────┐             ┌───────┐
    │   !   │              │  ×3   │             │  ↻3   │
    └───┬───┘              └───┬───┘             └───┬───┘
        │                      │                     │
        ▼                      ▼                     ▼
      [Child]                [Child]              [Child]

    Inverts result         Repeat 3 times        Retry up to 3
    SUCCESS→FAILURE        on success            times on failure


    TIMEOUT (⏱)            FORCE_SUCCESS (✓)    FORCE_FAILURE (✗)
    ┌───────┐              ┌───────┐             ┌───────┐
    │  ⏱5s  │              │   ✓   │             │   ✗   │
    └───┬───┘              └───┬───┘             └───┬───┘
        │                      │                     │
        ▼                      ▼                     ▼
      [Child]                [Child]              [Child]

    Fail after 5s          Always succeed        Always fail
    if still running       (ignore child)        (ignore child)
```

```python
class InverterNode(BehaviorNode):
    """
    Inverts child's result.

    SUCCESS becomes FAILURE, FAILURE becomes SUCCESS.
    RUNNING stays RUNNING.
    """

    def __init__(self, name: str, child: BehaviorNode):
        super().__init__(name)
        self.child = child

    def tick(self) -> NodeStatus:
        status = self.child.tick()

        if status == NodeStatus.SUCCESS:
            return NodeStatus.FAILURE
        elif status == NodeStatus.FAILURE:
            return NodeStatus.SUCCESS
        return NodeStatus.RUNNING

class RepeatNode(BehaviorNode):
    """
    Repeat child execution N times.

    Succeeds after N successful executions.
    Fails immediately if child fails.
    """

    def __init__(self, name: str, child: BehaviorNode, times: int):
        super().__init__(name)
        self.child = child
        self.times = times
        self.count = 0

    def tick(self) -> NodeStatus:
        while self.count < self.times:
            status = self.child.tick()

            if status == NodeStatus.RUNNING:
                return NodeStatus.RUNNING
            elif status == NodeStatus.FAILURE:
                self.reset()
                return NodeStatus.FAILURE

            # Child succeeded, increment and continue
            self.count += 1
            self.child.reset()

        self.reset()
        return NodeStatus.SUCCESS

    def reset(self):
        self.count = 0
        self.child.reset()

class RetryNode(BehaviorNode):
    """
    Retry child up to N times on failure.

    Succeeds if child ever succeeds.
    Fails only after N failures.
    """

    def __init__(self, name: str, child: BehaviorNode, max_retries: int):
        super().__init__(name)
        self.child = child
        self.max_retries = max_retries
        self.retry_count = 0

    def tick(self) -> NodeStatus:
        while self.retry_count <= self.max_retries:
            status = self.child.tick()

            if status == NodeStatus.SUCCESS:
                self.reset()
                return NodeStatus.SUCCESS
            elif status == NodeStatus.RUNNING:
                return NodeStatus.RUNNING

            # Child failed, retry
            self.retry_count += 1
            self.child.reset()

        self.reset()
        return NodeStatus.FAILURE

    def reset(self):
        self.retry_count = 0
        self.child.reset()

import time

class TimeoutNode(BehaviorNode):
    """
    Fail if child runs longer than timeout.
    """

    def __init__(self, name: str, child: BehaviorNode, timeout_sec: float):
        super().__init__(name)
        self.child = child
        self.timeout_sec = timeout_sec
        self.start_time = None

    def tick(self) -> NodeStatus:
        if self.start_time is None:
            self.start_time = time.time()

        elapsed = time.time() - self.start_time

        if elapsed >= self.timeout_sec:
            self.reset()
            return NodeStatus.FAILURE

        status = self.child.tick()

        if status != NodeStatus.RUNNING:
            self.reset()

        return status

    def reset(self):
        self.start_time = None
        self.child.reset()
```

## Building Complex Behaviors

### Example: Pick and Place Robot

```
Pick and Place Behavior Tree
===========================

    ┌─────────────────────────────────────────────────────────────┐
    │                         ROOT                                 │
    │                          │                                   │
    │                     ┌────?────┐  ← Selector: try strategies  │
    │                     │         │                              │
    │            ┌────────┴────┐    └───────────────┐              │
    │            ▼                                  ▼              │
    │       ┌────→────┐                       ┌────→────┐          │
    │       │ Main    │                       │ Search  │          │
    │       │ Task    │                       │ & Plan  │          │
    │       └────┬────┘                       └────┬────┘          │
    │            │                                 │               │
    │   ┌────┬───┴───┬────┬────┐          ┌───────┴───────┐       │
    │   ▼    ▼       ▼    ▼    ▼          ▼               ▼       │
    │  [Has] [Move] [Pick][Move][Place]  [Scan]        [Select]   │
    │  [Tgt] [To   ] [Up ] [To ] [Down]  [Area]        [Target]   │
    │       [Obj  ]      [Goal]                                    │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘

    Behavior:
    1. If has target: move→pick→move→place
    2. If no target: scan area→select new target
```

```python
class RobotInterface:
    """Interface to robot hardware/simulation."""

    def has_target(self) -> bool:
        """Check if a pick target is selected."""
        pass

    def get_target_pose(self):
        """Get pose of current target object."""
        pass

    def get_current_pose(self):
        """Get current end-effector pose."""
        pass

    def move_to(self, pose) -> NodeStatus:
        """Move arm to pose. Returns RUNNING until complete."""
        pass

    def pick(self) -> NodeStatus:
        """Execute pick. Returns RUNNING until complete."""
        pass

    def place(self) -> NodeStatus:
        """Execute place. Returns RUNNING until complete."""
        pass

    def scan_for_objects(self) -> NodeStatus:
        """Scan area for objects."""
        pass

    def select_best_target(self) -> NodeStatus:
        """Select best object as pick target."""
        pass

def create_pick_place_tree(robot: RobotInterface) -> BehaviorNode:
    """
    Create behavior tree for pick and place task.
    """
    # Main task sequence: pick known target and place it
    main_task = SequenceNode("MainTask", [
        ConditionNode("HasTarget", robot.has_target),
        ActionNode("MoveToObject", lambda: robot.move_to(robot.get_target_pose())),
        ActionNode("PickObject", robot.pick),
        ActionNode("MoveToGoal", lambda: robot.move_to(robot.get_goal_pose())),
        ActionNode("PlaceObject", robot.place),
    ])

    # Fallback: find new target
    find_target = SequenceNode("FindTarget", [
        ActionNode("ScanArea", robot.scan_for_objects),
        ActionNode("SelectTarget", robot.select_best_target),
    ])

    # Root selector: try main task, fall back to finding target
    root = SelectorNode("Root", [
        main_task,
        find_target,
    ])

    return root
```

### Example: Navigation with Recovery

```
Navigation with Recovery
=======================

    ┌─────────────────────────────────────────────────────────────┐
    │                         ROOT                                 │
    │                          │                                   │
    │                     ┌────?────┐  ← Try navigation strategies │
    │                     │         │                              │
    │       ┌─────────────┼─────────┼─────────────┐                │
    │       ▼             ▼         ▼             ▼                │
    │   ┌───→───┐    ┌────→────┐ ┌──→───┐    ┌────→────┐          │
    │   │Normal │    │Clear    │ │Backup│    │Ask for  │          │
    │   │Nav    │    │Costmap  │ │      │    │Help     │          │
    │   └───┬───┘    └────┬────┘ └──┬───┘    └────┬────┘          │
    │       │             │         │             │                │
    │   ┌───┴───┐    ┌────┴────┐   │         ┌───┴───┐            │
    │   ▼       ▼    ▼         ▼   │         ▼       ▼            │
    │ [Chk]  [Nav] [Clear] [Wait] [Backup] [Signal][Wait]         │
    │ [Path] [To ] [Map  ] [5s  ] [1m    ] [Human][Help]          │
    │ [OK  ] [Goal]                                                │
    └─────────────────────────────────────────────────────────────┘

    Recovery escalation:
    1. Normal navigation
    2. Clear costmap and retry
    3. Back up and retry
    4. Signal human and wait
```

## BehaviorTree.CPP and ROS 2

### Industry Standard Implementation

BehaviorTree.CPP is the standard library for behavior trees in ROS 2, used by Nav2 and other major packages.

```cpp
// BehaviorTree.CPP Example
#include <behaviortree_cpp_v3/bt_factory.h>
#include <behaviortree_cpp_v3/action_node.h>

// Synchronous action (completes immediately)
class CheckBattery : public BT::SyncActionNode {
public:
    CheckBattery(const std::string& name, const BT::NodeConfiguration& config)
        : BT::SyncActionNode(name, config) {}

    static BT::PortsList providedPorts() {
        return { BT::InputPort<double>("min_battery") };
    }

    BT::NodeStatus tick() override {
        double current_level = get_battery_level();
        double min_level;
        getInput("min_battery", min_level);

        return (current_level >= min_level) ?
            BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
    }
};

// Asynchronous action (runs over multiple ticks)
class NavigateToGoal : public BT::StatefulActionNode {
public:
    NavigateToGoal(const std::string& name, const BT::NodeConfiguration& config)
        : BT::StatefulActionNode(name, config) {}

    static BT::PortsList providedPorts() {
        return {
            BT::InputPort<Pose>("goal"),
            BT::OutputPort<double>("distance_traveled")
        };
    }

    // Called once when node becomes active
    BT::NodeStatus onStart() override {
        Pose goal;
        getInput("goal", goal);
        send_goal_to_navigator(goal);
        return BT::NodeStatus::RUNNING;
    }

    // Called repeatedly while RUNNING
    BT::NodeStatus onRunning() override {
        if (navigation_complete()) {
            setOutput("distance_traveled", get_distance());
            return BT::NodeStatus::SUCCESS;
        }
        if (navigation_failed()) {
            return BT::NodeStatus::FAILURE;
        }
        return BT::NodeStatus::RUNNING;
    }

    // Called when node is halted
    void onHalted() override {
        cancel_navigation();
    }
};

// Register and create tree
int main() {
    BT::BehaviorTreeFactory factory;

    factory.registerNodeType<CheckBattery>("CheckBattery");
    factory.registerNodeType<NavigateToGoal>("NavigateToGoal");

    auto tree = factory.createTreeFromFile("navigation_tree.xml");

    while (true) {
        tree.tickRoot();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}
```

### XML Tree Definition

```xml
<!-- navigation_tree.xml -->
<root main_tree_to_execute="MainTree">
    <BehaviorTree ID="MainTree">
        <Fallback name="root">
            <!-- Try normal navigation -->
            <Sequence name="navigate">
                <CheckBattery min_battery="20.0"/>
                <NavigateToGoal goal="{target_pose}"
                                distance_traveled="{distance}"/>
            </Sequence>

            <!-- Recovery: charge if battery low -->
            <Sequence name="charge_recovery">
                <Inverter>
                    <CheckBattery min_battery="20.0"/>
                </Inverter>
                <NavigateToGoal goal="{charging_station}"/>
                <ChargeBattery target_level="80.0"/>
            </Sequence>

            <!-- Ultimate fallback -->
            <SignalError message="Navigation failed"/>
        </Fallback>
    </BehaviorTree>
</root>
```

## Blackboard: Sharing Data

The blackboard pattern allows nodes to share data without tight coupling.

```
Blackboard Pattern
=================

    ┌─────────────────────────────────────────────────────────────┐
    │                      BLACKBOARD                              │
    │                                                              │
    │   ┌─────────────────────────────────────────────────────┐   │
    │   │  target_pose: (1.0, 2.0, 0.0)                       │   │
    │   │  battery_level: 75.0                                │   │
    │   │  detected_objects: [obj1, obj2, obj3]               │   │
    │   │  current_goal: "kitchen"                            │   │
    │   │  error_count: 0                                     │   │
    │   └─────────────────────────────────────────────────────┘   │
    │                  ▲           ▲           ▲                   │
    │                  │           │           │                   │
    │   ┌──────────────┼───────────┼───────────┼──────────────┐   │
    │   │              │           │           │              │   │
    │   ▼              ▼           ▼           ▼              ▼   │
    │ ┌───┐          ┌───┐       ┌───┐       ┌───┐          ┌───┐│
    │ │ A │ writes   │ B │ reads │ C │ reads │ D │ writes   │ E ││
    │ │   │ target   │   │target │   │battery│   │ objects  │   ││
    │ └───┘          └───┘       └───┘       └───┘          └───┘│
    │                                                              │
    └─────────────────────────────────────────────────────────────┘

    Benefits:
    - Nodes don't need references to each other
    - Easy to add/remove nodes
    - Centralized state for debugging
```

```python
from typing import Dict, Any, Optional

class Blackboard:
    """
    Shared memory for behavior tree nodes.

    Nodes read and write named values without coupling.
    """

    def __init__(self):
        self._data: Dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        """Read a value from the blackboard."""
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Write a value to the blackboard."""
        self._data[key] = value

    def has(self, key: str) -> bool:
        """Check if key exists."""
        return key in self._data

    def clear(self) -> None:
        """Clear all data."""
        self._data.clear()

class BlackboardNode(BehaviorNode):
    """Base class for nodes that use the blackboard."""

    def __init__(self, name: str, blackboard: Blackboard):
        super().__init__(name)
        self.blackboard = blackboard

class SetBlackboard(BlackboardNode):
    """Set a value on the blackboard."""

    def __init__(self, name: str, blackboard: Blackboard,
                 key: str, value_func: Callable[[], Any]):
        super().__init__(name, blackboard)
        self.key = key
        self.value_func = value_func

    def tick(self) -> NodeStatus:
        self.blackboard.set(self.key, self.value_func())
        return NodeStatus.SUCCESS

class CheckBlackboard(BlackboardNode):
    """Check a condition on the blackboard."""

    def __init__(self, name: str, blackboard: Blackboard,
                 key: str, condition: Callable[[Any], bool]):
        super().__init__(name, blackboard)
        self.key = key
        self.condition = condition

    def tick(self) -> NodeStatus:
        value = self.blackboard.get(self.key)
        if value is not None and self.condition(value):
            return NodeStatus.SUCCESS
        return NodeStatus.FAILURE
```

## Design Patterns

### Reactive Behaviors

Behavior trees naturally support reactive behaviors that respond to changing conditions.

```
Reactive Pattern
===============

    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │   ┌───────────────────────────────────────────────────────┐ │
    │   │                    REACTIVE ROOT                       │ │
    │   │                         │                              │ │
    │   │                    ┌────?────┐                         │ │
    │   │                    │         │                         │ │
    │   │         ┌──────────┴────┐    └──────────────┐          │ │
    │   │         ▼                                   ▼          │ │
    │   │    ┌────→────┐                        ┌────→────┐      │ │
    │   │    │Emergency│  ← Checked first!      │ Normal  │      │ │
    │   │    │ Handler │                        │ Task    │      │ │
    │   │    └────┬────┘                        └────┬────┘      │ │
    │   │         │                                  │           │ │
    │   │   ┌─────┴─────┐                           ...          │ │
    │   │   ▼           ▼                                        │ │
    │   │ [Obstacle]  [Stop]                                     │ │
    │   │ [Detected?] [All ]                                     │ │
    │   │                                                        │ │
    │   └────────────────────────────────────────────────────────┘ │
    │                                                             │
    │   Every tick:                                               │
    │   1. Check emergency conditions first                       │
    │   2. If triggered, interrupt normal task                    │
    │   3. Resume normal task when clear                          │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
```

### Subtree Reuse

Behavior trees support modular design through subtree reuse.

```python
def create_approach_object_subtree(robot: RobotInterface,
                                   object_key: str) -> BehaviorNode:
    """
    Reusable subtree for approaching an object.

    Can be used for approaching pick targets, charging stations, etc.
    """
    return SequenceNode(f"Approach_{object_key}", [
        ConditionNode("ObjectKnown",
                      lambda: robot.blackboard.has(object_key)),
        ActionNode("ComputePath",
                   lambda: robot.compute_path(robot.blackboard.get(object_key))),
        RetryNode("NavigateWithRetry",
                  ActionNode("Navigate", robot.execute_navigation),
                  max_retries=3),
        ActionNode("FinePosition",
                   lambda: robot.fine_approach(robot.blackboard.get(object_key))),
    ])

def create_manipulation_tree(robot: RobotInterface) -> BehaviorNode:
    """
    Main tree reusing approach subtree for different targets.
    """
    return SelectorNode("ManipulationRoot", [
        SequenceNode("PickSequence", [
            create_approach_object_subtree(robot, "pick_target"),
            ActionNode("Pick", robot.pick),
        ]),
        SequenceNode("PlaceSequence", [
            create_approach_object_subtree(robot, "place_location"),
            ActionNode("Place", robot.place),
        ]),
    ])
```

## Industry Perspectives

### Nav2 (ROS 2 Navigation)

The ROS 2 navigation stack uses behavior trees extensively:

| Component | Behavior Tree Role |
|-----------|-------------------|
| Navigation | High-level behavior coordination |
| Recovery | Fallback behaviors when stuck |
| Waypoint following | Sequence of navigation goals |
| Spin, backup | Recovery action nodes |

### Video Game AI

Where behavior trees originated:

| Game | Notable BT Usage |
|------|-----------------|
| Halo 2 | Pioneered BT for game AI |
| The Division | Complex squad behaviors |
| Spore | Creature AI |
| Far Cry | Enemy behaviors |

## Summary: The Behavior Tree Toolbox

Behavior trees provide a powerful, modular way to organize robot behavior. The key concepts:

**Key Takeaways:**

1. **Tick mechanism enables reactivity**: Regular tree traversal allows immediate response to changing conditions.

2. **Three node statuses (SUCCESS, FAILURE, RUNNING)**: Simple but expressive for representing action outcomes.

3. **Sequence and Selector are fundamental**: AND-like and OR-like composition covers most control flow needs.

4. **Decorators modify without restructuring**: Repeat, retry, timeout, and invert behaviors without changing the tree.

5. **Blackboard decouples nodes**: Shared state without explicit node references.

6. **Subtrees enable reuse**: Modular design for maintainable behavior specifications.

```
Behavior Tree Decision Guide
===========================

    Need to...                    Use...
    ────────────────────────────────────────
    Do A then B then C         →  Sequence
    Try A, else B, else C      →  Selector (Fallback)
    Do A and B simultaneously  →  Parallel
    Repeat action N times      →  Repeat decorator
    Retry on failure           →  Retry decorator
    Limit execution time       →  Timeout decorator
    Share data between nodes   →  Blackboard
    React to conditions        →  Condition at sequence start
```

---

## Further Reading

**Foundational Texts:**
- Colledanchise & Ögren, "Behavior Trees in Robotics and AI" - The comprehensive reference
- Marzinotto et al., "Towards a Unified Behavior Trees Framework" (2014)

**Software Resources:**
- [BehaviorTree.CPP](https://www.behaviortree.dev/) - Industry standard C++ library
- [py_trees](https://py-trees.readthedocs.io/) - Python implementation
- [Nav2 BT Navigator](https://navigation.ros.org/behavior_trees/index.html)

**Tutorials:**
- [ROS 2 Nav2 Behavior Tree Tutorial](https://navigation.ros.org/tutorials/docs/using_groot.html)
- [Groot BT Editor](https://www.behaviortree.dev/groot) - Visual tree editor
