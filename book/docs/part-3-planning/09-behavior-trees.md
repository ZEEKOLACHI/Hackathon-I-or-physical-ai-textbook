---
id: ch-3-09
title: Behavior Trees
sidebar_position: 3
difficulty: intermediate
estimated_time: 30
prerequisites: [ch-3-08]
---

# Behavior Trees

Behavior trees provide a modular, hierarchical way to organize robot behaviors.

## Core Concepts

### Node Types

```python
from enum import Enum

class NodeStatus(Enum):
    SUCCESS = 1
    FAILURE = 2
    RUNNING = 3

class BehaviorNode:
    def tick(self) -> NodeStatus:
        raise NotImplementedError

class ActionNode(BehaviorNode):
    def __init__(self, action_func):
        self.action = action_func

    def tick(self):
        return self.action()

class ConditionNode(BehaviorNode):
    def __init__(self, condition_func):
        self.condition = condition_func

    def tick(self):
        if self.condition():
            return NodeStatus.SUCCESS
        return NodeStatus.FAILURE
```

### Control Flow Nodes

```python
class SequenceNode(BehaviorNode):
    """Execute children in order until one fails."""
    def __init__(self, children):
        self.children = children
        self.current_index = 0

    def tick(self):
        while self.current_index < len(self.children):
            status = self.children[self.current_index].tick()

            if status == NodeStatus.RUNNING:
                return NodeStatus.RUNNING
            elif status == NodeStatus.FAILURE:
                self.current_index = 0
                return NodeStatus.FAILURE

            self.current_index += 1

        self.current_index = 0
        return NodeStatus.SUCCESS

class SelectorNode(BehaviorNode):
    """Execute children until one succeeds."""
    def __init__(self, children):
        self.children = children
        self.current_index = 0

    def tick(self):
        while self.current_index < len(self.children):
            status = self.children[self.current_index].tick()

            if status == NodeStatus.RUNNING:
                return NodeStatus.RUNNING
            elif status == NodeStatus.SUCCESS:
                self.current_index = 0
                return NodeStatus.SUCCESS

            self.current_index += 1

        self.current_index = 0
        return NodeStatus.FAILURE
```

### Decorators

```python
class RepeatNode(BehaviorNode):
    """Repeat child N times."""
    def __init__(self, child, times):
        self.child = child
        self.times = times
        self.count = 0

    def tick(self):
        while self.count < self.times:
            status = self.child.tick()
            if status == NodeStatus.RUNNING:
                return NodeStatus.RUNNING
            elif status == NodeStatus.FAILURE:
                return NodeStatus.FAILURE
            self.count += 1
        self.count = 0
        return NodeStatus.SUCCESS
```

## Example: Pick and Place

```python
def create_pick_place_tree(robot):
    return SelectorNode([
        # Main task sequence
        SequenceNode([
            ConditionNode(lambda: robot.has_target()),
            ActionNode(lambda: robot.move_to_target()),
            ActionNode(lambda: robot.pick_object()),
            ActionNode(lambda: robot.move_to_goal()),
            ActionNode(lambda: robot.place_object()),
        ]),
        # Fallback: search for objects
        SequenceNode([
            ActionNode(lambda: robot.search_for_objects()),
            ActionNode(lambda: robot.select_target()),
        ])
    ])
```

## BehaviorTree.CPP with ROS 2

```cpp
// Example using BehaviorTree.CPP
#include <behaviortree_cpp_v3/bt_factory.h>

class MoveToTarget : public BT::SyncActionNode {
public:
    MoveToTarget(const std::string& name) : BT::SyncActionNode(name, {}) {}

    BT::NodeStatus tick() override {
        // Execute movement
        return BT::NodeStatus::SUCCESS;
    }
};
```

## Summary

- Behavior trees organize complex robot behaviors
- Core nodes: Action, Condition, Sequence, Selector
- Decorators modify child behavior
- Modular design enables reuse and debugging

## Further Reading

- Colledanchise, M. & Ögren, P. "Behavior Trees in Robotics and AI"
- BehaviorTree.CPP Documentation
