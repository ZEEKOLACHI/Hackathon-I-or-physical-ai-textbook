---
id: ch-5-13
title: Reinforcement Learning for Robotics
sidebar_position: 1
difficulty: advanced
estimated_time: 55
prerequisites: [ch-4-12]
---

# Reinforcement Learning for Robotics

> "The key insight of reinforcement learning is that an agent can learn optimal behavior through trial and error, without being explicitly programmed for every situation."
> — Richard Sutton, Father of Reinforcement Learning

A child learning to walk doesn't follow a manual—they try, fall, adjust, and try again. Reinforcement learning brings this same learning-by-doing paradigm to robotics. Instead of hand-coding every behavior, we let robots discover optimal strategies through interaction with their environment.

## The Reinforcement Learning Framework

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    REINFORCEMENT LEARNING LOOP                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                          ┌─────────────────┐                                │
│                          │   Environment   │                                │
│                          │                 │                                │
│                          │  ┌───────────┐  │                                │
│              Action aₜ   │  │  State    │  │   State sₜ                    │
│           ─────────────▶ │  │  sₜ₊₁     │  │ ───────────▶                  │
│          │               │  └───────────┘  │              │                 │
│          │               │        │        │              │                 │
│          │               │        │ rₜ₊₁   │              │                 │
│          │               └────────┼────────┘              │                 │
│          │                        │                       │                 │
│          │                   Reward                       │                 │
│          │                        │                       │                 │
│          │               ┌────────▼────────┐              │                 │
│          │               │                 │              │                 │
│          └───────────────│     Agent       │◀─────────────┘                 │
│                          │    (Policy π)   │                                │
│                          │                 │                                │
│                          └─────────────────┘                                │
│                                                                             │
│   Goal: Find policy π* that maximizes expected cumulative reward:           │
│                                                                             │
│         π* = argmax E[ Σ γᵗ rₜ ]                                           │
│                π     t=0                                                    │
│                                                                             │
│   Where γ ∈ [0,1] is the discount factor                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Markov Decision Process Foundation

Every RL problem is formalized as a Markov Decision Process (MDP), which provides the mathematical framework for sequential decision-making.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MARKOV DECISION PROCESS (MDP)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Components:                                                               │
│                                                                             │
│   S = State Space        Example (Robot Arm):                               │
│   ┌─────────────────┐    • Joint positions [θ₁, θ₂, θ₃, θ₄, θ₅, θ₆]       │
│   │ s₁  s₂  s₃ ... │    • Joint velocities [θ̇₁, θ̇₂, ...]                  │
│   └─────────────────┘    • End-effector pose                                │
│                                                                             │
│   A = Action Space       Example:                                           │
│   ┌─────────────────┐    • Joint torques [τ₁, τ₂, τ₃, τ₄, τ₅, τ₆]         │
│   │ a₁  a₂  a₃ ... │    • Velocity commands                                │
│   └─────────────────┘    • Discrete: left/right/up/down                    │
│                                                                             │
│   P(s'|s,a) = Transition Probability                                       │
│   ┌─────────────────┐                                                       │
│   │    Given state s and action a,                                         │
│   │    probability of reaching s'                                          │
│   │    (determined by physics)                                             │
│   └─────────────────┘                                                       │
│                                                                             │
│   R(s,a,s') = Reward Function                                              │
│   ┌─────────────────┐    Examples:                                         │
│   │ Scalar feedback │    • +1 for reaching goal                            │
│   │ for each step   │    • -1 for collision                                │
│   └─────────────────┘    • -0.01 per timestep (encourage speed)            │
│                                                                             │
│   γ = Discount Factor (0 ≤ γ ≤ 1)                                          │
│   • γ = 0: Only immediate reward matters                                   │
│   • γ = 1: All future rewards equally important                            │
│   • γ = 0.99: Typical value, balances short and long term                  │
│                                                                             │
│   Markov Property: P(sₜ₊₁|sₜ,aₜ) = P(sₜ₊₁|s₀,a₀,...,sₜ,aₜ)               │
│   "The future depends only on the present, not the past"                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### MDP Implementation for Robotics

```python
"""
Markov Decision Process Framework for Robotics

Provides the foundational structure for all RL algorithms. The MDP
formalism allows us to precisely define what the robot should learn.

Key insight: A well-designed MDP is half the battle. Poor state
representation or reward shaping leads to poor policies.
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List, Optional, Any


@dataclass
class Transition:
    """Single step of experience."""
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    info: dict = None


class RobotMDP(ABC):
    """
    Abstract base class for robot MDPs.

    Defines the interface that all robot environments must implement.
    This follows the OpenAI Gym convention for compatibility.

    Example Usage:
    ```
    env = ReachingTaskMDP()
    state = env.reset()

    for _ in range(1000):
        action = policy.select_action(state)
        next_state, reward, done, info = env.step(action)

        if done:
            state = env.reset()
        else:
            state = next_state
    ```
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        dt: float = 0.01
    ):
        """
        Initialize MDP.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            gamma: Discount factor
            dt: Simulation timestep
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.dt = dt

        # State and action bounds
        self.state_low = -np.inf * np.ones(state_dim)
        self.state_high = np.inf * np.ones(state_dim)
        self.action_low = -np.ones(action_dim)
        self.action_high = np.ones(action_dim)

    @abstractmethod
    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state.

        Returns:
            Initial state observation
        """
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute action and return transition.

        Args:
            action: Action to execute

        Returns:
            Tuple of (next_state, reward, done, info)
        """
        pass

    @abstractmethod
    def compute_reward(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray
    ) -> float:
        """
        Compute reward for transition.

        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state

        Returns:
            Scalar reward
        """
        pass

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state to [-1, 1] range."""
        return 2 * (state - self.state_low) / (self.state_high - self.state_low) - 1

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        """Normalize action to [-1, 1] range."""
        return 2 * (action - self.action_low) / (self.action_high - self.action_low) - 1

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Convert normalized action to actual action."""
        return (action + 1) * (self.action_high - self.action_low) / 2 + self.action_low


class ReachingTaskMDP(RobotMDP):
    """
    Simple reaching task for a robot arm.

    State: [joint_positions, joint_velocities, target_position]
    Action: [joint_torques] or [joint_velocities]
    Reward: -distance_to_target - action_penalty

    ┌─────────────────────────────────────────────────────────────────────┐
    │                    REACHING TASK                                    │
    │                                                                     │
    │                     Target ⊕                                        │
    │                      ╱                                              │
    │                     ╱  d = distance                                 │
    │                    ╱                                                │
    │            ┌──────●  End effector                                   │
    │            │     ╱                                                  │
    │            │    ╱                                                   │
    │            │   ╱  Link 2                                            │
    │            │  ●                                                     │
    │            │ ╱                                                      │
    │            │╱  Link 1                                               │
    │            ●───────  Base                                           │
    │                                                                     │
    │   Reward = -d - α||τ||²   (minimize distance and effort)           │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        n_joints: int = 6,
        max_torque: float = 100.0,
        goal_threshold: float = 0.05
    ):
        """
        Initialize reaching task.

        Args:
            n_joints: Number of robot joints
            max_torque: Maximum joint torque
            goal_threshold: Distance threshold for success
        """
        # State: joints (pos + vel) + target (3D)
        state_dim = 2 * n_joints + 3
        action_dim = n_joints

        super().__init__(state_dim, action_dim)

        self.n_joints = n_joints
        self.max_torque = max_torque
        self.goal_threshold = goal_threshold

        # Set action bounds
        self.action_low = -max_torque * np.ones(n_joints)
        self.action_high = max_torque * np.ones(n_joints)

        # Internal state
        self.joint_pos = None
        self.joint_vel = None
        self.target = None
        self.steps = 0
        self.max_steps = 200

    def reset(self) -> np.ndarray:
        """Reset to random initial state with random target."""
        # Random initial joint positions
        self.joint_pos = np.random.uniform(-np.pi/2, np.pi/2, self.n_joints)
        self.joint_vel = np.zeros(self.n_joints)

        # Random target in workspace
        self.target = np.random.uniform(-1, 1, 3)
        self.target[2] = abs(self.target[2])  # Above ground

        self.steps = 0
        return self._get_state()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute action and simulate one timestep."""
        # Clip action to bounds
        action = np.clip(action, self.action_low, self.action_high)

        # Simple dynamics simulation (in practice, use physics engine)
        # τ = M(q)q̈ + C(q,q̇)q̇ + g(q)
        # Simplified: q̈ = τ (unit mass, no gravity/Coriolis)
        q_ddot = action / 10.0  # Scaled for stability

        # Integrate
        self.joint_vel += q_ddot * self.dt
        self.joint_pos += self.joint_vel * self.dt

        # Clip positions to joint limits
        self.joint_pos = np.clip(self.joint_pos, -np.pi, np.pi)

        self.steps += 1
        next_state = self._get_state()

        # Compute reward
        reward = self.compute_reward(None, action, next_state)

        # Check termination
        ee_pos = self._forward_kinematics(self.joint_pos)
        distance = np.linalg.norm(ee_pos - self.target)

        success = distance < self.goal_threshold
        timeout = self.steps >= self.max_steps
        done = success or timeout

        info = {
            'success': success,
            'distance': distance,
            'steps': self.steps
        }

        return next_state, reward, done, info

    def compute_reward(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray
    ) -> float:
        """
        Compute shaped reward.

        Reward components:
        1. Distance to target (main objective)
        2. Action penalty (encourage efficiency)
        3. Bonus for reaching target
        """
        ee_pos = self._forward_kinematics(self.joint_pos)
        distance = np.linalg.norm(ee_pos - self.target)

        # Distance reward (negative, closer is better)
        reward = -distance

        # Action penalty
        action_cost = 0.001 * np.sum(action ** 2)
        reward -= action_cost

        # Success bonus
        if distance < self.goal_threshold:
            reward += 10.0

        return reward

    def _get_state(self) -> np.ndarray:
        """Construct state observation."""
        return np.concatenate([
            self.joint_pos,
            self.joint_vel,
            self.target
        ])

    def _forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """
        Simplified forward kinematics.

        In practice, use proper FK from robot model (URDF, Pinocchio, etc.)
        """
        # Simple planar approximation for demonstration
        link_length = 0.3

        x = 0
        y = 0
        z = 0.1  # Base height
        angle = 0

        for i, theta in enumerate(q[:3]):  # Use first 3 joints
            angle += theta
            x += link_length * np.cos(angle)
            z += link_length * np.sin(angle)

        return np.array([x, y, z])


# Example usage
if __name__ == "__main__":
    print("Robot MDP Example: Reaching Task")
    print("=" * 50)

    env = ReachingTaskMDP(n_joints=6)
    state = env.reset()

    print(f"State dimension: {env.state_dim}")
    print(f"Action dimension: {env.action_dim}")
    print(f"Initial state shape: {state.shape}")

    # Random actions for demonstration
    total_reward = 0
    for step in range(50):
        action = np.random.uniform(-10, 10, env.action_dim)
        next_state, reward, done, info = env.step(action)
        total_reward += reward

        if done:
            print(f"Episode ended at step {step + 1}")
            print(f"Success: {info['success']}")
            print(f"Final distance: {info['distance']:.3f}")
            break

        state = next_state

    print(f"Total reward: {total_reward:.2f}")
```

## Value-Based Methods

Value-based methods learn a value function that estimates expected future rewards, then derive a policy from it.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    VALUE FUNCTIONS                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   State Value Function V(s):                                                │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   V^π(s) = E_π[ Σ γᵗrₜ | s₀ = s ]                                  │  │
│   │              t=0                                                    │  │
│   │                                                                     │  │
│   │   "Expected return starting from state s, following policy π"      │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Action-Value Function Q(s,a):                                            │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   Q^π(s,a) = E_π[ Σ γᵗrₜ | s₀ = s, a₀ = a ]                        │  │
│   │                t=0                                                  │  │
│   │                                                                     │  │
│   │   "Expected return starting from (s,a), then following policy π"   │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Bellman Equation (Recursive relationship):                               │
│                                                                             │
│   Q*(s,a) = E[ r + γ max Q*(s',a') ]                                       │
│                      a'                                                     │
│                                                                             │
│   Optimal Policy:  π*(s) = argmax Q*(s,a)                                  │
│                             a                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Q-Learning Implementation

```python
"""
Q-Learning for Discrete Action Spaces

Q-Learning is the foundational value-based RL algorithm. It learns
the optimal action-value function Q*(s,a) through temporal difference
updates.

Key insight: Q-learning is "off-policy" - it learns Q* regardless
of what policy generates the experience.

Update Rule:
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
                   └──────────────────────────────┘
                           TD target - current
"""

import numpy as np
from typing import Tuple, List, Optional
from collections import defaultdict


class QLearning:
    """
    Tabular Q-Learning algorithm.

    Works for:
    - Discrete state spaces
    - Discrete action spaces
    - Small state-action spaces (tabular storage)

    Block Diagram:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                                                                     │
    │   State ────▶ ┌──────────┐                                         │
    │               │  Q-Table │                                         │
    │               │ ┌───────┐│    ε-greedy                             │
    │   Action ◀─── │ │Q(s,a) ││ ─────────▶ Action                       │
    │               │ └───────┘│                                         │
    │               └──────────┘                                         │
    │                    ▲                                               │
    │                    │ TD Update                                     │
    │                    │                                               │
    │   (s,a,r,s') ──────┘                                               │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01
    ):
        """
        Initialize Q-Learning agent.

        Args:
            n_states: Number of discrete states
            n_actions: Number of discrete actions
            learning_rate: Step size for Q updates (α)
            gamma: Discount factor for future rewards
            epsilon: Exploration probability
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon value
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Initialize Q-table with zeros (optimistic: small positive values)
        self.Q = np.zeros((n_states, n_actions))

        # Statistics
        self.episode_rewards = []
        self.td_errors = []

    def select_action(self, state: int, training: bool = True) -> int:
        """
        Select action using ε-greedy policy.

        With probability ε: random action (exploration)
        With probability 1-ε: best action (exploitation)

        Args:
            state: Current state index
            training: Whether in training mode (affects exploration)

        Returns:
            Selected action index
        """
        if training and np.random.random() < self.epsilon:
            # Exploration: random action
            return np.random.randint(self.n_actions)
        else:
            # Exploitation: best known action
            return np.argmax(self.Q[state])

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool
    ) -> float:
        """
        Update Q-value using temporal difference learning.

        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Whether episode ended

        Returns:
            TD error (for monitoring)
        """
        # Current Q-value
        current_q = self.Q[state, action]

        # TD target
        if done:
            # Terminal state: no future rewards
            target = reward
        else:
            # Non-terminal: include discounted future value
            target = reward + self.gamma * np.max(self.Q[next_state])

        # TD error
        td_error = target - current_q

        # Update Q-value
        self.Q[state, action] += self.lr * td_error

        # Track TD error
        self.td_errors.append(abs(td_error))

        return td_error

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_policy(self) -> np.ndarray:
        """Extract greedy policy from Q-table."""
        return np.argmax(self.Q, axis=1)

    def get_value_function(self) -> np.ndarray:
        """Extract state value function from Q-table."""
        return np.max(self.Q, axis=1)


class DoubleQLearning(QLearning):
    """
    Double Q-Learning to reduce overestimation bias.

    Standard Q-learning tends to overestimate Q-values because it uses
    the max operator for both selection and evaluation.

    Double Q-learning uses two Q-tables:
    - One for action selection
    - One for value estimation

    This decouples the max and reduces bias.

    Reference: van Hasselt (2010) "Double Q-learning"
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Second Q-table
        self.Q2 = np.zeros_like(self.Q)

    def select_action(self, state: int, training: bool = True) -> int:
        """Select action using average of both Q-tables."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            # Use average Q-values
            avg_q = (self.Q[state] + self.Q2[state]) / 2
            return np.argmax(avg_q)

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool
    ) -> float:
        """Update using Double Q-learning."""
        # Randomly choose which Q-table to update
        if np.random.random() < 0.5:
            Q_update, Q_eval = self.Q, self.Q2
        else:
            Q_update, Q_eval = self.Q2, self.Q

        current_q = Q_update[state, action]

        if done:
            target = reward
        else:
            # Use Q_update to select action, Q_eval to evaluate
            best_action = np.argmax(Q_update[next_state])
            target = reward + self.gamma * Q_eval[next_state, best_action]

        td_error = target - current_q
        Q_update[state, action] += self.lr * td_error

        return td_error


# Grid world example
class GridWorld:
    """
    Simple grid world environment for testing Q-learning.

    ┌───┬───┬───┬───┐
    │ S │   │   │ G │
    ├───┼───┼───┼───┤
    │   │ X │   │   │
    ├───┼───┼───┼───┤
    │   │   │   │   │
    └───┴───┴───┴───┘

    S = Start, G = Goal (reward +10), X = Obstacle (reward -5)
    Actions: 0=up, 1=right, 2=down, 3=left
    """

    def __init__(self, size: int = 4):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4

        self.start = 0
        self.goal = size - 1
        self.obstacles = [size + 1]  # One obstacle

        self.state = self.start

    def reset(self) -> int:
        self.state = self.start
        return self.state

    def step(self, action: int) -> Tuple[int, float, bool]:
        row = self.state // self.size
        col = self.state % self.size

        # Action effects
        if action == 0 and row > 0:  # Up
            row -= 1
        elif action == 1 and col < self.size - 1:  # Right
            col += 1
        elif action == 2 and row < self.size - 1:  # Down
            row += 1
        elif action == 3 and col > 0:  # Left
            col -= 1

        new_state = row * self.size + col

        # Check obstacles
        if new_state in self.obstacles:
            reward = -5
            new_state = self.state  # Don't move into obstacle
        elif new_state == self.goal:
            reward = 10
        else:
            reward = -0.1  # Small penalty for each step

        self.state = new_state
        done = (new_state == self.goal)

        return new_state, reward, done


# Training example
if __name__ == "__main__":
    print("Q-Learning Demo: Grid World")
    print("=" * 50)

    env = GridWorld(size=4)
    agent = QLearning(
        n_states=env.n_states,
        n_actions=env.n_actions,
        learning_rate=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995
    )

    n_episodes = 500
    episode_rewards = []

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0

        for step in range(100):  # Max steps per episode
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)

            agent.update(state, action, reward, next_state, done)

            total_reward += reward
            state = next_state

            if done:
                break

        episode_rewards.append(total_reward)
        agent.decay_epsilon()

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}: Avg Reward = {avg_reward:.2f}, ε = {agent.epsilon:.3f}")

    # Show learned policy
    print("\nLearned Policy:")
    policy = agent.get_policy()
    actions = ['↑', '→', '↓', '←']
    for row in range(env.size):
        line = ""
        for col in range(env.size):
            s = row * env.size + col
            if s == env.goal:
                line += " G "
            elif s in env.obstacles:
                line += " X "
            else:
                line += f" {actions[policy[s]]} "
        print(line)
```

## Deep Q-Networks (DQN)

For continuous or high-dimensional state spaces, we approximate Q-values with neural networks.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DEEP Q-NETWORK (DQN)                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Problem: Tabular Q-learning doesn't scale                                │
│   • 6-DOF arm with 1° resolution = 360⁶ ≈ 2.2 trillion states            │
│   • Continuous states: infinite states                                     │
│                                                                             │
│   Solution: Function approximation with neural networks                     │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   State s ───▶ ┌─────────────────────────┐                         │  │
│   │                │   Neural Network θ       │                         │  │
│   │                │   ┌─────┐ ┌─────┐       │                         │  │
│   │                │   │Dense│→│Dense│→...   │────▶ Q(s,a₁)            │  │
│   │                │   └─────┘ └─────┘       │────▶ Q(s,a₂)            │  │
│   │                │                         │────▶ Q(s,a₃)            │  │
│   │                └─────────────────────────┘────▶ ...                │  │
│   │                                                                     │  │
│   │   Input: State (continuous)                                        │  │
│   │   Output: Q-value for each action                                  │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Key Innovations for Stability:                                           │
│                                                                             │
│   1. Experience Replay                                                     │
│      ┌────────────────────────────────────────────┐                        │
│      │ Buffer: [(s,a,r,s'), (s,a,r,s'), ...]     │                        │
│      │ Sample random minibatch for training       │                        │
│      │ Breaks temporal correlation                │                        │
│      └────────────────────────────────────────────┘                        │
│                                                                             │
│   2. Target Network                                                        │
│      ┌────────────────────────────────────────────┐                        │
│      │ Q_target(θ⁻): Slow-moving copy of Q(θ)   │                        │
│      │ Updated periodically: θ⁻ ← θ              │                        │
│      │ Stabilizes TD target                       │                        │
│      └────────────────────────────────────────────┘                        │
│                                                                             │
│   Loss: L = E[(r + γ max_a' Q_target(s',a') - Q(s,a))²]                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### DQN Implementation

```python
"""
Deep Q-Network (DQN) for Continuous State Spaces

DQN combines Q-learning with deep neural networks, enabling RL
for complex, high-dimensional state spaces like robot observations.

Key techniques for stability:
1. Experience Replay: Store and sample past experiences
2. Target Network: Use separate network for TD targets
3. Gradient Clipping: Prevent exploding gradients

Reference: Mnih et al. (2015) "Human-level control through deep RL"
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Tuple, List


class ReplayBuffer:
    """
    Experience replay buffer for DQN.

    Stores transitions and samples random minibatches.
    Breaks temporal correlation in training data.
    """

    def __init__(self, capacity: int = 100000):
        """
        Initialize buffer.

        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Add transition to buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Sample random minibatch.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)

        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch])

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    """
    Neural network for Q-function approximation.

    Architecture:
    State → [Linear → ReLU] × N → Q-values for each action
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256]
    ):
        """
        Initialize Q-network.

        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            hidden_dims: Sizes of hidden layers
        """
        super().__init__()

        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass: state → Q-values."""
        return self.network(state)


class DQNAgent:
    """
    Deep Q-Network agent.

    Learns to act in environments with continuous states
    and discrete actions.

    Block Diagram:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                                                                     │
    │   Environment ◀──────────────────────────────────────┐              │
    │       │                                              │              │
    │       │ state                                     action            │
    │       ▼                                              │              │
    │   ┌───────────┐    ┌───────────┐    ┌───────────┐   │              │
    │   │  Replay   │◀───│   Step    │───▶│  Select   │───┘              │
    │   │  Buffer   │    └───────────┘    │  Action   │                  │
    │   └─────┬─────┘                     └───────────┘                  │
    │         │                                 ▲                        │
    │         │ sample                          │                        │
    │         ▼                                 │                        │
    │   ┌───────────┐    ┌───────────┐         │                        │
    │   │   Train   │───▶│  Q-Net    │─────────┘                        │
    │   │   Step    │    │    θ      │                                   │
    │   └───────────┘    └───────────┘                                   │
    │         │                                                          │
    │         │ periodic                                                 │
    │         ▼                                                          │
    │   ┌───────────┐                                                    │
    │   │ Target Net│    θ⁻ ← θ                                         │
    │   │    θ⁻     │                                                    │
    │   └───────────┘                                                    │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        device: str = "cpu"
    ):
        """
        Initialize DQN agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            learning_rate: Optimizer learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay per episode
            epsilon_min: Minimum epsilon
            buffer_size: Replay buffer capacity
            batch_size: Training batch size
            target_update_freq: Steps between target network updates
            device: Torch device (cpu/cuda)
        """
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)

        # Q-networks
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)

        # Training stats
        self.train_steps = 0
        self.losses = []

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using ε-greedy policy.

        Args:
            state: Current state observation
            training: Whether in training mode

        Returns:
            Selected action index
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Store transition in replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> float:
        """
        Perform one training step.

        Returns:
            Loss value (or 0 if buffer too small)
        """
        if len(self.buffer) < self.batch_size:
            return 0.0

        # Sample minibatch
        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size
        )

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Target Q-values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Compute loss
        loss = nn.MSELoss()(current_q.squeeze(), target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)

        self.optimizer.step()

        # Update target network periodically
        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        loss_val = loss.item()
        self.losses.append(loss_val)

        return loss_val

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str):
        """Save model weights."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)

    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']


# Example usage
if __name__ == "__main__":
    print("DQN Example")
    print("=" * 50)

    # Simple environment simulation
    state_dim = 8
    action_dim = 4

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=1e-3,
        gamma=0.99
    )

    # Simulate training loop
    for episode in range(10):
        state = np.random.randn(state_dim)
        episode_reward = 0

        for step in range(100):
            action = agent.select_action(state)

            # Simulated environment step
            next_state = np.random.randn(state_dim)
            reward = np.random.randn()
            done = step == 99

            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.train_step()

            episode_reward += reward
            state = next_state

            if done:
                break

        agent.decay_epsilon()
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, ε = {agent.epsilon:.3f}")
```

## Policy Gradient Methods

Instead of learning values, policy gradient methods directly learn the policy.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    POLICY GRADIENT METHODS                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Key Idea: Directly parameterize and optimize the policy                  │
│                                                                             │
│   Policy: π_θ(a|s) = P(action = a | state = s, parameters = θ)            │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   State s ───▶ ┌─────────────────────────┐                         │  │
│   │                │   Policy Network π_θ    │                         │  │
│   │                │   ┌─────┐ ┌─────┐       │                         │  │
│   │                │   │Dense│→│Dense│→...   │                         │  │
│   │                │   └─────┘ └─────┘       │                         │  │
│   │                └────────────┬────────────┘                         │  │
│   │                             │                                       │  │
│   │                             ▼                                       │  │
│   │                     ┌───────────────┐                              │  │
│   │                     │  Distribution │   (e.g., Gaussian)           │  │
│   │                     │  π(a|s; θ)    │                              │  │
│   │                     └───────┬───────┘                              │  │
│   │                             │                                       │  │
│   │                             ▼                                       │  │
│   │                     ┌───────────────┐                              │  │
│   │                     │ Sample Action │                              │  │
│   │                     │    a ~ π      │                              │  │
│   │                     └───────────────┘                              │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Policy Gradient Theorem:                                                 │
│                                                                             │
│   ∇_θ J(θ) = E_π[ ∇_θ log π_θ(a|s) · Q^π(s,a) ]                           │
│                                                                             │
│   "Increase probability of actions with high Q-values"                     │
│                                                                             │
│   Advantage:                                                               │
│   • Can learn stochastic policies                                          │
│   • Works with continuous actions                                          │
│   • Smoother optimization landscape                                        │
│                                                                             │
│   Disadvantage:                                                            │
│   • High variance gradients                                                │
│   • Sample inefficient                                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Proximal Policy Optimization (PPO)

```python
"""
Proximal Policy Optimization (PPO)

PPO is the most widely used policy gradient algorithm for robotics.
It's robust, sample efficient, and works well across many domains.

Key innovations:
1. Clipped surrogate objective - prevents too large policy updates
2. Multiple epochs per batch - better sample efficiency
3. Advantage estimation - reduces variance

Reference: Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import Tuple, List


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.

    Actor (Policy): Maps state → action distribution
    Critic (Value): Maps state → expected return

    Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                                                                     │
    │   State ───▶ ┌─────────────────────────┐                           │
    │              │    Shared Backbone      │                           │
    │              │   (optional sharing)    │                           │
    │              └───────────┬─────────────┘                           │
    │                          │                                         │
    │              ┌───────────┴───────────┐                             │
    │              ▼                       ▼                             │
    │        ┌──────────┐           ┌──────────┐                        │
    │        │  Actor   │           │  Critic  │                        │
    │        │  (μ, σ)  │           │   V(s)   │                        │
    │        └──────────┘           └──────────┘                        │
    │              │                       │                             │
    │              ▼                       ▼                             │
    │        Action ~ N(μ,σ)         State Value                        │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        log_std_init: float = 0.0
    ):
        """
        Initialize actor-critic network.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: Sizes of hidden layers
            log_std_init: Initial log standard deviation
        """
        super().__init__()

        # Actor network (policy)
        actor_layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            actor_layers.append(nn.Linear(prev_dim, hidden_dim))
            actor_layers.append(nn.Tanh())  # Tanh is common for policy nets
            prev_dim = hidden_dim

        self.actor_backbone = nn.Sequential(*actor_layers)
        self.actor_mean = nn.Linear(prev_dim, action_dim)

        # Learnable log standard deviation
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)

        # Critic network (value function)
        critic_layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            critic_layers.append(nn.Linear(prev_dim, hidden_dim))
            critic_layers.append(nn.Tanh())
            prev_dim = hidden_dim
        critic_layers.append(nn.Linear(prev_dim, 1))

        self.critic = nn.Sequential(*critic_layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def forward(self, state: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        """
        Forward pass.

        Args:
            state: State tensor

        Returns:
            Tuple of (action_distribution, state_value)
        """
        # Actor
        actor_features = self.actor_backbone(state)
        action_mean = self.actor_mean(actor_features)
        action_std = self.log_std.exp().expand_as(action_mean)
        action_dist = Normal(action_mean, action_std)

        # Critic
        value = self.critic(state)

        return action_dist, value

    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Args:
            state: State tensor
            deterministic: Whether to use mean action

        Returns:
            Tuple of (action, log_prob, value)
        """
        action_dist, value = self.forward(state)

        if deterministic:
            action = action_dist.mean
        else:
            action = action_dist.sample()

        log_prob = action_dist.log_prob(action).sum(-1, keepdim=True)

        return action, log_prob, value

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.

        Args:
            states: Batch of states
            actions: Batch of actions

        Returns:
            Tuple of (log_probs, values, entropy)
        """
        action_dist, values = self.forward(states)

        log_probs = action_dist.log_prob(actions).sum(-1, keepdim=True)
        entropy = action_dist.entropy().sum(-1, keepdim=True)

        return log_probs, values, entropy


class RolloutBuffer:
    """
    Buffer for storing rollout data during policy execution.
    """

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """Add transition to buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def clear(self):
        """Clear buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute returns and GAE advantages.

        Generalized Advantage Estimation (GAE):
        A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
        where δ_t = r_t + γV(s_{t+1}) - V(s_t)

        Args:
            last_value: Value of final state
            gamma: Discount factor
            gae_lambda: GAE lambda parameter

        Returns:
            Tuple of (returns, advantages)
        """
        rewards = np.array(self.rewards)
        values = np.array(self.values + [last_value])
        dones = np.array(self.dones + [False])

        n = len(rewards)
        advantages = np.zeros(n)
        last_gae = 0

        # Compute GAE backwards
        for t in reversed(range(n)):
            if dones[t]:
                delta = rewards[t] - values[t]
                last_gae = delta
            else:
                delta = rewards[t] + gamma * values[t + 1] - values[t]
                last_gae = delta + gamma * gae_lambda * last_gae

            advantages[t] = last_gae

        returns = advantages + np.array(self.values)

        return returns, advantages

    def get_batches(
        self,
        batch_size: int,
        returns: np.ndarray,
        advantages: np.ndarray
    ):
        """
        Generate random minibatches for training.

        Yields:
            Tuple of tensors for each minibatch
        """
        n = len(self.states)
        indices = np.random.permutation(n)

        for start in range(0, n, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]

            yield (
                torch.FloatTensor(np.array(self.states)[batch_indices]),
                torch.FloatTensor(np.array(self.actions)[batch_indices]),
                torch.FloatTensor(np.array(self.log_probs)[batch_indices]),
                torch.FloatTensor(returns[batch_indices]),
                torch.FloatTensor(advantages[batch_indices])
            )


class PPO:
    """
    Proximal Policy Optimization algorithm.

    PPO Objective:
    L^CLIP(θ) = E[ min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A) ]

    where r(θ) = π_θ(a|s) / π_θ_old(a|s)

    This clips the objective to prevent too large policy updates.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        device: str = "cpu"
    ):
        """
        Initialize PPO.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm
            n_epochs: Training epochs per update
            batch_size: Minibatch size
            device: Torch device
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = torch.device(device)

        # Networks
        self.actor_critic = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)

        # Rollout buffer
        self.buffer = RolloutBuffer()

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """
        Select action from policy.

        Args:
            state: State observation
            deterministic: Use mean action

        Returns:
            Tuple of (action, log_prob, value)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, value = self.actor_critic.get_action(
                state_tensor, deterministic
            )

        return (
            action.cpu().numpy().squeeze(),
            log_prob.cpu().item(),
            value.cpu().item()
        )

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """Store transition in rollout buffer."""
        self.buffer.push(state, action, reward, value, log_prob, done)

    def update(self, last_value: float) -> dict:
        """
        Perform PPO update.

        Args:
            last_value: Value estimate for final state

        Returns:
            Dictionary of training statistics
        """
        # Compute returns and advantages
        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value, self.gamma, self.gae_lambda
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Training statistics
        policy_losses = []
        value_losses = []
        entropy_losses = []

        # Multiple epochs over the data
        for epoch in range(self.n_epochs):
            for batch in self.buffer.get_batches(
                self.batch_size, returns, advantages
            ):
                states, actions, old_log_probs, batch_returns, batch_advantages = batch

                # Move to device
                states = states.to(self.device)
                actions = actions.to(self.device)
                old_log_probs = old_log_probs.to(self.device)
                batch_returns = batch_returns.to(self.device)
                batch_advantages = batch_advantages.to(self.device)

                # Evaluate actions with current policy
                log_probs, values, entropy = self.actor_critic.evaluate_actions(
                    states, actions
                )

                # Policy loss (clipped surrogate)
                ratio = (log_probs - old_log_probs.unsqueeze(1)).exp()
                surr1 = ratio * batch_advantages.unsqueeze(1)
                surr2 = torch.clamp(
                    ratio,
                    1 - self.clip_epsilon,
                    1 + self.clip_epsilon
                ) * batch_advantages.unsqueeze(1)
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.MSELoss()(values.squeeze(), batch_returns)

                # Entropy loss (negative because we want to maximize entropy)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss +
                    self.value_coef * value_loss +
                    self.entropy_coef * entropy_loss
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(),
                    self.max_grad_norm
                )
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(-entropy_loss.item())

        # Clear buffer
        self.buffer.clear()

        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropy_losses)
        }

    def save(self, path: str):
        """Save model."""
        torch.save(self.actor_critic.state_dict(), path)

    def load(self, path: str):
        """Load model."""
        self.actor_critic.load_state_dict(torch.load(path))


# Example usage
if __name__ == "__main__":
    print("PPO Example")
    print("=" * 50)

    state_dim = 24  # e.g., robot joint states
    action_dim = 6  # e.g., joint torques

    ppo = PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=3e-4,
        gamma=0.99,
        n_epochs=10
    )

    # Simulated training loop
    rollout_length = 128
    n_updates = 5

    for update in range(n_updates):
        state = np.random.randn(state_dim)

        # Collect rollout
        for step in range(rollout_length):
            action, log_prob, value = ppo.select_action(state)

            # Simulated environment step
            next_state = np.random.randn(state_dim)
            reward = np.random.randn()
            done = step == rollout_length - 1

            ppo.store_transition(state, action, reward, value, log_prob, done)
            state = next_state

        # Get final value estimate
        _, _, last_value = ppo.select_action(state)

        # PPO update
        stats = ppo.update(last_value)

        print(f"Update {update + 1}: "
              f"Policy Loss = {stats['policy_loss']:.4f}, "
              f"Value Loss = {stats['value_loss']:.4f}, "
              f"Entropy = {stats['entropy']:.4f}")
```

## Sim-to-Real Transfer

One of the biggest challenges in robot RL is transferring policies from simulation to real hardware.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SIM-TO-REAL TRANSFER                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   The Reality Gap:                                                         │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   Simulation                          Reality                       │  │
│   │   ┌───────────┐                       ┌───────────┐                │  │
│   │   │ Perfect   │                       │ Noise,    │                │  │
│   │   │ Physics   │      ≠               │ Delays,   │                │  │
│   │   │ No Noise  │                       │ Friction  │                │  │
│   │   └───────────┘                       └───────────┘                │  │
│   │                                                                     │  │
│   │   Policy trained in simulation often fails on real robot!          │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Solutions:                                                               │
│                                                                             │
│   1. DOMAIN RANDOMIZATION                                                  │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │   Train on many randomized simulations                              │  │
│   │   • Randomize physics: mass, friction, damping                     │  │
│   │   • Randomize visuals: colors, textures, lighting                  │  │
│   │   • Randomize dynamics: delays, noise                              │  │
│   │   → Policy becomes robust to variations                            │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   2. SYSTEM IDENTIFICATION                                                 │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │   Identify real robot parameters                                    │  │
│   │   • Measure actual mass, friction, etc.                            │  │
│   │   • Tune simulation to match reality                               │  │
│   │   → Simulation becomes more realistic                              │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   3. DOMAIN ADAPTATION                                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │   Learn to adapt at deployment time                                 │  │
│   │   • Meta-learning: Learn to learn quickly                          │  │
│   │   • Online adaptation: Adjust policy on real robot                 │  │
│   │   → Policy adapts to new conditions                                │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Domain Randomization Implementation

```python
"""
Domain Randomization for Sim-to-Real Transfer

Domain randomization trains policies on many variations of the simulation,
making them robust to real-world conditions not seen during training.

Key insight: If the policy works across all variations in simulation,
it will likely work in reality (which is just another variation).

Reference: Tobin et al. (2017) "Domain Randomization for Sim2Real Transfer"
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Callable
from abc import ABC, abstractmethod


@dataclass
class PhysicsParams:
    """Physical parameters that can be randomized."""
    mass: float
    friction: float
    damping: float
    motor_strength: float
    gravity: float
    link_length: float
    sensor_noise: float
    action_delay: int  # timesteps


class DomainRandomizer:
    """
    Domain randomization manager.

    Randomizes simulation parameters within specified ranges.
    Parameters are sampled at episode reset.

    Usage:
    ```
    randomizer = DomainRandomizer()

    # Each episode
    params = randomizer.sample()
    env.set_physics_params(params)
    ```
    """

    def __init__(
        self,
        mass_range: Tuple[float, float] = (0.8, 1.2),
        friction_range: Tuple[float, float] = (0.5, 1.5),
        damping_range: Tuple[float, float] = (0.8, 1.2),
        motor_strength_range: Tuple[float, float] = (0.8, 1.2),
        gravity_range: Tuple[float, float] = (9.5, 10.1),
        link_length_range: Tuple[float, float] = (0.98, 1.02),
        sensor_noise_range: Tuple[float, float] = (0.0, 0.05),
        action_delay_range: Tuple[int, int] = (0, 3)
    ):
        """
        Initialize randomizer with parameter ranges.

        All ranges are specified as (min, max) tuples.
        Parameters are sampled uniformly within ranges.

        Args:
            mass_range: Multiplier for default mass
            friction_range: Friction coefficient range
            damping_range: Damping coefficient multiplier
            motor_strength_range: Motor torque multiplier
            gravity_range: Gravity magnitude range
            link_length_range: Link length multiplier
            sensor_noise_range: Sensor noise std range
            action_delay_range: Action delay range (timesteps)
        """
        self.ranges = {
            'mass': mass_range,
            'friction': friction_range,
            'damping': damping_range,
            'motor_strength': motor_strength_range,
            'gravity': gravity_range,
            'link_length': link_length_range,
            'sensor_noise': sensor_noise_range,
            'action_delay': action_delay_range
        }

        # Default values
        self.defaults = PhysicsParams(
            mass=1.0,
            friction=1.0,
            damping=1.0,
            motor_strength=1.0,
            gravity=9.81,
            link_length=1.0,
            sensor_noise=0.0,
            action_delay=0
        )

    def sample(self) -> PhysicsParams:
        """
        Sample random physics parameters.

        Returns:
            PhysicsParams with randomized values
        """
        return PhysicsParams(
            mass=np.random.uniform(*self.ranges['mass']),
            friction=np.random.uniform(*self.ranges['friction']),
            damping=np.random.uniform(*self.ranges['damping']),
            motor_strength=np.random.uniform(*self.ranges['motor_strength']),
            gravity=np.random.uniform(*self.ranges['gravity']),
            link_length=np.random.uniform(*self.ranges['link_length']),
            sensor_noise=np.random.uniform(*self.ranges['sensor_noise']),
            action_delay=np.random.randint(*self.ranges['action_delay'])
        )

    def sample_gaussian(self, std_scale: float = 0.1) -> PhysicsParams:
        """
        Sample parameters using Gaussian distribution around defaults.

        Args:
            std_scale: Standard deviation as fraction of default

        Returns:
            PhysicsParams with Gaussian-sampled values
        """
        return PhysicsParams(
            mass=np.random.normal(1.0, std_scale),
            friction=np.clip(np.random.normal(1.0, std_scale), 0.1, 2.0),
            damping=np.random.normal(1.0, std_scale),
            motor_strength=np.random.normal(1.0, std_scale),
            gravity=np.random.normal(9.81, 0.3),
            link_length=np.random.normal(1.0, std_scale * 0.2),
            sensor_noise=abs(np.random.normal(0, 0.02)),
            action_delay=max(0, int(np.random.normal(1, 1)))
        )


class RandomizedEnvWrapper:
    """
    Wrapper that applies domain randomization to an environment.

    Automatically:
    - Randomizes physics at episode reset
    - Adds sensor noise to observations
    - Applies action delays
    """

    def __init__(
        self,
        env,
        randomizer: DomainRandomizer,
        randomize_on_reset: bool = True
    ):
        """
        Initialize wrapper.

        Args:
            env: Base environment to wrap
            randomizer: Domain randomizer instance
            randomize_on_reset: Whether to randomize at each reset
        """
        self.env = env
        self.randomizer = randomizer
        self.randomize_on_reset = randomize_on_reset

        self.current_params = randomizer.defaults
        self.action_buffer = []

    def reset(self) -> np.ndarray:
        """Reset environment with randomized parameters."""
        if self.randomize_on_reset:
            self.current_params = self.randomizer.sample()
            self._apply_physics_params()

        # Reset action buffer
        self.action_buffer = [
            np.zeros(self.env.action_dim)
            for _ in range(self.current_params.action_delay + 1)
        ]

        state = self.env.reset()
        return self._add_observation_noise(state)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """Step environment with delays and noise."""
        # Apply action delay
        self.action_buffer.append(action)
        delayed_action = self.action_buffer.pop(0)

        # Execute action
        state, reward, done, info = self.env.step(delayed_action)

        # Add observation noise
        state = self._add_observation_noise(state)

        # Store current randomization in info
        info['physics_params'] = self.current_params

        return state, reward, done, info

    def _apply_physics_params(self):
        """Apply current physics parameters to environment."""
        params = self.current_params

        # These method calls depend on environment implementation
        if hasattr(self.env, 'set_mass_scale'):
            self.env.set_mass_scale(params.mass)
        if hasattr(self.env, 'set_friction'):
            self.env.set_friction(params.friction)
        if hasattr(self.env, 'set_damping'):
            self.env.set_damping(params.damping)
        if hasattr(self.env, 'set_motor_strength'):
            self.env.set_motor_strength(params.motor_strength)
        if hasattr(self.env, 'set_gravity'):
            self.env.set_gravity(params.gravity)

    def _add_observation_noise(self, state: np.ndarray) -> np.ndarray:
        """Add sensor noise to observations."""
        noise = np.random.normal(0, self.current_params.sensor_noise, state.shape)
        return state + noise


class CurriculumRandomizer:
    """
    Curriculum-based domain randomization.

    Starts with less randomization and gradually increases difficulty.
    This can lead to faster learning compared to full randomization
    from the start.

    Reference: OpenAI (2019) "Solving Rubik's Cube with a Robot Hand"
    """

    def __init__(
        self,
        base_randomizer: DomainRandomizer,
        initial_scale: float = 0.1,
        final_scale: float = 1.0,
        schedule_steps: int = 100000
    ):
        """
        Initialize curriculum randomizer.

        Args:
            base_randomizer: Full-scale randomizer
            initial_scale: Starting randomization scale
            final_scale: Final randomization scale
            schedule_steps: Steps to reach final scale
        """
        self.base = base_randomizer
        self.initial_scale = initial_scale
        self.final_scale = final_scale
        self.schedule_steps = schedule_steps

        self.step = 0

    @property
    def current_scale(self) -> float:
        """Get current randomization scale."""
        progress = min(1.0, self.step / self.schedule_steps)
        return self.initial_scale + progress * (self.final_scale - self.initial_scale)

    def sample(self) -> PhysicsParams:
        """Sample parameters with current curriculum scale."""
        scale = self.current_scale

        # Interpolate between default and random
        full_random = self.base.sample()
        defaults = self.base.defaults

        return PhysicsParams(
            mass=defaults.mass + scale * (full_random.mass - defaults.mass),
            friction=defaults.friction + scale * (full_random.friction - defaults.friction),
            damping=defaults.damping + scale * (full_random.damping - defaults.damping),
            motor_strength=defaults.motor_strength + scale * (full_random.motor_strength - defaults.motor_strength),
            gravity=defaults.gravity + scale * (full_random.gravity - defaults.gravity),
            link_length=defaults.link_length + scale * (full_random.link_length - defaults.link_length),
            sensor_noise=scale * full_random.sensor_noise,
            action_delay=int(scale * full_random.action_delay)
        )

    def update(self, n_steps: int = 1):
        """Update curriculum progress."""
        self.step += n_steps


# Example usage
if __name__ == "__main__":
    print("Domain Randomization Demo")
    print("=" * 50)

    # Create randomizer
    randomizer = DomainRandomizer(
        mass_range=(0.7, 1.3),
        friction_range=(0.4, 1.6),
        sensor_noise_range=(0.0, 0.1)
    )

    # Sample some physics configurations
    print("\nSampled Physics Configurations:")
    for i in range(5):
        params = randomizer.sample()
        print(f"\nConfig {i + 1}:")
        print(f"  Mass multiplier: {params.mass:.3f}")
        print(f"  Friction: {params.friction:.3f}")
        print(f"  Sensor noise: {params.sensor_noise:.3f}")
        print(f"  Action delay: {params.action_delay} steps")

    # Curriculum example
    print("\n\nCurriculum Randomization:")
    curriculum = CurriculumRandomizer(
        randomizer,
        initial_scale=0.1,
        final_scale=1.0,
        schedule_steps=1000
    )

    for step in [0, 250, 500, 750, 1000]:
        curriculum.step = step
        params = curriculum.sample()
        print(f"\nStep {step} (scale={curriculum.current_scale:.2f}):")
        print(f"  Mass: {params.mass:.3f}")
        print(f"  Friction: {params.friction:.3f}")
```

## Industry Perspective: RL in Robot Learning

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RL IN INDUSTRY ROBOTICS                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   BOSTON DYNAMICS                                                          │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │ • RL for locomotion: Spot learns to walk on varied terrain     │      │
│   │ • Model-based RL combined with trajectory optimization          │      │
│   │ • Extensive sim-to-real with domain randomization              │      │
│   │ • Recovery behaviors learned through RL                         │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│   GOOGLE DEEPMIND (Robotics)                                               │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │ • RT-1/RT-2: RL fine-tuning of foundation models               │      │
│   │ • Offline RL from human demonstrations                          │      │
│   │ • Multi-task learning across robot fleets                       │      │
│   │ • QT-Opt: Large-scale RL for grasping                          │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│   NVIDIA (Isaac Gym/Lab)                                                   │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │ • Massively parallel RL training on GPU                         │      │
│   │ • Thousands of environments simulated simultaneously            │      │
│   │ • Minutes to hours for policy training                          │      │
│   │ • Integrated domain randomization                               │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│   COVARIANT                                                                │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │ • Deep RL for warehouse picking                                 │      │
│   │ • Continuous adaptation to new objects                          │      │
│   │ • Production systems handling millions of picks                 │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│   Common Patterns:                                                         │
│   • Simulation-first development, transfer to real                         │
│   • Large-scale parallel training                                          │
│   • Combination of RL with other techniques (MPC, IL)                     │
│   • Extensive safety constraints                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Summary

### Key Takeaways

1. **MDP Foundation**: All RL problems are formalized as MDPs with states, actions, transitions, rewards, and discount factor

2. **Value vs Policy Methods**:
   - Value-based (Q-learning, DQN): Learn Q-values, derive policy
   - Policy gradient (PPO): Directly optimize policy

3. **Deep RL**: Neural networks enable RL for continuous, high-dimensional state spaces

4. **Sim-to-Real**: Domain randomization is key to transferring policies from simulation to reality

5. **PPO is Standard**: For most robot learning, PPO is the go-to algorithm due to stability and sample efficiency

### Algorithm Comparison

| Algorithm | State Space | Action Space | Sample Efficiency | Stability |
|-----------|-------------|--------------|-------------------|-----------|
| Q-Learning | Discrete | Discrete | Low | High |
| DQN | Continuous | Discrete | Medium | Medium |
| PPO | Continuous | Continuous | Medium | High |
| SAC | Continuous | Continuous | High | High |
| TD3 | Continuous | Continuous | High | Medium |

### Practical Checklist

- [ ] Designed appropriate state representation
- [ ] Defined clear reward function
- [ ] Set up simulation environment
- [ ] Implemented domain randomization
- [ ] Chosen appropriate algorithm (PPO for most cases)
- [ ] Set up proper hyperparameters
- [ ] Created evaluation metrics
- [ ] Tested sim-to-real transfer
- [ ] Implemented safety constraints

## Further Reading

### Foundational Works
- Sutton, R.S. & Barto, A.G. "Reinforcement Learning: An Introduction" (2018)
- Schulman et al. "Proximal Policy Optimization Algorithms" (2017)

### Robotics Applications
- Levine et al. "Learning Hand-Eye Coordination for Robotic Grasping" (2018)
- Hwangbo et al. "Learning Agile and Dynamic Motor Skills for Legged Robots" (2019)

### Software
- Stable Baselines3: Production-ready RL algorithms
- Isaac Gym: GPU-accelerated robot simulation
- OpenAI Gym: Standard RL environment interface

---

*"Reinforcement learning is not just an algorithm—it's a paradigm shift in how we think about programming robots. Instead of telling them what to do, we tell them what we want."*
