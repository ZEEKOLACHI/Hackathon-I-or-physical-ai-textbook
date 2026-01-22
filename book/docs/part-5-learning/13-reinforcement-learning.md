---
id: ch-5-13
title: Reinforcement Learning for Robotics
sidebar_position: 1
difficulty: advanced
estimated_time: 45
prerequisites: [ch-4-12]
---

# Reinforcement Learning for Robotics

Reinforcement learning (RL) enables robots to learn behaviors through trial and error.

## RL Fundamentals

### Markov Decision Process

```python
class MDP:
    """
    Markov Decision Process framework.
    """
    def __init__(self, states, actions, transition, reward, gamma=0.99):
        self.states = states
        self.actions = actions
        self.transition = transition  # P(s'|s,a)
        self.reward = reward  # R(s,a,s')
        self.gamma = gamma  # Discount factor
```

### Q-Learning

```python
import numpy as np

class QLearning:
    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.99, epsilon=0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])
        return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target += self.gamma * np.max(self.Q[next_state])

        self.Q[state, action] += self.lr * (target - self.Q[state, action])
```

## Policy Gradient Methods

### PPO (Proximal Policy Optimization)

```python
import torch
import torch.nn as nn

class PPOPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, obs):
        action_mean = self.actor(obs)
        value = self.critic(obs)
        return action_mean, value
```

## Sim-to-Real Transfer

### Domain Randomization

```python
def randomize_environment(env):
    """Randomize physics parameters for sim-to-real."""
    env.set_friction(np.random.uniform(0.5, 1.5))
    env.set_mass(np.random.uniform(0.8, 1.2) * env.default_mass)
    env.set_damping(np.random.uniform(0.8, 1.2) * env.default_damping)
    return env
```

## Summary

- RL learns from interaction with the environment
- Q-learning is fundamental value-based method
- PPO is a stable policy gradient algorithm
- Sim-to-real requires careful domain randomization

## Further Reading

- Sutton, R.S. & Barto, A.G. "Reinforcement Learning: An Introduction"
- Stable Baselines3 Documentation
