---
id: ch-5-14
title: Imitation Learning
sidebar_position: 2
difficulty: advanced
estimated_time: 35
prerequisites: [ch-5-13]
---

# Imitation Learning

Imitation learning enables robots to learn from human demonstrations.

## Behavioral Cloning

```python
import torch
import torch.nn as nn

class BehavioralCloning:
    def __init__(self, obs_dim, action_dim):
        self.policy = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)

    def train(self, demonstrations):
        """Train policy to mimic demonstrations."""
        for obs, action in demonstrations:
            pred_action = self.policy(obs)
            loss = nn.MSELoss()(pred_action, action)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
```

## DAgger (Dataset Aggregation)

```python
class DAgger:
    def __init__(self, policy, expert):
        self.policy = policy
        self.expert = expert
        self.dataset = []

    def train(self, env, iterations=100):
        for i in range(iterations):
            obs = env.reset()
            trajectory = []

            while not done:
                # Use mixture of policy and expert
                if np.random.random() < 0.5:
                    action = self.policy.predict(obs)
                else:
                    action = self.expert.predict(obs)

                # Get expert label
                expert_action = self.expert.predict(obs)
                trajectory.append((obs, expert_action))

                obs, _, done, _ = env.step(action)

            self.dataset.extend(trajectory)
            self.policy.train(self.dataset)
```

## Inverse Reinforcement Learning

```python
def inverse_rl(demonstrations, env, n_iterations=100):
    """Learn reward function from demonstrations."""
    reward_net = RewardNetwork(env.obs_dim)

    for _ in range(n_iterations):
        # Train policy with current reward
        policy = train_policy(env, reward_net)

        # Update reward to distinguish expert from policy
        expert_features = compute_features(demonstrations)
        policy_features = compute_features(policy.rollout(env))

        reward_net.update(expert_features - policy_features)

    return reward_net
```

## Summary

- Behavioral cloning directly mimics demonstrations
- DAgger improves by querying expert during execution
- IRL recovers the underlying reward function
- Combined approaches achieve robust learning

## Further Reading

- Argall, B. "A Survey of Robot Learning from Demonstration"
