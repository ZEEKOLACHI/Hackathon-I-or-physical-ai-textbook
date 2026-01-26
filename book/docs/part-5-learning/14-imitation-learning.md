---
id: ch-5-14
title: Imitation Learning
sidebar_position: 2
difficulty: advanced
estimated_time: 50
prerequisites: [ch-5-13]
---

# Imitation Learning

> "The best way to learn is to watch and imitate. For robots, as for children, demonstration is often more effective than instruction."
> — Pieter Abbeel, UC Berkeley

When teaching a child to tie their shoes, you don't write down equations—you show them. Imitation learning brings this intuitive approach to robotics, enabling robots to learn complex behaviors by observing human demonstrations rather than through tedious reward engineering or explicit programming.

## Why Imitation Learning?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHY IMITATION LEARNING?                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Problem with Reinforcement Learning:                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   Reward Engineering is HARD                                        │  │
│   │                                                                     │  │
│   │   Task: "Make the robot pour water into a glass"                   │  │
│   │                                                                     │  │
│   │   Reward attempt 1: +1 if water in glass                           │  │
│   │   → Robot throws water, some lands in glass ✗                      │  │
│   │                                                                     │  │
│   │   Reward attempt 2: +1 if water in glass, -1 if spilled            │  │
│   │   → Robot doesn't move (can't spill if you don't pour) ✗          │  │
│   │                                                                     │  │
│   │   Reward attempt 3: Shape reward based on bottle tilt...           │  │
│   │   → Getting complicated! And still might not work ✗                │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Solution: JUST SHOW THE ROBOT                                            │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   Human demonstrates pouring water                                  │  │
│   │                ▼                                                    │  │
│   │   Robot observes: states → actions                                 │  │
│   │                ▼                                                    │  │
│   │   Robot learns: π(a|s) that mimics human                          │  │
│   │                ▼                                                    │  │
│   │   Robot can now pour water! ✓                                      │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Advantages:                                                              │
│   • No reward engineering needed                                          │
│   • Leverages human expertise directly                                    │
│   • Often faster than RL from scratch                                     │
│   • Can capture subtle behaviors                                          │
│                                                                             │
│   Challenges:                                                              │
│   • Need good demonstrations                                              │
│   • Distribution shift (policy sees states demos didn't cover)           │
│   • Hard to exceed expert performance                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## The Imitation Learning Landscape

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    IMITATION LEARNING TAXONOMY                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                         Imitation Learning                                  │
│                               │                                             │
│              ┌────────────────┼────────────────┐                           │
│              ▼                ▼                ▼                           │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                  │
│   │  Behavioral  │   │  Interactive │   │   Inverse    │                  │
│   │   Cloning    │   │   Learning   │   │     RL       │                  │
│   │   (BC)       │   │   (DAgger)   │   │    (IRL)     │                  │
│   └──────────────┘   └──────────────┘   └──────────────┘                  │
│         │                   │                  │                           │
│         ▼                   ▼                  ▼                           │
│   Direct copying     Query expert        Learn reward                     │
│   of behavior        during learning     from demos                       │
│                                                                             │
│   Pros:              Pros:               Pros:                             │
│   • Simple           • Handles drift    • Transfers to                    │
│   • Fast             • Better coverage    new situations                  │
│   • Works offline    • Guaranteed        • Can exceed                      │
│                       convergence         expert                           │
│                                                                             │
│   Cons:              Cons:               Cons:                             │
│   • Distribution     • Needs online     • Computationally                 │
│     shift             expert              expensive                        │
│   • Compounds        • More complex     • Ambiguous                       │
│     errors                                solutions                        │
│                                                                             │
│   Use when:          Use when:           Use when:                         │
│   • Enough demos     • Expert available • Need to                         │
│   • Simple task      • Critical task     generalize                       │
│   • Quick baseline   • Safety matters   • Explain behavior                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Behavioral Cloning

Behavioral cloning (BC) is the simplest form of imitation learning: treat demonstrations as supervised learning data.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BEHAVIORAL CLONING                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Step 1: Collect Demonstrations                                           │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   Human operator demonstrates task                                  │  │
│   │                                                                     │  │
│   │   Record: D = {(s₁,a₁), (s₂,a₂), ..., (sₙ,aₙ)}                    │  │
│   │                                                                     │  │
│   │   Where:                                                           │  │
│   │   s = observation (camera, joint positions, etc.)                  │  │
│   │   a = action taken by human                                        │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Step 2: Train Policy                                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   Loss = Σ ||π_θ(sᵢ) - aᵢ||²                                       │  │
│   │          i                                                          │  │
│   │                                                                     │  │
│   │   "Minimize difference between predicted and demonstrated action"  │  │
│   │                                                                     │  │
│   │   Just supervised learning!                                        │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   The Problem: Distribution Shift                                          │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   Training: s ∈ D_expert     Testing: s ∈ D_policy                 │  │
│   │                                                                     │  │
│   │   Small error → different state → bigger error → ...              │  │
│   │                                                                     │  │
│   │   ┌─────────────────────────────────────────────────────┐          │  │
│   │   │    Expert trajectory                                │          │  │
│   │   │    ════════════════════════▶                        │          │  │
│   │   │              ╲                                      │          │  │
│   │   │               ╲ small error                         │          │  │
│   │   │                ╲                                    │          │  │
│   │   │                 ╲═══════════════▶ Policy diverges! │          │  │
│   │   └─────────────────────────────────────────────────────┘          │  │
│   │                                                                     │  │
│   │   Error compounds exponentially with trajectory length!            │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Behavioral Cloning Implementation

```python
"""
Behavioral Cloning (BC) for Robot Learning

BC treats imitation learning as supervised learning: given demonstrations
(state, action) pairs, learn a policy that maps states to actions.

Simple but powerful—often the first approach to try for a new task.

Key considerations:
1. Data quality matters more than quantity
2. Diverse demonstrations help with distribution shift
3. Action representation affects learning difficulty
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Demonstration:
    """A single demonstration trajectory."""
    states: np.ndarray      # Shape: (T, state_dim)
    actions: np.ndarray     # Shape: (T, action_dim)
    rewards: Optional[np.ndarray] = None  # Optional rewards


class DemonstrationDataset(Dataset):
    """
    PyTorch dataset for demonstration data.

    Handles:
    - Multiple demonstrations
    - State/action normalization
    - Data augmentation (optional)
    """

    def __init__(
        self,
        demonstrations: List[Demonstration],
        normalize: bool = True
    ):
        """
        Initialize dataset from demonstrations.

        Args:
            demonstrations: List of demonstration trajectories
            normalize: Whether to normalize states and actions
        """
        # Concatenate all demonstrations
        all_states = np.concatenate([d.states for d in demonstrations])
        all_actions = np.concatenate([d.actions for d in demonstrations])

        # Compute normalization statistics
        self.state_mean = all_states.mean(axis=0)
        self.state_std = all_states.std(axis=0) + 1e-8
        self.action_mean = all_actions.mean(axis=0)
        self.action_std = all_actions.std(axis=0) + 1e-8

        if normalize:
            all_states = (all_states - self.state_mean) / self.state_std
            all_actions = (all_actions - self.action_mean) / self.action_std

        self.states = torch.FloatTensor(all_states)
        self.actions = torch.FloatTensor(all_actions)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.states[idx], self.actions[idx]


class BCPolicy(nn.Module):
    """
    Behavioral Cloning policy network.

    Architecture choices:
    - MLP for proprioceptive observations
    - CNN + MLP for visual observations
    - Transformer for sequence modeling

    Output types:
    - Deterministic: Direct action prediction
    - Gaussian: Mean and variance (for continuous actions)
    - Mixture: Gaussian Mixture Model (for multimodal actions)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        output_type: str = "deterministic"
    ):
        """
        Initialize BC policy.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: Sizes of hidden layers
            activation: Activation function ("relu", "tanh", "gelu")
            output_type: "deterministic", "gaussian", or "mixture"
        """
        super().__init__()

        self.output_type = output_type
        self.action_dim = action_dim

        # Select activation
        activations = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "gelu": nn.GELU
        }
        act_fn = activations.get(activation, nn.ReLU)

        # Build encoder
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_fn())
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Output head(s)
        if output_type == "deterministic":
            self.action_head = nn.Linear(prev_dim, action_dim)
        elif output_type == "gaussian":
            self.mean_head = nn.Linear(prev_dim, action_dim)
            self.log_std_head = nn.Linear(prev_dim, action_dim)
        elif output_type == "mixture":
            n_components = 5
            self.n_components = n_components
            self.weight_head = nn.Linear(prev_dim, n_components)
            self.mean_head = nn.Linear(prev_dim, n_components * action_dim)
            self.log_std_head = nn.Linear(prev_dim, n_components * action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: State tensor (batch_size, state_dim)

        Returns:
            Action or action distribution parameters
        """
        features = self.encoder(state)

        if self.output_type == "deterministic":
            return self.action_head(features)
        elif self.output_type == "gaussian":
            mean = self.mean_head(features)
            log_std = self.log_std_head(features)
            return mean, log_std
        elif self.output_type == "mixture":
            weights = torch.softmax(self.weight_head(features), dim=-1)
            means = self.mean_head(features).view(-1, self.n_components, self.action_dim)
            log_stds = self.log_std_head(features).view(-1, self.n_components, self.action_dim)
            return weights, means, log_stds

    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = True
    ) -> torch.Tensor:
        """
        Get action from policy.

        Args:
            state: State tensor
            deterministic: Whether to use mean/mode

        Returns:
            Action tensor
        """
        if self.output_type == "deterministic":
            return self.forward(state)

        elif self.output_type == "gaussian":
            mean, log_std = self.forward(state)
            if deterministic:
                return mean
            else:
                std = torch.exp(log_std)
                return mean + std * torch.randn_like(std)

        elif self.output_type == "mixture":
            weights, means, log_stds = self.forward(state)
            if deterministic:
                # Return mean of highest-weight component
                idx = weights.argmax(dim=-1)
                batch_idx = torch.arange(means.size(0))
                return means[batch_idx, idx]
            else:
                # Sample from mixture
                component = torch.multinomial(weights, 1).squeeze(-1)
                batch_idx = torch.arange(means.size(0))
                mean = means[batch_idx, component]
                log_std = log_stds[batch_idx, component]
                std = torch.exp(log_std)
                return mean + std * torch.randn_like(std)


class BehavioralCloning:
    """
    Behavioral Cloning trainer.

    Handles:
    - Dataset management
    - Training loop
    - Evaluation
    - Checkpointing
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        hidden_dims: List[int] = [256, 256],
        output_type: str = "deterministic",
        device: str = "cpu"
    ):
        """
        Initialize BC trainer.

        Args:
            state_dim: State dimension
            action_dim: Action dimension
            lr: Learning rate
            weight_decay: L2 regularization
            hidden_dims: Hidden layer sizes
            output_type: Policy output type
            device: Torch device
        """
        self.device = torch.device(device)
        self.output_type = output_type

        # Create policy
        self.policy = BCPolicy(
            state_dim, action_dim, hidden_dims, output_type=output_type
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # Training statistics
        self.train_losses = []
        self.val_losses = []

    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute training loss.

        Args:
            states: Batch of states
            actions: Batch of expert actions

        Returns:
            Loss tensor
        """
        states = states.to(self.device)
        actions = actions.to(self.device)

        if self.output_type == "deterministic":
            pred_actions = self.policy(states)
            loss = nn.MSELoss()(pred_actions, actions)

        elif self.output_type == "gaussian":
            mean, log_std = self.policy(states)
            # Negative log likelihood
            var = torch.exp(2 * log_std)
            loss = 0.5 * (
                ((actions - mean) ** 2) / var + 2 * log_std
            ).sum(dim=-1).mean()

        elif self.output_type == "mixture":
            weights, means, log_stds = self.policy(states)
            # Mixture negative log likelihood
            actions_expanded = actions.unsqueeze(1)  # (B, 1, A)
            vars = torch.exp(2 * log_stds)  # (B, K, A)

            # Log probability for each component
            log_probs = -0.5 * (
                ((actions_expanded - means) ** 2) / vars + 2 * log_stds
            ).sum(dim=-1)  # (B, K)

            # Weighted sum (log-sum-exp for numerical stability)
            log_weights = torch.log(weights + 1e-8)
            loss = -torch.logsumexp(log_probs + log_weights, dim=-1).mean()

        return loss

    def train(
        self,
        demonstrations: List[Demonstration],
        n_epochs: int = 100,
        batch_size: int = 64,
        val_split: float = 0.1,
        early_stopping: int = 10
    ) -> dict:
        """
        Train BC policy on demonstrations.

        Args:
            demonstrations: List of demonstration trajectories
            n_epochs: Number of training epochs
            batch_size: Batch size
            val_split: Validation split ratio
            early_stopping: Patience for early stopping

        Returns:
            Training statistics
        """
        # Create dataset
        dataset = DemonstrationDataset(demonstrations)

        # Split into train/val
        n_val = int(len(dataset) * val_split)
        n_train = len(dataset) - n_val
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [n_train, n_val]
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(n_epochs):
            # Training
            self.policy.train()
            train_loss = 0

            for states, actions in train_loader:
                loss = self.compute_loss(states, actions)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)

            # Validation
            self.policy.eval()
            val_loss = 0

            with torch.no_grad():
                for states, actions in val_loader:
                    loss = self.compute_loss(states, actions)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            self.val_losses.append(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_weights = self.policy.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= early_stopping:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        # Restore best weights
        self.policy.load_state_dict(best_weights)

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss
        }

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """Get action from trained policy."""
        self.policy.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.policy.get_action(state_tensor, deterministic=True)
        return action.cpu().numpy().squeeze()


# Example usage
if __name__ == "__main__":
    print("Behavioral Cloning Demo")
    print("=" * 50)

    # Generate synthetic demonstrations
    state_dim = 10
    action_dim = 4
    n_demos = 50
    demo_length = 100

    demonstrations = []
    for _ in range(n_demos):
        # Simulate expert behavior (linear policy + noise)
        expert_weights = np.random.randn(action_dim, state_dim)
        states = np.random.randn(demo_length, state_dim)
        actions = states @ expert_weights.T + 0.1 * np.random.randn(demo_length, action_dim)

        demonstrations.append(Demonstration(states=states, actions=actions))

    print(f"Created {len(demonstrations)} demonstrations")
    print(f"Total transitions: {sum(len(d.states) for d in demonstrations)}")

    # Train BC policy
    bc = BehavioralCloning(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=1e-3,
        hidden_dims=[256, 256],
        output_type="gaussian"
    )

    stats = bc.train(demonstrations, n_epochs=50, batch_size=64)

    print(f"\nTraining complete!")
    print(f"Best validation loss: {stats['best_val_loss']:.4f}")

    # Test policy
    test_state = np.random.randn(state_dim)
    action = bc.get_action(test_state)
    print(f"\nTest state shape: {test_state.shape}")
    print(f"Predicted action: {action}")
```

## DAgger: Dataset Aggregation

DAgger addresses distribution shift by iteratively collecting data from the learned policy and querying the expert for corrections.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DAgger ALGORITHM                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Key Insight: Train on states the POLICY visits, not just expert states  │
│                                                                             │
│   Algorithm:                                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   1. Initialize D with expert demonstrations                        │  │
│   │   2. Train initial policy π₁ on D                                  │  │
│   │                                                                     │  │
│   │   For i = 1, 2, ... :                                              │  │
│   │     3. Execute πᵢ to collect trajectory                            │  │
│   │     4. For each state s in trajectory:                             │  │
│   │        Query expert for action a* = π*(s)                          │  │
│   │        Add (s, a*) to D                                            │  │
│   │     5. Train πᵢ₊₁ on aggregated D                                  │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Visual:                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   Iteration 1:                                                     │  │
│   │   Expert demo:  ●───●───●───●───●                                  │  │
│   │   Policy exec:  ●───○───○───○───○  (different states!)            │  │
│   │   Query expert: ●───●───●───●───●  (labels for policy states)     │  │
│   │                                                                     │  │
│   │   Iteration 2:                                                     │  │
│   │   Policy now trained on BOTH expert and policy states             │  │
│   │   → Less distribution shift!                                       │  │
│   │                                                                     │  │
│   │   Eventually: Policy visits same states it was trained on         │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Guarantee: O(log T) regret bound (vs O(T²) for BC)                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### DAgger Implementation

```python
"""
DAgger (Dataset Aggregation) for Interactive Imitation Learning

DAgger solves the distribution shift problem by iteratively:
1. Running the learned policy
2. Querying the expert for labels on visited states
3. Aggregating into the training dataset

This ensures the policy is trained on states it actually visits.

Reference: Ross et al. (2011) "A Reduction of Imitation Learning and
           Structured Prediction to No-Regret Online Learning"
"""

import numpy as np
import torch
from typing import List, Callable, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


class Expert(ABC):
    """Abstract expert interface."""

    @abstractmethod
    def get_action(self, state: np.ndarray) -> np.ndarray:
        """Return expert action for given state."""
        pass


class DAgger:
    """
    DAgger: Dataset Aggregation for Imitation Learning.

    Iteratively:
    1. Execute current policy
    2. Query expert for correct actions
    3. Add to dataset and retrain

    Block Diagram:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                                                                     │
    │   ┌──────────┐     Execute      ┌──────────┐                       │
    │   │  Policy  │ ───────────────▶ │  States  │                       │
    │   │   πᵢ     │                  │   Sᵢ     │                       │
    │   └────▲─────┘                  └────┬─────┘                       │
    │        │                              │                            │
    │        │ Train                        │ Query                      │
    │        │                              ▼                            │
    │   ┌────┴─────┐                  ┌──────────┐                       │
    │   │ Dataset  │ ◀─── Aggregate ──│  Expert  │                       │
    │   │    D     │      (Sᵢ, Aᵢ*)   │   π*     │                       │
    │   └──────────┘                  └──────────┘                       │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        policy,
        expert: Expert,
        state_dim: int,
        action_dim: int,
        beta_schedule: str = "linear"
    ):
        """
        Initialize DAgger.

        Args:
            policy: Learnable policy (with train and get_action methods)
            expert: Expert to query
            state_dim: State dimension
            action_dim: Action dimension
            beta_schedule: How to mix policy/expert ("linear", "constant", "exponential")
        """
        self.policy = policy
        self.expert = expert
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.beta_schedule = beta_schedule

        # Aggregated dataset
        self.all_states = []
        self.all_actions = []

    def get_beta(self, iteration: int, max_iterations: int) -> float:
        """
        Get mixing coefficient beta.

        beta = probability of using expert action during rollout
        Higher beta = more expert (safer but less coverage)
        Lower beta = more policy (better coverage but may fail)

        Args:
            iteration: Current iteration
            max_iterations: Total iterations

        Returns:
            Beta value in [0, 1]
        """
        if self.beta_schedule == "constant":
            return 0.5
        elif self.beta_schedule == "linear":
            return max(0, 1 - iteration / max_iterations)
        elif self.beta_schedule == "exponential":
            return 0.5 ** iteration
        else:
            return 0.5

    def collect_rollout(
        self,
        env,
        beta: float,
        max_steps: int = 1000
    ) -> Tuple[List[np.ndarray], List[np.ndarray], float]:
        """
        Collect rollout with expert/policy mixture.

        Args:
            env: Environment
            beta: Probability of using expert
            max_steps: Maximum steps per episode

        Returns:
            Tuple of (states, expert_actions, total_reward)
        """
        states = []
        expert_actions = []
        total_reward = 0

        state = env.reset()

        for step in range(max_steps):
            states.append(state)

            # Always query expert for label
            expert_action = self.expert.get_action(state)
            expert_actions.append(expert_action)

            # Choose action based on beta
            if np.random.random() < beta:
                action = expert_action
            else:
                action = self.policy.get_action(state)

            # Step environment
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            state = next_state

            if done:
                break

        return states, expert_actions, total_reward

    def train(
        self,
        env,
        n_iterations: int = 100,
        rollouts_per_iter: int = 10,
        epochs_per_iter: int = 10,
        initial_demonstrations: Optional[List] = None
    ) -> dict:
        """
        Train policy using DAgger.

        Args:
            env: Environment
            n_iterations: Number of DAgger iterations
            rollouts_per_iter: Rollouts to collect each iteration
            epochs_per_iter: Training epochs per iteration
            initial_demonstrations: Optional initial expert demos

        Returns:
            Training statistics
        """
        stats = {
            'rewards': [],
            'dataset_size': [],
            'policy_rewards': []
        }

        # Initialize with demonstrations if provided
        if initial_demonstrations:
            for demo in initial_demonstrations:
                self.all_states.extend(demo.states.tolist())
                self.all_actions.extend(demo.actions.tolist())

        print("DAgger Training")
        print("=" * 50)

        for iteration in range(n_iterations):
            beta = self.get_beta(iteration, n_iterations)

            # Collect rollouts
            iter_states = []
            iter_actions = []
            iter_rewards = []

            for _ in range(rollouts_per_iter):
                states, actions, reward = self.collect_rollout(env, beta)
                iter_states.extend(states)
                iter_actions.extend(actions)
                iter_rewards.append(reward)

            # Aggregate data
            self.all_states.extend(iter_states)
            self.all_actions.extend(iter_actions)

            # Train policy on aggregated dataset
            states_arr = np.array(self.all_states)
            actions_arr = np.array(self.all_actions)

            # Create demonstration for training
            demo = type('Demonstration', (), {
                'states': states_arr,
                'actions': actions_arr
            })()

            self.policy.train([demo], n_epochs=epochs_per_iter)

            # Evaluate policy (no expert)
            policy_reward = self.evaluate_policy(env)

            # Record stats
            avg_reward = np.mean(iter_rewards)
            stats['rewards'].append(avg_reward)
            stats['dataset_size'].append(len(self.all_states))
            stats['policy_rewards'].append(policy_reward)

            print(f"Iter {iteration + 1}: β={beta:.2f}, "
                  f"Reward={avg_reward:.2f}, "
                  f"Policy Reward={policy_reward:.2f}, "
                  f"Dataset={len(self.all_states)}")

        return stats

    def evaluate_policy(self, env, n_episodes: int = 5) -> float:
        """Evaluate policy without expert."""
        total_reward = 0

        for _ in range(n_episodes):
            state = env.reset()
            episode_reward = 0

            for _ in range(1000):
                action = self.policy.get_action(state)
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                state = next_state
                if done:
                    break

            total_reward += episode_reward

        return total_reward / n_episodes


class HGDAgger(DAgger):
    """
    Human-Gated DAgger (HG-DAgger).

    Instead of always querying expert, allows human to intervene
    only when they think the robot is about to fail.

    More practical for real-world deployment where constant
    expert queries are expensive.

    Reference: Kelly et al. (2019) "HG-DAgger: Interactive Imitation
               Learning with Human Experts"
    """

    def __init__(self, *args, intervention_threshold: float = 0.5, **kwargs):
        """
        Initialize HG-DAgger.

        Args:
            intervention_threshold: Uncertainty threshold for intervention
        """
        super().__init__(*args, **kwargs)
        self.intervention_threshold = intervention_threshold

    def should_intervene(self, state: np.ndarray) -> bool:
        """
        Determine if expert should intervene.

        In practice, this could be:
        - Human judgment (pressing button)
        - Uncertainty estimation from policy
        - Anomaly detection
        """
        # Example: Check if policy is uncertain (for Gaussian policy)
        if hasattr(self.policy.policy, 'output_type') and \
           self.policy.policy.output_type == 'gaussian':
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                _, log_std = self.policy.policy(state_tensor)
                uncertainty = log_std.exp().mean().item()
                return uncertainty > self.intervention_threshold

        # Default: Random intervention (for testing)
        return np.random.random() < 0.3

    def collect_rollout(
        self,
        env,
        beta: float,
        max_steps: int = 1000
    ) -> Tuple[List[np.ndarray], List[np.ndarray], float]:
        """Collect rollout with intervention-based expert queries."""
        states = []
        expert_actions = []
        total_reward = 0
        interventions = 0

        state = env.reset()

        for step in range(max_steps):
            # Check if intervention needed
            if self.should_intervene(state):
                action = self.expert.get_action(state)
                states.append(state)
                expert_actions.append(action)
                interventions += 1
            else:
                action = self.policy.get_action(state)

            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state

            if done:
                break

        # Return info about interventions
        return states, expert_actions, total_reward


# Example usage
if __name__ == "__main__":
    print("DAgger Demo")
    print("=" * 50)

    # Simple expert (linear policy)
    class LinearExpert(Expert):
        def __init__(self, state_dim, action_dim):
            self.weights = np.random.randn(action_dim, state_dim) * 0.5

        def get_action(self, state):
            return self.weights @ state

    # Simple environment
    class SimpleEnv:
        def __init__(self, state_dim=10, action_dim=4):
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.state = None

        def reset(self):
            self.state = np.random.randn(self.state_dim)
            return self.state

        def step(self, action):
            # Simple dynamics: state updates based on action
            self.state = self.state + 0.1 * np.random.randn(self.state_dim)
            reward = -np.linalg.norm(action)
            done = np.random.random() < 0.01  # Random termination
            return self.state, reward, done, {}

    state_dim = 10
    action_dim = 4

    expert = LinearExpert(state_dim, action_dim)
    env = SimpleEnv(state_dim, action_dim)

    # Create BC policy for DAgger
    bc = BehavioralCloning(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[64, 64]
    )

    # Create DAgger trainer
    dagger = DAgger(
        policy=bc,
        expert=expert,
        state_dim=state_dim,
        action_dim=action_dim,
        beta_schedule="linear"
    )

    print("\nRunning DAgger for 5 iterations...")
    stats = dagger.train(
        env=env,
        n_iterations=5,
        rollouts_per_iter=3,
        epochs_per_iter=5
    )

    print(f"\nFinal dataset size: {stats['dataset_size'][-1]}")
```

## Inverse Reinforcement Learning

IRL learns the underlying reward function from demonstrations, rather than just mimicking actions.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    INVERSE REINFORCEMENT LEARNING                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Standard RL:    Reward R  →  Optimal Policy π*                           │
│                                                                             │
│   Inverse RL:     Demonstrations D  →  Reward R  →  Policy π*             │
│                                                                             │
│   Why learn rewards?                                                       │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   1. Transfer: Reward transfers better than policy                  │  │
│   │      • Same reward works in different environments                 │  │
│   │      • Policy may need retraining                                  │  │
│   │                                                                     │  │
│   │   2. Interpretability: Understand WHAT the expert optimizes        │  │
│   │      • "Expert prefers shorter paths"                              │  │
│   │      • "Expert avoids obstacles more than delays"                  │  │
│   │                                                                     │  │
│   │   3. Generalization: Can exceed expert performance                 │  │
│   │      • Expert may be suboptimal                                    │  │
│   │      • RL with learned reward finds better policy                  │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   The IRL Problem:                                                         │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   Given:    Expert demonstrations D = {τ₁, τ₂, ..., τₙ}            │  │
│   │                                                                     │  │
│   │   Find:     Reward function R(s,a) such that                       │  │
│   │             expert policy is optimal under R                        │  │
│   │                                                                     │  │
│   │   Challenge: INFINITELY MANY rewards explain any behavior!         │  │
│   │             • R = 0 makes everything optimal                       │  │
│   │             • Need additional constraints                          │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Approaches:                                                              │
│   • Maximum Entropy IRL: Prefer simplest explanations                     │
│   • GAIL: Adversarial approach (discriminator as reward)                  │
│   • AIRL: Disentangles dynamics from reward                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Maximum Entropy IRL Implementation

```python
"""
Maximum Entropy Inverse Reinforcement Learning

MaxEnt IRL assumes experts act to maximize expected reward while
maintaining maximum entropy (randomness). This resolves the ambiguity
of IRL by preferring the simplest reward function.

Key idea: Among all rewards that make expert behavior optimal,
choose the one that makes expert behavior MOST LIKELY.

Reference: Ziebart et al. (2008) "Maximum Entropy Inverse
           Reinforcement Learning"
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Optional


class RewardNetwork(nn.Module):
    """
    Neural network for reward function approximation.

    R(s,a) = f_θ(s,a) or R(s) = f_θ(s)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 0,
        hidden_dims: List[int] = [64, 64],
        state_only: bool = True
    ):
        """
        Initialize reward network.

        Args:
            state_dim: State dimension
            action_dim: Action dimension (0 if state-only reward)
            hidden_dims: Hidden layer sizes
            state_only: Whether reward depends only on state
        """
        super().__init__()

        self.state_only = state_only
        input_dim = state_dim if state_only else state_dim + action_dim

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute reward.

        Args:
            state: State tensor
            action: Action tensor (optional if state_only)

        Returns:
            Reward value
        """
        if self.state_only:
            return self.network(state)
        else:
            x = torch.cat([state, action], dim=-1)
            return self.network(x)


class MaxEntIRL:
    """
    Maximum Entropy Inverse Reinforcement Learning.

    Learns reward function that makes expert demonstrations
    most likely under the maximum entropy RL objective:

    π*(a|s) ∝ exp(Q(s,a))

    Training:
    1. Compute expected features under current reward
    2. Compare with expert feature expectations
    3. Update reward to match
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [64, 64],
        lr: float = 1e-3,
        device: str = "cpu"
    ):
        """
        Initialize MaxEnt IRL.

        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dims: Reward network hidden layers
            lr: Learning rate
            device: Torch device
        """
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Reward network
        self.reward_net = RewardNetwork(
            state_dim, action_dim, hidden_dims, state_only=True
        ).to(self.device)

        self.optimizer = optim.Adam(self.reward_net.parameters(), lr=lr)

    def compute_expert_features(
        self,
        demonstrations: List[np.ndarray]
    ) -> torch.Tensor:
        """
        Compute average state features from demonstrations.

        In linear IRL, we match feature expectations.
        With neural networks, we use state visitation frequencies.

        Args:
            demonstrations: List of state trajectories

        Returns:
            Average state visitation (as tensor)
        """
        all_states = np.concatenate(demonstrations)
        return torch.FloatTensor(all_states.mean(axis=0)).to(self.device)

    def compute_policy_features(
        self,
        env,
        policy,
        n_trajectories: int = 50,
        max_steps: int = 200
    ) -> torch.Tensor:
        """
        Compute average state features from policy rollouts.

        Args:
            env: Environment
            policy: Current policy
            n_trajectories: Number of rollouts
            max_steps: Max steps per rollout

        Returns:
            Average state visitation
        """
        all_states = []

        for _ in range(n_trajectories):
            state = env.reset()

            for _ in range(max_steps):
                all_states.append(state)
                action = policy.get_action(state)
                next_state, _, done, _ = env.step(action)
                state = next_state
                if done:
                    break

        return torch.FloatTensor(np.array(all_states).mean(axis=0)).to(self.device)

    def train_step(
        self,
        expert_features: torch.Tensor,
        policy_features: torch.Tensor
    ) -> float:
        """
        Single training step for reward function.

        The gradient is the difference between expert and policy
        feature expectations:
        ∇L = μ_expert - μ_policy

        Args:
            expert_features: Expert state feature expectations
            policy_features: Policy state feature expectations

        Returns:
            Loss value
        """
        # Compute rewards for feature matching
        expert_reward = self.reward_net(expert_features.unsqueeze(0))
        policy_reward = self.reward_net(policy_features.unsqueeze(0))

        # MaxEnt IRL loss: maximize expert reward, minimize policy reward
        loss = -(expert_reward - policy_reward).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def get_reward(self, state: np.ndarray) -> float:
        """Get reward for a state."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            reward = self.reward_net(state_tensor)
        return reward.item()


class GAIL:
    """
    Generative Adversarial Imitation Learning (GAIL).

    Uses adversarial training:
    - Discriminator: Classifies expert vs policy transitions
    - Generator (Policy): Tries to fool discriminator

    The discriminator serves as a learned reward signal.

    Reference: Ho & Ermon (2016) "Generative Adversarial Imitation Learning"

    Block Diagram:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                                                                     │
    │   Expert demos ────┐                                               │
    │                    ▼                                               │
    │              ┌──────────────┐                                      │
    │              │ Discriminator│◀─── Policy rollouts                  │
    │              │  D(s, a)     │                                      │
    │              └──────┬───────┘                                      │
    │                     │                                              │
    │                     │ Reward = -log(D)                            │
    │                     ▼                                              │
    │              ┌──────────────┐                                      │
    │              │    Policy    │───▶ Actions                          │
    │              │   (PPO/TRPO) │                                      │
    │              └──────────────┘                                      │
    │                                                                     │
    │   Discriminator goal: D(expert) → 1, D(policy) → 0                │
    │   Policy goal: Make D(policy) → 1                                 │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        lr_discriminator: float = 3e-4,
        lr_policy: float = 3e-4,
        device: str = "cpu"
    ):
        """
        Initialize GAIL.

        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dims: Hidden layer sizes
            lr_discriminator: Discriminator learning rate
            lr_policy: Policy learning rate
            device: Torch device
        """
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Discriminator: classifies (s,a) as expert or policy
        self.discriminator = self._build_discriminator(
            state_dim + action_dim, hidden_dims
        ).to(self.device)

        self.disc_optimizer = optim.Adam(
            self.discriminator.parameters(), lr=lr_discriminator
        )

        # Policy would be initialized here (e.g., PPO)
        # For simplicity, we focus on discriminator

    def _build_discriminator(
        self,
        input_dim: int,
        hidden_dims: List[int]
    ) -> nn.Module:
        """Build discriminator network."""
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    def update_discriminator(
        self,
        expert_states: np.ndarray,
        expert_actions: np.ndarray,
        policy_states: np.ndarray,
        policy_actions: np.ndarray
    ) -> float:
        """
        Update discriminator to distinguish expert from policy.

        Args:
            expert_states: Expert state samples
            expert_actions: Expert action samples
            policy_states: Policy state samples
            policy_actions: Policy action samples

        Returns:
            Discriminator loss
        """
        # Convert to tensors
        expert_sa = torch.FloatTensor(
            np.concatenate([expert_states, expert_actions], axis=1)
        ).to(self.device)
        policy_sa = torch.FloatTensor(
            np.concatenate([policy_states, policy_actions], axis=1)
        ).to(self.device)

        # Discriminator predictions
        expert_pred = self.discriminator(expert_sa)
        policy_pred = self.discriminator(policy_sa)

        # Binary cross-entropy loss
        expert_loss = -torch.log(expert_pred + 1e-8).mean()
        policy_loss = -torch.log(1 - policy_pred + 1e-8).mean()
        loss = expert_loss + policy_loss

        self.disc_optimizer.zero_grad()
        loss.backward()
        self.disc_optimizer.step()

        return loss.item()

    def get_reward(
        self,
        state: np.ndarray,
        action: np.ndarray
    ) -> float:
        """
        Get reward from discriminator.

        Reward = -log(1 - D(s,a))

        This encourages policy to look like expert.
        """
        with torch.no_grad():
            sa = torch.FloatTensor(
                np.concatenate([state, action])
            ).unsqueeze(0).to(self.device)
            d = self.discriminator(sa)
            # Reward for looking like expert
            reward = -torch.log(1 - d + 1e-8).item()
        return reward


# Example usage
if __name__ == "__main__":
    print("Inverse Reinforcement Learning Demo")
    print("=" * 50)

    state_dim = 10
    action_dim = 4

    # Create IRL learner
    irl = MaxEntIRL(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[64, 64]
    )

    # Synthetic expert demonstrations (reaching goal state)
    goal_state = np.ones(state_dim) * 0.5
    demonstrations = []

    for _ in range(20):
        trajectory = []
        state = np.random.randn(state_dim)
        for step in range(50):
            # Expert moves toward goal
            state = state + 0.1 * (goal_state - state) + 0.01 * np.random.randn(state_dim)
            trajectory.append(state)
        demonstrations.append(np.array(trajectory))

    print(f"Collected {len(demonstrations)} demonstrations")

    # Compute expert features
    expert_features = irl.compute_expert_features(demonstrations)
    print(f"Expert feature mean: {expert_features.mean().item():.3f}")

    # Get reward for goal vs random state
    goal_reward = irl.get_reward(goal_state)
    random_reward = irl.get_reward(np.random.randn(state_dim))

    print(f"\nReward at goal state: {goal_reward:.3f}")
    print(f"Reward at random state: {random_reward:.3f}")
```

## Industry Perspective: Imitation Learning in Practice

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    IMITATION LEARNING IN INDUSTRY                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   GOOGLE DEEPMIND ROBOTICS                                                 │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │ • RT-1/RT-2: Large-scale behavior cloning from 130K demos      │      │
│   │ • Transformer-based policies for multi-task learning           │      │
│   │ • Language-conditioned imitation learning                      │      │
│   │ • Data collection at scale with multiple robots                │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│   TESLA (FSD / Optimus)                                                    │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │ • Massive-scale imitation from human driving                    │      │
│   │ • Shadow mode: Collect data without taking control             │      │
│   │ • Neural network learns from millions of miles                 │      │
│   │ • Similar approach being applied to Optimus humanoid           │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│   FIGURE AI                                                                │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │ • Teleoperation for demonstration collection                    │      │
│   │ • Imitation learning for manipulation tasks                     │      │
│   │ • Combining IL with LLM-based task planning                    │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│   MOBILE ALOHA (Stanford)                                                  │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │ • Action Chunking with Transformers (ACT)                       │      │
│   │ • Bimanual manipulation from teleoperation                      │      │
│   │ • Low-cost data collection setup                                │      │
│   │ • Open-source implementation                                    │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│   Best Practices:                                                          │
│   • Collect diverse demonstrations (not just successful ones)             │
│   • Use multiple operators to capture different strategies                │
│   • Combine IL with RL fine-tuning for best results                       │
│   • Careful attention to action representation (delta vs absolute)        │
│   • Data augmentation helps significantly                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Summary

### Key Takeaways

1. **Behavioral Cloning**: Simple supervised learning from demos, but suffers from distribution shift

2. **DAgger**: Solves distribution shift by querying expert on policy-visited states

3. **Inverse RL**: Learns underlying reward function, enabling generalization and transfer

4. **GAIL**: Uses adversarial training to learn policies that are indistinguishable from expert

5. **Scale Matters**: Large-scale demonstration collection is key to modern robot IL

### Method Comparison

| Method | Expert Queries | Sample Efficiency | Can Exceed Expert | Distribution Shift |
|--------|---------------|-------------------|-------------------|-------------------|
| BC | Offline only | High | No | High |
| DAgger | Online | Medium | No | Low |
| IRL | Offline only | Low | Yes | Low |
| GAIL | Offline only | Medium | Yes | Low |
| BC + RL | Offline + Env | Medium | Yes | Low |

### Practical Checklist

- [ ] Collected diverse demonstrations from multiple operators
- [ ] Verified demonstration quality and consistency
- [ ] Chose appropriate action representation (delta vs absolute)
- [ ] Implemented proper data normalization
- [ ] Applied data augmentation where applicable
- [ ] Set up evaluation metrics (success rate, tracking error)
- [ ] Considered combining IL with RL fine-tuning
- [ ] Tested on held-out scenarios

## Further Reading

### Foundational Works
- Argall et al. "A Survey of Robot Learning from Demonstration" (2009)
- Ross et al. "A Reduction of Imitation Learning to No-Regret Online Learning" (2011)
- Ho & Ermon "Generative Adversarial Imitation Learning" (2016)

### Modern Approaches
- Brohan et al. "RT-1: Robotics Transformer" (2022)
- Zhao et al. "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" (2023)

### Software
- robomimic: Imitation learning framework
- ACT: Action Chunking with Transformers
- imitation: Library for IRL and IL algorithms

---

*"Teaching by showing is as old as humanity itself. Imitation learning is simply formalizing what we've always known—that watching and copying is one of the most powerful ways to learn."*
