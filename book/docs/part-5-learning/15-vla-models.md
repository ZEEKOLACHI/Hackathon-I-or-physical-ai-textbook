---
id: ch-5-15
title: Vision-Language-Action Models
sidebar_position: 3
difficulty: advanced
estimated_time: 40
prerequisites: [ch-5-14]
---

# Vision-Language-Action Models

VLA models combine vision, language, and action for general-purpose robot control.

## Architecture Overview

```python
import torch
import torch.nn as nn
from transformers import AutoModel

class VLAModel(nn.Module):
    """
    Vision-Language-Action model architecture.
    """
    def __init__(self, vision_encoder, language_encoder, action_dim):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.language_encoder = language_encoder

        # Fusion network
        self.fusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8),
            num_layers=6
        )

        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, image, instruction):
        # Encode inputs
        vision_features = self.vision_encoder(image)
        language_features = self.language_encoder(instruction)

        # Fuse modalities
        combined = torch.cat([vision_features, language_features], dim=1)
        fused = self.fusion(combined)

        # Predict action
        action = self.action_head(fused[:, 0, :])
        return action
```

## Training Pipeline

```python
def train_vla(model, dataset, epochs=100):
    """Train VLA model on robot demonstrations."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        for batch in dataset:
            images = batch['images']
            instructions = batch['instructions']
            actions = batch['actions']

            pred_actions = model(images, instructions)
            loss = nn.MSELoss()(pred_actions, actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

## Language Conditioning

```python
class LanguageConditionedPolicy:
    def __init__(self, vla_model, tokenizer):
        self.model = vla_model
        self.tokenizer = tokenizer

    def act(self, image, instruction):
        # Tokenize instruction
        tokens = self.tokenizer(instruction, return_tensors='pt')

        # Get action
        with torch.no_grad():
            action = self.model(image, tokens)

        return action.numpy()

# Usage
policy = LanguageConditionedPolicy(model, tokenizer)
action = policy.act(camera_image, "Pick up the red cup")
```

## Key VLA Models

| Model | Features |
|-------|----------|
| RT-1 | Real-world manipulation, 700+ tasks |
| RT-2 | Web-scale pretraining, reasoning |
| PaLM-E | Embodied multimodal model |
| OpenVLA | Open-source VLA |

## Summary

- VLA models unify vision, language, and action
- Transformer architectures enable multimodal fusion
- Language conditioning enables task generalization
- Large-scale pretraining improves performance

## Further Reading

- Brohan et al. "RT-2: Vision-Language-Action Models"
- Driess et al. "PaLM-E: An Embodied Multimodal Language Model"
