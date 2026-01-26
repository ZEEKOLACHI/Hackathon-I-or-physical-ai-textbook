---
id: ch-5-15
title: Vision-Language-Action Models
sidebar_position: 3
difficulty: advanced
estimated_time: 55
prerequisites: [ch-5-14]
---

# Vision-Language-Action Models

> "The future of robotics lies not in programming specific behaviors, but in teaching robots to understand and execute natural language instructions in the physical world."
> — Sergey Levine, UC Berkeley

Imagine telling a robot "pick up the coffee mug and place it next to the laptop" and having it understand not just the words, but the visual scene, the objects involved, and the precise motor actions needed. Vision-Language-Action (VLA) models represent the frontier of robot learning, combining the visual understanding of computer vision, the reasoning capabilities of large language models, and the physical action generation needed for real-world manipulation.

## The VLA Revolution

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    THE VLA PARADIGM SHIFT                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Traditional Approach:                                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   Perception ───▶ Planning ───▶ Control                            │  │
│   │      │               │              │                               │  │
│   │      ▼               ▼              ▼                               │  │
│   │   Object         Task         Motor                                 │  │
│   │   Detection     Planner      Commands                              │  │
│   │                                                                     │  │
│   │   Problems:                                                        │  │
│   │   • Brittle pipelines                                              │  │
│   │   • Error propagation                                              │  │
│   │   • Hand-designed interfaces                                       │  │
│   │   • Limited generalization                                         │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   VLA Approach:                                                            │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │        Image ───────┐                                              │  │
│   │                     │                                               │  │
│   │                     ▼                                               │  │
│   │               ┌──────────┐                                         │  │
│   │        Text ─▶│   VLA    │───▶ Actions                             │  │
│   │               │  Model   │                                         │  │
│   │               └──────────┘                                         │  │
│   │                     ▲                                               │  │
│   │                     │                                               │  │
│   │        State ───────┘                                              │  │
│   │                                                                     │  │
│   │   Benefits:                                                        │  │
│   │   • End-to-end learning                                            │  │
│   │   • Natural language interface                                     │  │
│   │   • Transfer from web-scale data                                   │  │
│   │   • Strong generalization                                          │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Key Insight: Treat robot control as a sequence modeling problem         │
│   • Input: Images + Language instruction                                  │
│   • Output: Action sequence (as tokens)                                   │
│   • Model: Transformer-based multimodal architecture                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## VLA Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    VLA MODEL ARCHITECTURE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                                                                             │
│   ┌────────────────────────────────────────────────────────────────────┐   │
│   │                        VLA Model                                   │   │
│   │                                                                    │   │
│   │   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐         │   │
│   │   │   Vision     │   │   Language   │   │  Proprio-    │         │   │
│   │   │   Encoder    │   │   Encoder    │   │  ception     │         │   │
│   │   │   (ViT)      │   │   (LLM)      │   │  Encoder     │         │   │
│   │   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘         │   │
│   │          │                  │                   │                 │   │
│   │          ▼                  ▼                   ▼                 │   │
│   │   ┌──────────────────────────────────────────────────────┐       │   │
│   │   │               Multimodal Fusion                       │       │   │
│   │   │    (Cross-attention / Concatenation / Perceiver)     │       │   │
│   │   └────────────────────────┬─────────────────────────────┘       │   │
│   │                            │                                      │   │
│   │                            ▼                                      │   │
│   │   ┌──────────────────────────────────────────────────────┐       │   │
│   │   │             Transformer Backbone                      │       │   │
│   │   │          (Decoder / Encoder-Decoder)                 │       │   │
│   │   └────────────────────────┬─────────────────────────────┘       │   │
│   │                            │                                      │   │
│   │              ┌─────────────┼─────────────┐                       │   │
│   │              ▼             ▼             ▼                       │   │
│   │   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │   │
│   │   │  Action      │ │  Action      │ │  Gripper     │            │   │
│   │   │  Position    │ │  Rotation    │ │  Open/Close  │            │   │
│   │   │  (Δx,Δy,Δz)  │ │  (Δr,Δp,Δy)  │ │  (0 or 1)    │            │   │
│   │   └──────────────┘ └──────────────┘ └──────────────┘            │   │
│   │                                                                    │   │
│   └────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   Action Representation Options:                                           │
│   • Continuous: Direct regression of action values                        │
│   • Discrete: Tokenize actions (like RT-2)                                │
│   • Diffusion: Generate actions via denoising                             │
│   • Action Chunking: Predict multiple timesteps                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core VLA Implementation

```python
"""
Vision-Language-Action (VLA) Model for Robot Control

VLA models enable robots to follow natural language instructions by
combining visual understanding with language comprehension to generate
appropriate motor actions.

Key components:
1. Vision Encoder: Processes camera images
2. Language Encoder: Understands instructions
3. Fusion Module: Combines modalities
4. Action Decoder: Generates robot actions

Reference: Brohan et al. "RT-2: Vision-Language-Action Models" (2023)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class VLAInput:
    """Input to VLA model."""
    image: torch.Tensor           # (B, C, H, W) camera image
    instruction: List[str]        # Natural language instructions
    proprio: Optional[torch.Tensor] = None  # (B, D) proprioceptive state


@dataclass
class VLAOutput:
    """Output from VLA model."""
    action: torch.Tensor          # (B, action_dim) or (B, T, action_dim)
    action_logits: Optional[torch.Tensor] = None  # For discrete actions
    hidden_states: Optional[torch.Tensor] = None  # For analysis


class VisionEncoder(nn.Module):
    """
    Vision encoder using Vision Transformer (ViT).

    Converts images into a sequence of visual tokens that can be
    processed by the transformer backbone.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 14,
        embed_dim: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        pretrained: bool = True
    ):
        """
        Initialize vision encoder.

        Args:
            image_size: Input image size
            patch_size: Size of image patches
            embed_dim: Embedding dimension
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
            pretrained: Whether to use pretrained weights
        """
        super().__init__()

        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            3, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        # Position embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.n_patches + 1, embed_dim) * 0.02
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=n_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image into visual tokens.

        Args:
            x: Image tensor (B, 3, H, W)

        Returns:
            Visual tokens (B, n_patches + 1, embed_dim)
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, embed_dim)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, n_patches + 1, embed_dim)

        # Add position embedding
        x = x + self.pos_embed

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        return x


class LanguageEncoder(nn.Module):
    """
    Language encoder for processing instructions.

    Uses a transformer-based architecture similar to BERT/RoBERTa
    or can load pretrained language models.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        embed_dim: int = 768,
        n_layers: int = 6,
        n_heads: int = 12,
        max_length: int = 77
    ):
        """
        Initialize language encoder.

        Args:
            vocab_size: Vocabulary size
            embed_dim: Embedding dimension
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
            max_length: Maximum sequence length
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.max_length = max_length

        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, embed_dim)

        # Position embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_length, embed_dim) * 0.02
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=n_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode text into language tokens.

        Args:
            input_ids: Token IDs (B, L)
            attention_mask: Attention mask (B, L)

        Returns:
            Language tokens (B, L, embed_dim)
        """
        B, L = input_ids.shape

        # Token embedding
        x = self.token_embed(input_ids)  # (B, L, embed_dim)

        # Add position embedding
        x = x + self.pos_embed[:, :L, :]

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        return x


class MultimodalFusion(nn.Module):
    """
    Fuses visual and language features using cross-attention.

    Multiple fusion strategies are supported:
    - Concatenation: Simple feature stacking
    - Cross-attention: Language attends to vision
    - Perceiver: Learned queries attend to all inputs
    """

    def __init__(
        self,
        embed_dim: int = 768,
        n_heads: int = 12,
        n_layers: int = 2,
        fusion_type: str = "cross_attention"
    ):
        """
        Initialize fusion module.

        Args:
            embed_dim: Embedding dimension
            n_heads: Number of attention heads
            n_layers: Number of fusion layers
            fusion_type: "concat", "cross_attention", or "perceiver"
        """
        super().__init__()

        self.fusion_type = fusion_type
        self.embed_dim = embed_dim

        if fusion_type == "cross_attention":
            # Language tokens attend to vision tokens
            self.cross_attn = nn.ModuleList([
                nn.MultiheadAttention(
                    embed_dim, n_heads, dropout=0.1, batch_first=True
                )
                for _ in range(n_layers)
            ])
            self.ffn = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.GELU(),
                    nn.Linear(embed_dim * 4, embed_dim)
                )
                for _ in range(n_layers)
            ])
            self.norm1 = nn.ModuleList([
                nn.LayerNorm(embed_dim) for _ in range(n_layers)
            ])
            self.norm2 = nn.ModuleList([
                nn.LayerNorm(embed_dim) for _ in range(n_layers)
            ])

        elif fusion_type == "perceiver":
            # Learned queries attend to concatenated inputs
            n_queries = 32
            self.queries = nn.Parameter(
                torch.randn(1, n_queries, embed_dim) * 0.02
            )
            self.cross_attn = nn.MultiheadAttention(
                embed_dim, n_heads, dropout=0.1, batch_first=True
            )
            self.self_attn = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=n_heads,
                    dim_feedforward=embed_dim * 4,
                    dropout=0.1,
                    batch_first=True
                ),
                num_layers=n_layers
            )

    def forward(
        self,
        vision_features: torch.Tensor,
        language_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse vision and language features.

        Args:
            vision_features: (B, N_v, D) visual tokens
            language_features: (B, N_l, D) language tokens

        Returns:
            Fused features (B, N, D)
        """
        if self.fusion_type == "concat":
            # Simple concatenation
            return torch.cat([vision_features, language_features], dim=1)

        elif self.fusion_type == "cross_attention":
            # Language attends to vision
            x = language_features
            for i, (attn, ffn, norm1, norm2) in enumerate(zip(
                self.cross_attn, self.ffn, self.norm1, self.norm2
            )):
                # Cross attention
                attn_out, _ = attn(x, vision_features, vision_features)
                x = norm1(x + attn_out)

                # FFN
                x = norm2(x + ffn(x))

            # Concatenate with vision for downstream use
            return torch.cat([vision_features, x], dim=1)

        elif self.fusion_type == "perceiver":
            B = vision_features.shape[0]

            # Concatenate vision and language
            combined = torch.cat([vision_features, language_features], dim=1)

            # Queries attend to combined features
            queries = self.queries.expand(B, -1, -1)
            x, _ = self.cross_attn(queries, combined, combined)

            # Self-attention on queries
            x = self.self_attn(x)

            return x


class ActionDecoder(nn.Module):
    """
    Decodes fused features into robot actions.

    Supports multiple action representations:
    - Continuous: Direct regression
    - Discrete: Action tokenization
    - Chunked: Multiple timesteps at once
    """

    def __init__(
        self,
        embed_dim: int = 768,
        action_dim: int = 7,
        n_action_tokens: int = 256,
        action_type: str = "continuous",
        chunk_size: int = 1
    ):
        """
        Initialize action decoder.

        Args:
            embed_dim: Input embedding dimension
            action_dim: Dimension of action space
            n_action_tokens: Number of discrete action tokens
            action_type: "continuous" or "discrete"
            chunk_size: Number of action steps to predict
        """
        super().__init__()

        self.action_type = action_type
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.n_action_tokens = n_action_tokens

        if action_type == "continuous":
            # MLP for continuous actions
            self.decoder = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Linear(embed_dim // 2, embed_dim // 4),
                nn.GELU(),
                nn.Linear(embed_dim // 4, action_dim * chunk_size)
            )
        else:
            # Classification head for discrete tokens
            self.decoder = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, n_action_tokens * action_dim * chunk_size)
            )

        # Readout token processing
        self.readout = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        """
        Decode features into actions.

        Args:
            fused_features: (B, N, D) fused multimodal features

        Returns:
            Actions (B, action_dim) or (B, chunk_size, action_dim)
        """
        B = fused_features.shape[0]

        # Pool features (use first token as readout)
        x = fused_features[:, 0, :]  # (B, D)
        x = self.readout(x)

        # Decode to actions
        output = self.decoder(x)  # (B, action_dim * chunk_size)

        if self.action_type == "continuous":
            if self.chunk_size > 1:
                output = output.view(B, self.chunk_size, self.action_dim)
            return output
        else:
            # Reshape for discrete actions
            output = output.view(
                B, self.chunk_size, self.action_dim, self.n_action_tokens
            )
            return output


class VLAModel(nn.Module):
    """
    Complete Vision-Language-Action Model.

    Combines all components into a unified architecture that takes
    images and text, and outputs robot actions.

    Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                                                                     │
    │   Image ─────▶ [Vision Encoder] ─────┐                             │
    │                                       │                             │
    │                                       ▼                             │
    │                               [Multimodal Fusion] ───▶ [Action     │
    │                                       ▲                   Decoder]  │
    │                                       │                     │       │
    │   Text ──────▶ [Language Encoder] ───┘                     ▼       │
    │                                                         Actions     │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        image_size: int = 224,
        embed_dim: int = 768,
        vision_layers: int = 12,
        language_layers: int = 6,
        fusion_layers: int = 2,
        action_dim: int = 7,
        vocab_size: int = 32000,
        max_text_length: int = 77,
        action_type: str = "continuous",
        chunk_size: int = 1,
        fusion_type: str = "cross_attention"
    ):
        """
        Initialize VLA model.

        Args:
            image_size: Input image size
            embed_dim: Model dimension
            vision_layers: Number of vision transformer layers
            language_layers: Number of language transformer layers
            fusion_layers: Number of fusion layers
            action_dim: Robot action dimension
            vocab_size: Text vocabulary size
            max_text_length: Maximum instruction length
            action_type: "continuous" or "discrete"
            chunk_size: Number of action steps to predict
            fusion_type: Multimodal fusion strategy
        """
        super().__init__()

        self.vision_encoder = VisionEncoder(
            image_size=image_size,
            embed_dim=embed_dim,
            n_layers=vision_layers
        )

        self.language_encoder = LanguageEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            n_layers=language_layers,
            max_length=max_text_length
        )

        self.fusion = MultimodalFusion(
            embed_dim=embed_dim,
            n_layers=fusion_layers,
            fusion_type=fusion_type
        )

        self.action_decoder = ActionDecoder(
            embed_dim=embed_dim,
            action_dim=action_dim,
            action_type=action_type,
            chunk_size=chunk_size
        )

        # Optional: proprioception encoder
        self.proprio_encoder = nn.Sequential(
            nn.Linear(action_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None
    ) -> VLAOutput:
        """
        Forward pass through VLA model.

        Args:
            images: Camera images (B, C, H, W)
            input_ids: Tokenized instructions (B, L)
            attention_mask: Text attention mask (B, L)
            proprio: Proprioceptive state (B, D)

        Returns:
            VLAOutput with actions
        """
        # Encode vision
        vision_features = self.vision_encoder(images)

        # Encode language
        language_features = self.language_encoder(input_ids, attention_mask)

        # Optionally add proprioception
        if proprio is not None:
            proprio_features = self.proprio_encoder(proprio).unsqueeze(1)
            vision_features = torch.cat([vision_features, proprio_features], dim=1)

        # Fuse modalities
        fused_features = self.fusion(vision_features, language_features)

        # Decode actions
        actions = self.action_decoder(fused_features)

        return VLAOutput(action=actions, hidden_states=fused_features)


# Example usage
if __name__ == "__main__":
    print("VLA Model Demo")
    print("=" * 50)

    # Create model
    model = VLAModel(
        image_size=224,
        embed_dim=768,
        vision_layers=6,
        language_layers=4,
        action_dim=7,
        chunk_size=1
    )

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.1f}M")

    # Test forward pass
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)
    input_ids = torch.randint(0, 32000, (batch_size, 20))

    output = model(images, input_ids)

    print(f"Input image shape: {images.shape}")
    print(f"Input text shape: {input_ids.shape}")
    print(f"Output action shape: {output.action.shape}")
```

## Action Chunking with Transformers (ACT)

A key innovation for imitation learning with VLAs is action chunking—predicting multiple future actions at once.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ACTION CHUNKING                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Problem: Single-step prediction is noisy and can drift                   │
│                                                                             │
│   Single-step:                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │   t=0    t=1    t=2    t=3    t=4                                  │  │
│   │   ───▶   ───▶   ───▶   ───▶   ───▶                                │  │
│   │    ↓      ↓      ↓      ↓      ↓                                   │  │
│   │   predict predict predict predict predict                          │  │
│   │    a₀     a₁     a₂     a₃     a₄                                 │  │
│   │                                                                     │  │
│   │   Issues:                                                          │  │
│   │   • Reactive, no temporal consistency                              │  │
│   │   • Errors compound quickly                                        │  │
│   │   • Jerky motions                                                  │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Action Chunking (k=4):                                                   │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │   t=0                t=4                t=8                         │  │
│   │   ═══════════════▶   ═══════════════▶   ═══════════════▶          │  │
│   │    ↓                  ↓                  ↓                         │  │
│   │   predict            predict            predict                    │  │
│   │   [a₀,a₁,a₂,a₃]     [a₄,a₅,a₆,a₇]     [a₈,a₉,...]              │  │
│   │                                                                     │  │
│   │   Benefits:                                                        │  │
│   │   • Temporal consistency within chunk                              │  │
│   │   • Smoother motions                                               │  │
│   │   • Fewer predictions, less compounding error                      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Temporal Ensembling:                                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   Chunk from t=0:    a₀ a₁ a₂ a₃                                  │  │
│   │   Chunk from t=1:       â₁ â₂ â₃ â₄                               │  │
│   │   Chunk from t=2:          ã₂ ã₃ ã₄ ã₅                            │  │
│   │                            ─────────                                │  │
│   │   Executed action at t=2:  average(a₂, â₂, ã₂)                    │  │
│   │                                                                     │  │
│   │   → Smoother trajectories through averaging                        │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### ACT Implementation

```python
"""
Action Chunking with Transformers (ACT)

ACT predicts a sequence of future actions at once, providing temporal
consistency and smoother robot motions compared to single-step prediction.

Key features:
1. CVAE architecture for multimodal action distributions
2. Action chunking for temporal consistency
3. Temporal ensembling for smooth execution

Reference: Zhao et al. (2023) "Learning Fine-Grained Bimanual
           Manipulation with Low-Cost Hardware"
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from collections import deque


class ACTCVAE(nn.Module):
    """
    Conditional Variational Autoencoder for ACT.

    The CVAE learns a latent distribution conditioned on observations,
    allowing the model to capture multimodal action distributions
    (important when there are multiple valid ways to complete a task).

    Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                                                                     │
    │   Training:                                                        │
    │   ┌─────────┐    ┌─────────┐                                      │
    │   │ Obs     │───▶│ Encoder │                                      │
    │   │ Actions │───▶│   q(z)  │───▶ z ~ N(μ, σ) ───▶ Decoder ─▶ â   │
    │   └─────────┘    └─────────┘                                      │
    │                                                                     │
    │   Inference:                                                       │
    │   ┌─────────┐    ┌─────────┐                                      │
    │   │ Obs     │───▶│ Prior   │                                      │
    │   └─────────┘    │  p(z)   │───▶ z ~ N(μ, σ) ───▶ Decoder ─▶ â   │
    │                  └─────────┘                                      │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 512,
        chunk_size: int = 100,
        n_heads: int = 8,
        n_layers: int = 4
    ):
        """
        Initialize ACT CVAE.

        Args:
            state_dim: Observation dimension
            action_dim: Action dimension
            latent_dim: Latent space dimension
            hidden_dim: Hidden layer dimension
            chunk_size: Number of actions to predict
            n_heads: Transformer attention heads
            n_layers: Number of transformer layers
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        # Encoder: q(z | obs, actions) - used during training
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=n_layers
        )

        # Input projections for encoder
        self.obs_encoder_proj = nn.Linear(state_dim, hidden_dim)
        self.action_encoder_proj = nn.Linear(action_dim, hidden_dim)

        # Encoder output to latent
        self.encoder_to_mu = nn.Linear(hidden_dim, latent_dim)
        self.encoder_to_logvar = nn.Linear(hidden_dim, latent_dim)

        # Prior: p(z | obs) - used during inference
        self.prior_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.prior_to_mu = nn.Linear(hidden_dim, latent_dim)
        self.prior_to_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: p(actions | z, obs)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=n_layers
        )

        # Latent and obs projection for decoder
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.obs_decoder_proj = nn.Linear(state_dim, hidden_dim)

        # Action queries for decoder
        self.action_queries = nn.Parameter(
            torch.randn(1, chunk_size, hidden_dim) * 0.02
        )

        # Action output projection
        self.action_out = nn.Linear(hidden_dim, action_dim)

    def encode(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode observations and actions to latent distribution.

        Args:
            obs: Observations (B, D_obs)
            actions: Action sequence (B, T, D_action)

        Returns:
            Tuple of (mu, logvar) for latent distribution
        """
        B = obs.shape[0]

        # Project inputs
        obs_embed = self.obs_encoder_proj(obs).unsqueeze(1)  # (B, 1, H)
        action_embed = self.action_encoder_proj(actions)  # (B, T, H)

        # Concatenate
        encoder_input = torch.cat([obs_embed, action_embed], dim=1)

        # Encode
        encoded = self.encoder(encoder_input)

        # Use first token for latent
        z_embed = encoded[:, 0, :]

        mu = self.encoder_to_mu(z_embed)
        logvar = self.encoder_to_logvar(z_embed)

        return mu, logvar

    def prior(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get prior distribution from observations.

        Args:
            obs: Observations (B, D_obs)

        Returns:
            Tuple of (mu, logvar) for prior distribution
        """
        hidden = self.prior_encoder(obs)
        mu = self.prior_to_mu(hidden)
        logvar = self.prior_to_logvar(hidden)
        return mu, logvar

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Reparameterization trick for VAE.

        z = mu + sigma * epsilon, where epsilon ~ N(0, 1)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(
        self,
        z: torch.Tensor,
        obs: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode latent and observations to action sequence.

        Args:
            z: Latent vector (B, D_latent)
            obs: Observations (B, D_obs)

        Returns:
            Action sequence (B, T, D_action)
        """
        B = z.shape[0]

        # Project latent and obs
        z_embed = self.latent_proj(z).unsqueeze(1)  # (B, 1, H)
        obs_embed = self.obs_decoder_proj(obs).unsqueeze(1)  # (B, 1, H)

        # Memory for decoder
        memory = torch.cat([z_embed, obs_embed], dim=1)  # (B, 2, H)

        # Action queries
        queries = self.action_queries.expand(B, -1, -1)  # (B, T, H)

        # Decode
        decoded = self.decoder(queries, memory)  # (B, T, H)

        # Project to actions
        actions = self.action_out(decoded)  # (B, T, D_action)

        return actions

    def forward(
        self,
        obs: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        sample_prior: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            obs: Observations (B, D_obs)
            actions: Ground truth actions (B, T, D_action) - for training
            sample_prior: Whether to use prior (inference) or encoder (training)

        Returns:
            Tuple of (predicted_actions, mu, logvar)
        """
        if sample_prior or actions is None:
            # Inference: use prior
            mu, logvar = self.prior(obs)
        else:
            # Training: use encoder
            mu, logvar = self.encode(obs, actions)

        # Sample latent
        z = self.reparameterize(mu, logvar)

        # Decode to actions
        pred_actions = self.decode(z, obs)

        return pred_actions, mu, logvar


class ACTPolicy:
    """
    Action Chunking with Transformers policy.

    Implements the full ACT pipeline including:
    - Action prediction using CVAE
    - Temporal ensembling for smooth execution
    - Action chunking for consistency
    """

    def __init__(
        self,
        model: ACTCVAE,
        chunk_size: int = 100,
        temporal_ensemble_k: int = 0.01
    ):
        """
        Initialize ACT policy.

        Args:
            model: Trained ACT CVAE model
            chunk_size: Number of actions per chunk
            temporal_ensemble_k: Exponential weight for ensembling
        """
        self.model = model
        self.chunk_size = chunk_size
        self.k = temporal_ensemble_k

        # Buffer for temporal ensembling
        self.action_buffer = deque(maxlen=chunk_size)

    def reset(self):
        """Reset action buffer for new episode."""
        self.action_buffer.clear()

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Get action with temporal ensembling.

        Args:
            obs: Current observation

        Returns:
            Action to execute
        """
        self.model.eval()

        # Convert observation
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

        # Get new action chunk
        pred_actions, _, _ = self.model(obs_tensor, sample_prior=True)
        pred_actions = pred_actions.squeeze(0).numpy()  # (T, D)

        # Add to buffer
        self.action_buffer.append(pred_actions)

        # Temporal ensembling
        if len(self.action_buffer) == 1:
            action = pred_actions[0]
        else:
            # Weighted average of overlapping predictions
            weights = []
            actions_at_t = []

            for i, chunk in enumerate(self.action_buffer):
                # Index into this chunk for current timestep
                t_in_chunk = len(self.action_buffer) - 1 - i
                if t_in_chunk < len(chunk):
                    actions_at_t.append(chunk[t_in_chunk])
                    # Exponential weighting: more recent chunks have higher weight
                    weights.append(np.exp(-self.k * i))

            weights = np.array(weights)
            weights = weights / weights.sum()

            action = np.average(actions_at_t, axis=0, weights=weights)

        return action


class ACTTrainer:
    """
    Trainer for ACT model.

    Handles:
    - CVAE training with KL divergence loss
    - Action reconstruction loss
    - Learning rate scheduling
    """

    def __init__(
        self,
        model: ACTCVAE,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        kl_weight: float = 10.0
    ):
        """
        Initialize trainer.

        Args:
            model: ACT CVAE model
            lr: Learning rate
            weight_decay: L2 regularization
            kl_weight: Weight for KL divergence loss
        """
        self.model = model
        self.kl_weight = kl_weight

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    def compute_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute training loss.

        Args:
            obs: Observations (B, D_obs)
            actions: Ground truth action chunks (B, T, D_action)

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Forward pass
        pred_actions, mu, logvar = self.model(obs, actions)

        # Reconstruction loss
        recon_loss = F.mse_loss(pred_actions, actions)

        # KL divergence: D_KL(q(z|obs,a) || p(z|obs))
        prior_mu, prior_logvar = self.model.prior(obs)

        kl_loss = -0.5 * torch.sum(
            1 + logvar - prior_logvar
            - (logvar.exp() + (mu - prior_mu).pow(2)) / prior_logvar.exp()
        ) / obs.shape[0]

        # Total loss
        total_loss = recon_loss + self.kl_weight * kl_loss

        return total_loss, {
            'total': total_loss.item(),
            'recon': recon_loss.item(),
            'kl': kl_loss.item()
        }

    def train_step(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor
    ) -> dict:
        """
        Single training step.

        Args:
            obs: Observations
            actions: Ground truth action chunks

        Returns:
            Loss dictionary
        """
        self.model.train()

        loss, loss_dict = self.compute_loss(obs, actions)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()

        return loss_dict


# Example usage
if __name__ == "__main__":
    print("ACT Model Demo")
    print("=" * 50)

    # Model parameters
    state_dim = 14  # 7 joint positions + 7 joint velocities
    action_dim = 7   # 7 joint actions
    chunk_size = 50  # Predict 50 future actions

    # Create model
    model = ACTCVAE(
        state_dim=state_dim,
        action_dim=action_dim,
        chunk_size=chunk_size,
        hidden_dim=256,
        n_layers=4
    )

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.2f}M")

    # Test forward pass
    batch_size = 4
    obs = torch.randn(batch_size, state_dim)
    actions = torch.randn(batch_size, chunk_size, action_dim)

    pred_actions, mu, logvar = model(obs, actions)

    print(f"Input observation shape: {obs.shape}")
    print(f"Input action chunk shape: {actions.shape}")
    print(f"Predicted action shape: {pred_actions.shape}")
    print(f"Latent mu shape: {mu.shape}")
```

## Language-Conditioned Policies

A key advantage of VLAs is the ability to follow natural language instructions.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LANGUAGE CONDITIONING                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Single-task vs Multi-task:                                               │
│                                                                             │
│   Single-task:                Multi-task (VLA):                            │
│   ┌──────────────┐            ┌──────────────────────────────────────┐    │
│   │              │            │                                      │    │
│   │  Model_pick  │            │  "Pick up the apple"  ───┐          │    │
│   │  Model_place │            │  "Open the drawer"    ───┼──▶ VLA   │    │
│   │  Model_push  │            │  "Pour the water"     ───┘          │    │
│   │  ...         │            │                                      │    │
│   └──────────────┘            └──────────────────────────────────────┘    │
│                                                                             │
│   One model per task         One model, infinite tasks!                    │
│                                                                             │
│   Language enables:                                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   1. TASK SPECIFICATION                                            │  │
│   │      "Pick up the red cup" → specify task via language             │  │
│   │                                                                     │  │
│   │   2. ZERO-SHOT GENERALIZATION                                      │  │
│   │      Never saw "purple mug" → Still understands from language      │  │
│   │                                                                     │  │
│   │   3. COMPOSITIONAL TASKS                                           │  │
│   │      "Pick up the cup AND place it on the plate"                  │  │
│   │                                                                     │  │
│   │   4. CLARIFICATION / FEEDBACK                                      │  │
│   │      "No, the OTHER cup" → Refine through dialogue                │  │
│   │                                                                     │  │
│   │   5. REASONING                                                     │  │
│   │      "Get me something to drink" → Must infer cup/bottle/etc.     │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Language-Conditioned Policy Implementation

```python
"""
Language-Conditioned Robot Policy

Enables robots to follow natural language instructions by conditioning
action predictions on text embeddings.

Key components:
1. Text encoder (CLIP, BERT, or custom)
2. Visual encoder for scene understanding
3. Cross-modal fusion
4. Action decoder conditioned on instruction
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, List
from transformers import CLIPTextModel, CLIPTokenizer


class CLIPLanguageEncoder:
    """
    Language encoder using CLIP's text model.

    CLIP embeddings work well because they're trained to align
    with visual concepts, which is useful for grounded language.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize CLIP language encoder.

        Args:
            model_name: HuggingFace model name
        """
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name)
        self.model.eval()

        # Freeze CLIP weights
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        """
        Encode text to embeddings.

        Args:
            texts: List of instruction strings

        Returns:
            Text embeddings (B, D)
        """
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )

        outputs = self.model(**inputs)
        # Use pooled output (CLS token)
        return outputs.pooler_output


class LanguageConditionedPolicy(nn.Module):
    """
    Policy that conditions actions on language instructions.

    Fuses visual observations with language embeddings to produce
    context-aware actions.
    """

    def __init__(
        self,
        visual_dim: int,
        language_dim: int = 512,
        hidden_dim: int = 512,
        action_dim: int = 7,
        proprio_dim: int = 0
    ):
        """
        Initialize language-conditioned policy.

        Args:
            visual_dim: Visual feature dimension
            language_dim: Language embedding dimension
            hidden_dim: Hidden layer dimension
            action_dim: Action output dimension
            proprio_dim: Proprioception dimension (0 if not used)
        """
        super().__init__()

        self.visual_dim = visual_dim
        self.language_dim = language_dim

        # Visual processing
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Language processing (project to same dim)
        self.language_encoder = nn.Sequential(
            nn.Linear(language_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Optional proprioception
        if proprio_dim > 0:
            self.proprio_encoder = nn.Sequential(
                nn.Linear(proprio_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim)
            )
            fusion_dim = hidden_dim * 3
        else:
            self.proprio_encoder = None
            fusion_dim = hidden_dim * 2

        # FiLM conditioning: language modulates visual features
        self.film_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2)  # gamma and beta
        )

        # Policy head
        self.policy = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

    def forward(
        self,
        visual_features: torch.Tensor,
        language_embedding: torch.Tensor,
        proprio: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            visual_features: Visual observations (B, visual_dim)
            language_embedding: Language embeddings (B, language_dim)
            proprio: Proprioceptive state (B, proprio_dim)

        Returns:
            Actions (B, action_dim)
        """
        # Encode modalities
        visual = self.visual_encoder(visual_features)
        language = self.language_encoder(language_embedding)

        # FiLM conditioning: language modulates visual
        film_params = self.film_generator(language)
        gamma, beta = film_params.chunk(2, dim=-1)
        visual_conditioned = gamma * visual + beta

        # Concatenate features
        if self.proprio_encoder is not None and proprio is not None:
            proprio_enc = self.proprio_encoder(proprio)
            features = torch.cat([visual_conditioned, language, proprio_enc], dim=-1)
        else:
            features = torch.cat([visual_conditioned, language], dim=-1)

        # Predict action
        action = self.policy(features)

        return action


class GoalConditionedVLA(nn.Module):
    """
    VLA with explicit goal conditioning.

    Supports both language goals and visual goals (images of desired state).
    """

    def __init__(
        self,
        image_encoder,
        language_encoder,
        hidden_dim: int = 512,
        action_dim: int = 7
    ):
        """
        Initialize goal-conditioned VLA.

        Args:
            image_encoder: Vision encoder module
            language_encoder: Language encoder module
            hidden_dim: Hidden dimension
            action_dim: Action dimension
        """
        super().__init__()

        self.image_encoder = image_encoder
        self.language_encoder = language_encoder

        # Goal type embedding
        self.goal_type_embed = nn.Embedding(2, hidden_dim)  # 0: language, 1: image

        # Fusion
        self.fusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=4
        )

        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

    def forward(
        self,
        current_image: torch.Tensor,
        goal_text: Optional[torch.Tensor] = None,
        goal_image: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with flexible goal specification.

        Args:
            current_image: Current observation
            goal_text: Language goal embedding
            goal_image: Goal image

        Returns:
            Actions
        """
        B = current_image.shape[0]

        # Encode current state
        current_features = self.image_encoder(current_image)  # (B, N, D)

        # Encode goal
        if goal_text is not None:
            goal_features = self.language_encoder(goal_text)  # (B, D)
            goal_type = torch.zeros(B, dtype=torch.long, device=current_image.device)
        elif goal_image is not None:
            goal_features = self.image_encoder(goal_image)[:, 0, :]  # Use CLS
            goal_type = torch.ones(B, dtype=torch.long, device=current_image.device)
        else:
            raise ValueError("Must provide either goal_text or goal_image")

        # Add goal type embedding
        goal_features = goal_features + self.goal_type_embed(goal_type)

        # Concatenate
        goal_features = goal_features.unsqueeze(1)  # (B, 1, D)
        combined = torch.cat([current_features, goal_features], dim=1)

        # Fuse
        fused = self.fusion(combined)

        # Predict action from first token
        action = self.action_head(fused[:, 0, :])

        return action
```

## The VLA Landscape

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MAJOR VLA MODELS                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   RT-1 (Google, 2022)                                                      │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │ • First large-scale VLA (35M params)                            │      │
│   │ • Trained on 130K robot demonstrations                          │      │
│   │ • 700+ tasks with language conditioning                         │      │
│   │ • Uses EfficientNet + TokenLearner + Transformer                │      │
│   │ • Discrete action tokens (256 bins per dimension)              │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│   RT-2 (Google, 2023)                                                      │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │ • Built on PaLI-X (55B params) and PaLM-E (12B params)        │      │
│   │ • Web-scale pretraining enables semantic understanding         │      │
│   │ • Chain-of-thought reasoning for complex tasks                 │      │
│   │ • Zero-shot transfer to new objects/concepts                   │      │
│   │ • Actions as text tokens in vocabulary                         │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│   PaLM-E (Google, 2023)                                                    │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │ • 562B parameter embodied multimodal model                      │      │
│   │ • Integrates vision, language, and embodiment                  │      │
│   │ • Enables complex reasoning about physical world               │      │
│   │ • Can plan multi-step tasks from high-level instructions       │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│   OpenVLA (Berkeley, 2024)                                                 │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │ • Open-source VLA (7B params)                                   │      │
│   │ • Based on Prismatic VLM architecture                          │      │
│   │ • Trained on Open X-Embodiment dataset                         │      │
│   │ • Competitive with RT-2 on many benchmarks                     │      │
│   │ • Enables community research on VLAs                           │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│   Octo (Berkeley, 2023)                                                    │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │ • Generalist robot policy (93M params)                         │      │
│   │ • Transformer-based, supports multiple embodiments             │      │
│   │ • Goal conditioned (language + image goals)                    │      │
│   │ • Designed for fine-tuning on new robots                       │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│   Comparison:                                                              │
│   ┌────────────┬──────────┬────────────┬───────────────┬───────────┐      │
│   │   Model    │  Params  │  Training  │   Strengths   │  Access   │      │
│   ├────────────┼──────────┼────────────┼───────────────┼───────────┤      │
│   │ RT-1      │   35M    │  130K demo │  Real robots  │  Closed   │      │
│   │ RT-2      │   55B    │  Web + demo│  Reasoning    │  Closed   │      │
│   │ PaLM-E    │  562B    │  Multimodal│  Scale        │  Closed   │      │
│   │ OpenVLA   │    7B    │  OXE data  │  Open source  │  Open     │      │
│   │ Octo      │   93M    │  OXE data  │  Efficiency   │  Open     │      │
│   └────────────┴──────────┴────────────┴───────────────┴───────────┘      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Training VLA Models

```python
"""
VLA Training Pipeline

Training VLAs requires:
1. Large-scale demonstration data
2. Appropriate loss functions
3. Careful action representation
4. Efficient batching for multimodal data
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class VLADataPoint:
    """Single training example for VLA."""
    image: np.ndarray           # (H, W, C) camera image
    instruction: str            # Natural language instruction
    action: np.ndarray          # (action_dim,) ground truth action
    proprio: Optional[np.ndarray] = None  # Optional proprioception


class VLADataset(Dataset):
    """
    Dataset for VLA training.

    Handles:
    - Image preprocessing and augmentation
    - Text tokenization
    - Action normalization
    - Proper batching
    """

    def __init__(
        self,
        data: List[VLADataPoint],
        tokenizer,
        image_size: int = 224,
        action_mean: np.ndarray = None,
        action_std: np.ndarray = None,
        augment: bool = True
    ):
        """
        Initialize dataset.

        Args:
            data: List of training examples
            tokenizer: Text tokenizer
            image_size: Target image size
            action_mean: Action normalization mean
            action_std: Action normalization std
            augment: Whether to apply data augmentation
        """
        self.data = data
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.augment = augment

        # Compute action statistics if not provided
        if action_mean is None or action_std is None:
            all_actions = np.stack([d.action for d in data])
            self.action_mean = all_actions.mean(axis=0)
            self.action_std = all_actions.std(axis=0) + 1e-6
        else:
            self.action_mean = action_mean
            self.action_std = action_std

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        # Process image
        image = self._process_image(item.image)

        # Tokenize text
        tokens = self.tokenizer(
            item.instruction,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )

        # Normalize action
        action = (item.action - self.action_mean) / self.action_std

        result = {
            "image": image,
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "action": torch.FloatTensor(action)
        }

        if item.proprio is not None:
            result["proprio"] = torch.FloatTensor(item.proprio)

        return result

    def _process_image(self, image: np.ndarray) -> torch.Tensor:
        """Process and optionally augment image."""
        import torchvision.transforms as T

        transforms = [
            T.ToPILImage(),
            T.Resize((self.image_size, self.image_size)),
        ]

        if self.augment:
            transforms.extend([
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            ])

        transforms.extend([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        transform = T.Compose(transforms)
        return transform(image)


class VLATrainer:
    """
    Trainer for VLA models.

    Features:
    - Mixed precision training
    - Gradient accumulation
    - Learning rate scheduling
    - Action loss computation
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        grad_accumulation: int = 1,
        use_amp: bool = True,
        device: str = "cuda"
    ):
        """
        Initialize trainer.

        Args:
            model: VLA model to train
            lr: Learning rate
            weight_decay: Weight decay
            warmup_steps: LR warmup steps
            max_steps: Maximum training steps
            grad_accumulation: Gradient accumulation steps
            use_amp: Use automatic mixed precision
            device: Training device
        """
        self.model = model.to(device)
        self.device = device
        self.grad_accumulation = grad_accumulation
        self.use_amp = use_amp
        self.max_steps = max_steps

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # LR scheduler with warmup
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            return 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / (max_steps - warmup_steps)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda
        )

        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None

    def compute_loss(
        self,
        pred_actions: torch.Tensor,
        target_actions: torch.Tensor,
        action_type: str = "continuous"
    ) -> torch.Tensor:
        """
        Compute action prediction loss.

        Args:
            pred_actions: Predicted actions
            target_actions: Ground truth actions
            action_type: "continuous" or "discrete"

        Returns:
            Loss value
        """
        if action_type == "continuous":
            # MSE loss for continuous actions
            return nn.MSELoss()(pred_actions, target_actions)
        else:
            # Cross-entropy for discrete action tokens
            return nn.CrossEntropyLoss()(
                pred_actions.view(-1, pred_actions.size(-1)),
                target_actions.view(-1)
            )

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step.

        Args:
            batch: Batch of training data

        Returns:
            Dictionary of loss values
        """
        self.model.train()

        # Move to device
        images = batch["image"].to(self.device)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        actions = batch["action"].to(self.device)
        proprio = batch.get("proprio", None)
        if proprio is not None:
            proprio = proprio.to(self.device)

        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            output = self.model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                proprio=proprio
            )

            loss = self.compute_loss(output.action, actions)
            loss = loss / self.grad_accumulation

        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return {"loss": loss.item() * self.grad_accumulation}

    def optimizer_step(self):
        """Perform optimizer step after accumulation."""
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        self.scheduler.step()
        self.optimizer.zero_grad()

    def train(
        self,
        dataloader: DataLoader,
        n_epochs: int = 10,
        log_interval: int = 100
    ) -> List[float]:
        """
        Full training loop.

        Args:
            dataloader: Training data loader
            n_epochs: Number of epochs
            log_interval: Logging frequency

        Returns:
            List of loss values
        """
        losses = []
        step = 0

        for epoch in range(n_epochs):
            epoch_loss = 0
            n_batches = 0

            for batch_idx, batch in enumerate(dataloader):
                loss_dict = self.train_step(batch)
                epoch_loss += loss_dict["loss"]
                n_batches += 1

                # Optimizer step after accumulation
                if (batch_idx + 1) % self.grad_accumulation == 0:
                    self.optimizer_step()
                    step += 1

                    if step % log_interval == 0:
                        avg_loss = epoch_loss / n_batches
                        lr = self.scheduler.get_last_lr()[0]
                        print(f"Step {step}: Loss = {avg_loss:.4f}, LR = {lr:.6f}")
                        losses.append(avg_loss)

            print(f"Epoch {epoch + 1}: Avg Loss = {epoch_loss / n_batches:.4f}")

        return losses
```

## Industry Perspective: VLA in Practice

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    VLA IN INDUSTRY                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   GOOGLE DEEPMIND                                                          │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │ • RT-1/RT-2: Production VLAs for mobile manipulation           │      │
│   │ • Fleet of 13 robots collecting data continuously              │      │
│   │ • 130K demonstrations across 700+ tasks                        │      │
│   │ • RT-2 shows emergent reasoning from web pretraining          │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│   FIGURE AI                                                                │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │ • Integration of VLA with humanoid robots                       │      │
│   │ • Partnership with OpenAI for language capabilities            │      │
│   │ • Focus on natural human-robot interaction                     │      │
│   │ • Real-time conversation while performing tasks                │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│   1X (NEO)                                                                 │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │ • VLA-based humanoid robots for home tasks                      │      │
│   │ • End-to-end learning from demonstrations                      │      │
│   │ • Focus on generalization to everyday environments             │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│   PHYSICAL INTELLIGENCE                                                    │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │ • π₀ (pi-zero): VLA for general-purpose manipulation          │      │
│   │ • Flow matching for action generation                          │      │
│   │ • Trained on diverse robot platforms                           │      │
│   │ • Strong zero-shot and few-shot transfer                       │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│   Key Trends:                                                              │
│   • Increasing model scale (35M → 55B parameters)                         │
│   • Web-scale pretraining becoming standard                                │
│   • Open source alternatives emerging (OpenVLA, Octo)                     │
│   • Integration with LLMs for reasoning                                   │
│   • Real-world deployment accelerating                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Summary

### Key Takeaways

1. **End-to-End Learning**: VLAs unify perception, language understanding, and action generation in a single model

2. **Language Conditioning**: Natural language enables flexible task specification and zero-shot generalization

3. **Scale Matters**: Larger models pretrained on web data show emergent reasoning capabilities

4. **Action Representation**: Both discrete tokens (RT-2) and continuous outputs (ACT) work well

5. **Temporal Consistency**: Action chunking and temporal ensembling are crucial for smooth execution

### VLA Architecture Choices

| Component | Options | Trade-offs |
|-----------|---------|------------|
| Vision Encoder | ViT, EfficientNet, ResNet | Accuracy vs Speed |
| Language Encoder | CLIP, BERT, T5 | Grounding vs Reasoning |
| Fusion | Concat, Cross-attention, Perceiver | Simplicity vs Expressiveness |
| Action Output | Continuous, Discrete, Diffusion | Precision vs Multimodality |
| Action Chunking | Single-step, Multi-step | Reactivity vs Consistency |

### Practical Checklist

- [ ] Selected appropriate model scale for deployment constraints
- [ ] Designed action representation (continuous vs discrete)
- [ ] Implemented proper action normalization
- [ ] Set up data collection pipeline with language annotations
- [ ] Configured multimodal data loading and batching
- [ ] Implemented evaluation on held-out tasks
- [ ] Tested language generalization to new instructions
- [ ] Verified real-time inference capability

## Further Reading

### Foundational Works
- Brohan et al. "RT-1: Robotics Transformer" (2022)
- Brohan et al. "RT-2: Vision-Language-Action Models" (2023)
- Driess et al. "PaLM-E: An Embodied Multimodal Language Model" (2023)

### Implementation Resources
- OpenVLA: Open-source 7B VLA
- Octo: Generalist robot policy
- Mobile ALOHA / ACT: Action chunking

### Datasets
- Open X-Embodiment: 1M+ robot trajectories
- BridgeData V2: Diverse tabletop manipulation
- DROID: Robot manipulation demonstrations

---

*"VLAs represent a paradigm shift in robotics—from programming robots to talking to them. The question is no longer 'can robots understand language?' but 'what can't they understand?'"*
