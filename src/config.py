"""
Configuration and Hyperparameters for Tri-Modal Deception Detection System
Accredited to Gemini.pdf Specifications (November 2026).
Target Hardware: NVIDIA GeForce RTX 5070 (12 GB VRAM).
"""

import os
from dataclasses import dataclass
from typing import Any

try:
    import torch
    TORCH_AVAILABLE = True
    DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    DEFAULT_DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
except ImportError:
    torch = None
    TORCH_AVAILABLE = False
    DEFAULT_DEVICE = "cpu"
    DEFAULT_DTYPE = None

@dataclass
class ModelConfig:
    # Model checkpoints & paths
    dasheng_model_path: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models",
        "dasheng-base"
    )
    
    # Target temporal tubelets
    temporal_steps: int = 32
    
    # Modality dimensions
    visual_dim: int = 1024       # V-JEPA 2 pooled tubelet dim
    audio_dasheng_dim: int = 768 # Dasheng-base frame hidden dim
    audio_full_dim: int = 1792   # Dasheng 768 + emotion2vec 1024
    text_dim: int = 128          # Qwen3 MRL sliced dimension
    shared_latent_dim: int = 256 # d = 256 shared space
    
    # 250-feature Z-score tabular vector
    tabular_dim: int = 250
    
    # Loss hyperparameters (locked at Gemini.pdf Page 25-26)
    lambda_1_dmc: float = 0.25
    lambda_2_ent: float = 0.05
    lambda_3_mmpda: float = 0.30
    delta_margin: float = 0.45
    weight_decay: float = 1e-4
    learning_rate: float = 2e-4
    
    # DARE quantile
    dare_rho: float = 0.50 # Retains top 50% salience tokens
    
    # Device and precision
    device: str = DEFAULT_DEVICE
    dtype: Any = DEFAULT_DTYPE

    # Training and hardware constraints
    batch_size: int = 16 # Optimal for 12GB RTX 5070 VRAM with BF16
    vram_budget_mb: float = 4200.0 # Strict hackathon limit from Page 40 of PDF
    
    # Sub-par threshold for run termination
    min_acceptable_accuracy: float = 0.65
    loss_explosion_threshold: float = 5.0
