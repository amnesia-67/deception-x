"""Test DARE and ReDA Multimodal Fusion."""
import pytest
import torch
from src.multimodal_fusion import DAREModule, Branch1MultimodalNetwork

def test_dare_module():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dare = DAREModule(d=256, rho=0.50).to(device)
    
    B, T, d = 4, 32, 256
    h_v = torch.randn(B, T, d, device=device)
    h_a = torch.randn(B, T, d, device=device)
    h_l = torch.randn(B, d, device=device)
    
    h_v_hat, h_a_hat, D_va, D_vl, D_al = dare(h_v, h_a, h_l)
    
    assert h_v_hat.shape == (B, d)
    assert h_a_hat.shape == (B, d)
    assert D_va.shape == (B,)
    assert D_vl.shape == (B,)
    assert D_al.shape == (B,)
    assert (D_va >= 0.0).all() and (D_va <= 2.0).all()

def test_branch1_network_forward():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    net = Branch1MultimodalNetwork(
        visual_dim=1024,
        audio_dim=768,
        text_dim=128,
        d=256
    ).to(device, dtype=dtype)
    
    B = 4
    F_v = torch.randn(B, 32, 1024, device=device, dtype=dtype)
    F_a = torch.randn(B, 32, 768, device=device, dtype=dtype)
    f_t = torch.randn(B, 128, device=device, dtype=dtype)
    
    out = net(F_v, F_a, f_t)
    
    assert "p_neural" in out
    assert "z_fused" in out
    assert "alpha" in out
    assert out["p_neural"].shape == (B,)
    assert out["z_fused"].shape == (B, 256)
    assert out["alpha"].shape == (B, 3)
    
    # Probabilities in [0, 1]
    assert (out["p_neural"] >= 0.0).all() and (out["p_neural"] <= 1.0).all()
    # Modality weights sum to 1
    alpha_sum = out["alpha"].sum(dim=-1)
    assert torch.allclose(alpha_sum, torch.ones_like(alpha_sum), atol=1e-3)
