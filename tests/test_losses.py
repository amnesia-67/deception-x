"""Test Loss Functions and Gradient Flow."""
import pytest
import torch
from src.multimodal_fusion import Branch1MultimodalNetwork, MultiKernelMMD

def test_mmd_loss():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    mmd = MultiKernelMMD().to(device)
    
    src = torch.randn(10, 256, device=device)
    tgt = torch.randn(10, 256, device=device)
    
    loss_diff = mmd(src, tgt)
    loss_same = mmd(src, src)
    
    assert loss_diff.item() >= 0.0
    assert torch.isclose(loss_same, torch.tensor(0.0, device=device), atol=1e-5)

def test_loss_backward_and_gradients():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    net = Branch1MultimodalNetwork(
        visual_dim=1024,
        audio_dim=768,
        text_dim=128,
        d=256
    ).to(device, dtype=dtype)
    
    B = 4
    F_v = torch.randn(B, 32, 1024, device=device, dtype=dtype, requires_grad=True)
    F_a = torch.randn(B, 32, 768, device=device, dtype=dtype, requires_grad=True)
    f_t = torch.randn(B, 128, device=device, dtype=dtype, requires_grad=True)
    labels = torch.tensor([1, 0, 1, 0], device=device, dtype=torch.float32)
    
    out = net(F_v, F_a, f_t)
    losses = net.compute_loss(out, labels)
    
    total_loss = losses["loss_total"]
    assert torch.isfinite(total_loss)
    
    total_loss.backward()
    
    # Check that network parameters received gradients
    has_grads = [p.grad is not None for p in net.parameters() if p.requires_grad]
    assert all(has_grads), "All trainable parameters must have valid gradients!"
