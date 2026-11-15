"""Test GPU Availability and CUDA operations on RTX 5070."""
import pytest
import torch

def test_cuda_available():
    assert torch.cuda.is_available(), "CUDA must be available for execution!"
    assert torch.cuda.device_count() >= 1, "At least 1 CUDA device must be present."

def test_rtx_gpu_detection():
    device_name = torch.cuda.get_device_name(0)
    print(f"\nDetected GPU: {device_name}")
    assert "RTX" in device_name or "NVIDIA" in device_name, f"Expected NVIDIA GPU, found {device_name}"

def test_gpu_memory_and_tensor_compute():
    torch.cuda.empty_cache()
    start_mem = torch.cuda.memory_allocated(0)
    
    # Allocate BF16 tensor on GPU
    x = torch.randn(2048, 2048, device="cuda:0", dtype=torch.bfloat16)
    y = torch.matmul(x, x)
    torch.cuda.synchronize()
    
    end_mem = torch.cuda.memory_allocated(0)
    assert end_mem > start_mem, "VRAM must increase when allocating GPU tensors."
    assert y.device.type == "cuda", "Computed tensor must reside on CUDA device."
    assert y.dtype == torch.bfloat16, "Tensor dtype must be bfloat16."
