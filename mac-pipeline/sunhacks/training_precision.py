"""Reproduce the reviewed trainer's frozen BF16 values with FP32 Mac kernels."""
def bf16_base_in_fp32(module):
    # The trainer calls .bfloat16() BEFORE injecting FP32 LoRA/head weights.
    # Widening later cannot recover discarded mantissa bits. Apply this only
    # to the frozen base, before adapters; inference remains FP32 on the Mac.
    return module.bfloat16().float()
