import os

APHRODITE_USE_CUDA_LORA = bool(os.getenv("APHRODITE_USE_CUDA_LORA", False))


if not APHRODITE_USE_CUDA_LORA:
    from aphrodite.lora.utils import LoRATritonMapping as LoRAMapping
else:
    from aphrodite.lora.utils import LoRACUDAMapping as LoRAMapping

__all__ = ["LoRAMapping"]
