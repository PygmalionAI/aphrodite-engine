import torch

from aphrodite.platforms import current_platform
from aphrodite.quantization import QUANTIZATION_METHODS


def is_quant_method_supported(quant_method: str) -> bool:
    # Currently, all quantization methods require Nvidia or AMD GPUs
    if not torch.cuda.is_available():
        return False

    capability = current_platform.get_device_capability()
    capability = capability[0] * 10 + capability[1]
    return (capability >=
            QUANTIZATION_METHODS[quant_method].get_min_capability())
