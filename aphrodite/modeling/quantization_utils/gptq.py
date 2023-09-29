from typing import Any, Dict, List

import torch

from aphrodite.modeling.quantization_utils.base import QuantizationConfig


class GPTQConfig(QuantizationConfig):
    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        desc_act: bool,
    ) -> None:
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.desc_act = desc_act
        self.pack_factor = 32 // self.weight_bits
        if self.weight_bits != 4:
            raise ValueError(
                f"Currently only 4-bit quant is supported for GPTQ, you passed {self.weight_bits} bits.")
    
    def __repr__(self) -> str:
        return (f"GPTQConfig(weight_bits={self.weight_bits}), "
                f"group_size={self.group_size}, "
                f"desc_act={self.desc_act}")
    
    @classmethod
    def get_name(cls) -> str:
        return "gptq"
    
    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half]
    
    @classmethod
    def get_min_capability(cls) -> int:
        return 60
    
    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return [
            "quantize_config.json",
        ]
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GPTQConfig":
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        desc_act = cls.get_from_keys(config, ["desc_act"])
        return cls(weight_bits, group_size, desc_act)
    
    @classmethod
    def get_packed_tensor_names(cls) -> List[str]:
        return ["qzeros"]
    
    @classmethod
    def get_transposed_tensor_names(cls) -> List[str]:
        return ["qweight", "qzeros", "scales"]
    
    def get_row_tp_tensor_names(self) -> List[str]:
        if self.desc_act and self.group_size != -1:
            return ["qweight", "g_idx"]
        if self.group_size == -1:
            return ["qweight"]
        return ["qweight", "qzeros", "scales"]
    
    def get_column_tp_tensor_names(self) -> List[str]:
        return ["qweight", "qzeros", "scales", "bias"]
    
    def get_ignore_tensor_names(self) -> List[str]:
        if self.desc_act and self.group_size != -1:
            return []
        return ["g_idx"]
        