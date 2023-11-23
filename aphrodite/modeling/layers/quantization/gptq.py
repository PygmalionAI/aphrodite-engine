from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from aphrodite import quantization_ops
from aphrodite.modeling.layers.linear import (LinearMethodBase,
                                              set_weight_attrs)
from aphrodite.modeling.layers.quantization.base_config import (
    QuantizationConfig)

class GPTQConfig(QuantizationConfig):
    """Config class for GPTQ.

    Reference: https://arxiv.org/abs/2306.00978
    """

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
        # exllama kernel v1 only supports 4 bit
        if self.weight_bits != 4:
            raise ValueError(
                "Currently, only 4-bit weight quantization is supported for "
                f"GPTQ, but got {self.weight_bits} bits.")

    def __repr__(self) -> str:
        return (f"GPTQConfig(weight_bits={self.weight_bits}, "
                f"group_size={self.group_size}, "
                f"desc_act={self.desc_act})")

    @classmethod
    def get_name(cls) -> str:
        return "gptq"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half]

    @classmethod
    # Need to figure it out
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
    
    def get_linear_method(self) -> "GPTQLinearMethod":
        return GPTQLinearMethod(self)

class ExLlamaV2DeviceTensors:

    def __init__(self, device_idx, scratch_bytes):
        self.device_idx = device_idx
        self.scratch_bytes = scratch_bytes
        self.scratch = None

    def prepare(self):
        self.scratch = torch.empty(
            (self.scratch_bytes // 2, ),
            dtype=torch.half,
            device=f"cuda:{self.device_idx}",
        )

    def get_scratch_slice(self, size_bytes):
        if self.scratch is None:
            self.prepare()
        size_bytes = ((size_bytes + 127) // 128) * 128
        size_half = size_bytes // 2
        scratch_slice = self.scratch.narrow(0, 0, size_half)
        return scratch_slice


class GPTQLinearMethod(LinearMethodBase):
    """Linear method for GPTQ.
    Args:
        quant_config: The GPTQ quantization config.
    """
    def __init__(self, quant_config: GPTQConfig):
        self.quant_config = quant_config
    
    def create_weights(self, input_size: int, output_size: int,
                       params_dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        if input_size % self.quant_config.group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")
        if output_size % self.quant_config.pack_factor != 0:
            raise ValueError(
                "The output size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")
        if self.quant_config.desc_act and self.quant_config.group_size != -1:
            group_number = input_size // self.quant_config.group_size
            self.use_exllama = False
        else:
            group_number = input_size // self.quant_config.group_size
            self.use_exllama = True
        self.input_size = input_size
        self.output_size = output_size
        
        qweight = Parameter(
            torch.empty(
                input_size // self.quant_config.pack_factor,
                output_size,
                device="cuda",
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight, {
                "input_dim": 0,
                "output_dim": 1,
                "packed_dim": 0,
                "pack_factor": self.quant_config.pack_factor,
            })
        qzeros = Parameter(
            torch.empty(
                input_size // self.quant_config.group_size,
                output_size // self.quant_config.pack_factor,
                device="cuda",
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qzeros, {
                "input_dim": 0,
                "output_dim": 1,
                "packed_dim": 1,
                "pack_factor": self.quant_config.pack_factor,
            })
        scales = Parameter(
            torch.empty(
                input_size // self.quant_config.group_size,
                output_size,
                device="cuda",
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(scales, {
            "input_dim": 0,
            "output_dim": 1,
        })
        g_idx = Parameter(
            torch.tensor(
                [
                    i // self.quant_config.group_size
                    for i in range(input_size)
                ],
                device="cuda",
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(g_idx, {
            "input_dim": 0,
        })
        return {
            "qweight": qweight,
            "qzeros": qzeros,
            "scales": scales,
            "g_idx": g_idx,
        }
    
    def apply_weights(self, x: torch.Tensor) -> torch.Tensor:
        out_shape = x.shape[:-1] + (self.qweight.shape[-1], )
        reshaped_x = x.reshape(-1, x.shape[-1])

        if self.use_exllama:
            output = torch.empty((reshaped_x.shape[0], self.qweight.shape[-1]),
                                 dtype=torch.float16,
                                 device=x.device)
            quantization_ops.gemm_half_q_half(reshaped_x, self.q4, output,
                                              False)
        else:
            output = torch.zeros((reshaped_x.shape[0], self.qweight.shape[-1]),
                                 dtype=torch.float32,
                                 device=x.device)
            quantization_ops.gptq_descact_matmul(reshaped_x.float(),
                                                 self.qweight, output,
                                                 self.scales.float(),
                                                 self.qzeros, self.g_idx)
            output = output.half()
        return output.reshape(out_shape)

        

class GPTQLinear(torch.nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 *,
                 bias=True,
                 quant_config=None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.quant_config = quant_config
        self.use_exllama = True
        group_size = self.quant_config.group_size if (
            self.quant_config.group_size != -1) else self.input_size
        self.qweight = Parameter(
            torch.empty(
                self.input_size // self.quant_config.pack_factor,
                self.output_size,
                device="cuda",
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        self.qzeros = Parameter(
            torch.empty(
                self.input_size // group_size,
                self.output_size // self.quant_config.pack_factor,
                device="cuda",
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        self.scales = Parameter(
            torch.empty(
                self.input_size // group_size,
                self.output_size,
                device="cuda",
                dtype=torch.float16,
            ),
            requires_grad=False,
        )
        self.g_idx = Parameter(
            torch.tensor(
                [i // group_size for i in range(self.input_size)],
                device="cuda",
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size,
                            device="cuda",
                            dtype=torch.float16))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

    def post_init(self, temp_dq):
        assert self.qweight.device.type == "cuda"
        assert self.qweight.device.index is not None

        none_tensor = torch.empty((1, 1), device="meta")
        temp_dq = temp_dq.get_scratch_slice(self.temp_dq_size())
        if not self.quant_config.desc_act:
            self.q4 = quantization_ops.make_q_matrix(
                self.qweight,
                none_tensor,
                none_tensor,
                self.qzeros,
                self.scales,
                none_tensor,
                temp_dq,
            )
        else:
            self.q_perm = torch.empty((self.input_size, ),
                                      dtype=torch.short,
                                      device=self.qweight.device)
            self.q_invperm = torch.empty_like(self.q_perm)
            self.q4 = quantization_ops.make_q_matrix(
                self.qweight,
                self.q_perm,
                self.q_invperm,
                self.qzeros,
                self.scales,
                self.g_idx.cpu(),
                temp_dq,
            )

    def forward(self, input_):
        out_shape = input_.shape[:-1] + (self.qweight.shape[-1], )
        reshaped_x = input_.reshape(-1, input_.shape[-1])
        output = torch.empty((reshaped_x.shape[0], self.qweight.shape[-1]),
                             dtype=torch.float16,
                             device=input_.device)
        quantization_ops.gemm_half_q_half(reshaped_x, self.q4, output, False)
        output = output.reshape(out_shape)

        output = output + self.bias if self.bias is not None else output
        return output

    def temp_dq_size(self):
        return self.input_size * self.output_size * 2 + 128

    def temp_fwd_size(self, max_tokens):
        return self.output_size * max_tokens * 4 + 128

    def scratch_space_fixed(self, max_tokens):
        return self.temp_dq_size() + self.temp_fwd_size(max_tokens)


# class GPTQColumnParallelLinear(ColumnParallelLinear):

#     def create_weights(self, dtype: torch.dtype) -> None:
#         assert self.input_size % self.quant_config.pack_factor == 0
#         assert (self.output_size_per_partition %
#                 self.quant_config.pack_factor == 0)
#         self.use_exllama = True
#         group_size = self.quant_config.group_size if (
#             self.quant_config.group_size != -1) else self.input_size

#         self.qweight = Parameter(
#             torch.empty(
#                 self.input_size // self.quant_config.pack_factor,
#                 self.output_size_per_partition,
#                 device="cuda",
#                 dtype=torch.int32,
#             ),
#             requires_grad=False,
#         )
#         self.qzeros = Parameter(
#             torch.empty(
#                 self.input_size // group_size,
#                 self.output_size_per_partition //
#                 self.quant_config.pack_factor,
#                 device="cuda",
#                 dtype=torch.int32,
#             ),
#             requires_grad=False,
#         )
#         self.scales = Parameter(
#             torch.empty(
#                 self.input_size // group_size,
#                 self.output_size_per_partition,
#                 device="cuda",
#                 dtype=dtype,
#             ),
#             requires_grad=False,
#         )
#         self.g_idx = Parameter(
#             torch.tensor(
#                 [i // group_size for i in range(self.input_size)],
#                 device="cuda",
#                 dtype=torch.int32,
#             ),
#             requires_grad=False,
#         )

#     def post_init(self, temp_dq):
#         assert self.qweight.device.type == "cuda"
#         assert self.qweight.device.index is not None

#         none_tensor = torch.empty((1, 1), device="meta")
#         temp_dq = temp_dq.get_scratch_slice(self.temp_dq_size())
#         if not self.quant_config.desc_act:
#             self.q4 = quantization_ops.make_q_matrix(
#                 self.qweight,
#                 none_tensor,
#                 none_tensor,
#                 self.qzeros,
#                 self.scales,
#                 none_tensor,
#                 temp_dq,
#             )
#         else:
#             self.q_perm = torch.empty((self.input_size, ),
#                                       dtype=torch.short,
#                                       device=self.qweight.device)
#             self.q_invperm = torch.empty_like(self.q_perm)
#             self.q4 = quantization_ops.make_q_matrix(
#                 self.qweight,
#                 self.q_perm,
#                 self.q_invperm,
#                 self.qzeros,
#                 self.scales,
#                 self.g_idx.cpu(),
#                 temp_dq,
#             )

#     def apply_weights(
#         self,
#         x: torch.Tensor,
#         bias: Optional[torch.Tensor],
#     ) -> torch.Tensor:
#         out_shape = x.shape[:-1] + (self.qweight.shape[-1], )
#         reshaped_x = x.reshape(-1, x.shape[-1])
#         output = torch.empty((reshaped_x.shape[0], self.qweight.shape[-1]),
#                              dtype=torch.float16,
#                              device=x.device)
#         quantization_ops.gemm_half_q_half(reshaped_x, self.q4, output, False)
#         if bias is not None:
#             output = output + bias
#         return output.reshape(out_shape)

#     def temp_dq_size(self):
#         return self.input_size * self.output_size_per_partition * 2 + 128

#     def temp_fwd_size(self, max_tokens):
#         return self.output_size_per_partition * max_tokens * 4 + 128

#     def scratch_space_fixed(self, max_tokens):
#         return self.temp_dq_size() + self.temp_fwd_size(max_tokens)


# class GPTQRowParallelLinear(RowParallelLinear):

#     def create_weights(self, dtype: torch.dtype) -> None:
#         assert (self.input_size_per_partition %
#                 self.quant_config.pack_factor == 0)
#         assert self.output_size % self.quant_config.pack_factor == 0
#         group_size = self.quant_config.group_size if (
#             self.quant_config.group_size != -1
#         ) else self.input_size_per_partition
#         if self.tp_size > 1 and (self.quant_config.desc_act
#                                  and self.quant_config.group_size != -1):
#             group_number = self.input_size // group_size
#             self.use_exllama = False
#         else:
#             group_number = self.input_size_per_partition // group_size
#             self.use_exllama = True
#         self.qweight = Parameter(
#             torch.empty(
#                 self.input_size_per_partition // self.quant_config.pack_factor,
#                 self.output_size,
#                 device="cuda",
#                 dtype=torch.int32,
#             ),
#             requires_grad=False,
#         )
#         self.qzeros = Parameter(
#             torch.empty(
#                 group_number,
#                 self.output_size // self.quant_config.pack_factor,
#                 device="cuda",
#                 dtype=torch.int32,
#             ),
#             requires_grad=False,
#         )
#         self.scales = Parameter(
#             torch.empty(
#                 group_number,
#                 self.output_size,
#                 device="cuda",
#                 dtype=dtype,
#             ),
#             requires_grad=False,
#         )
#         self.g_idx = Parameter(
#             torch.tensor(
#                 [
#                     i // group_size
#                     for i in range(self.input_size_per_partition)
#                 ],
#                 device="cuda",
#                 dtype=torch.int32,
#             ),
#             requires_grad=False,
#         )

#     def post_init(self, temp_dq):
#         if not self.use_exllama:
#             return
#         assert self.qweight.device.type == "cuda"
#         assert self.qweight.device.index is not None

#         none_tensor = torch.empty((1, 1), device="meta")
#         temp_dq = temp_dq.get_scratch_slice(self.temp_dq_size())
#         if not self.quant_config.desc_act:
#             self.q4 = quantization_ops.make_q_matrix(
#                 self.qweight,
#                 none_tensor,
#                 none_tensor,
#                 self.qzeros,
#                 self.scales,
#                 none_tensor,
#                 temp_dq,
#             )
#         else:
#             self.q_perm = torch.empty((self.input_size, ),
#                                       dtype=torch.short,
#                                       device=self.qweight.device)
#             self.q_invperm = torch.empty_like(self.q_perm)
#             self.q4 = quantization_ops.make_q_matrix(
#                 self.qweight,
#                 self.q_perm,
#                 self.q_invperm,
#                 self.qzeros,
#                 self.scales,
#                 self.g_idx.cpu(),
#                 temp_dq,
#             )

#     def apply_weights(self, x: torch.Tensor) -> torch.Tensor:
#         out_shape = x.shape[:-1] + (self.qweight.shape[-1], )
#         reshaped_x = x.reshape(-1, x.shape[-1])

#         if self.use_exllama:
#             output = torch.empty((reshaped_x.shape[0], self.qweight.shape[-1]),
#                                  dtype=torch.float16,
#                                  device=x.device)
#             quantization_ops.gemm_half_q_half(reshaped_x, self.q4, output,
#                                               False)
#         else:
#             output = torch.zeros((reshaped_x.shape[0], self.qweight.shape[-1]),
#                                  dtype=torch.float32,
#                                  device=x.device)
#             quantization_ops.gptq_descact_matmul(reshaped_x.float(),
#                                                  self.qweight, output,
#                                                  self.scales.float(),
#                                                  self.qzeros, self.g_idx)
#             output = output.half()
#         return output.reshape(out_shape)

#     def temp_dq_size(self):
#         if not self.use_exllama:
#             return 0
#         return self.input_size_per_partition * self.output_size * 2 + 128

#     def temp_fwd_size(self, max_tokens):
#         if not self.use_exllama:
#             return 0
#         return self.output_size * max_tokens * 4 + 128

#     def scratch_space_fixed(self, max_tokens):
#         if not self.use_exllama:
#             return 0
#         return self.temp_dq_size() + self.temp_fwd_size(max_tokens)
