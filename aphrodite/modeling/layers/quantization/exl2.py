from typing import Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from vllm import quantization_ops
from vllm.model_executor.parallel_utils.layers import (ColumnParallelLinear,
                                                       RowParallelLinear)
from vllm.model_executor.parallel_utils.communication_op import (
    tensor_model_parallel_all_gather)
from aphrodite.modeling.layers.quantization.base_config import (
    QuantizationConfig)

class EXL2Config(QuantizationConfig):
    """Config class for EXL2."""

    def __repr__(self) -> str:
        return "EXL2Config"

    @classmethod
    def get_name(cls) -> str:
        return "exl2"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half]

    @classmethod
    # Need to figure it out
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_packed_tensors(cls) -> Dict[str, int]:
        return {"qzeros": 1}

    @classmethod
    def get_transposed_tensor_names(cls) -> List[str]:
        return ["q_weight", "q_scale"]

    def get_row_parallel_tensor_names(self) -> List[str]:
        return []

    def get_col_parallel_tensor_names(self) -> List[str]:
        return ["q_weight", "q_scale", "bias"]

    @classmethod
    def merge_weight(cls) -> bool:
        return False

    @classmethod
    def quantize_vocab(cls) -> bool:
        return True


class EXL2Linear(torch.nn.Module):

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
        self.q_weight = Parameter(
            torch.empty(
                (1, self.output_size), device="cuda", dtype=torch.int32),
            requires_grad=False,
        )
        self.q_scale = Parameter(
            torch.empty(
                (1, self.output_size // 8), device="cuda", dtype=torch.int32),
            requires_grad=False,
        )
        self.q_groups = Parameter(
            torch.empty((1,), device="cuda", dtype=torch.short),
            requires_grad=False,
        )
        self.q_scale_max = Parameter(
            torch.empty((1,), device="cuda", dtype=torch.float16),
            requires_grad=False,
        )
        self.q_invperm = Parameter(
            torch.empty((1,), device="cuda", dtype=torch.int32),
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
        assert self.q_weight.device.type == "cuda"
        assert self.q_weight.device.index is not None

        none_tensor = torch.empty((1, 1), device="meta")
        temp_dq = temp_dq.get_scratch_slice(self.temp_dq_size())
        self.q_scale_max /= 256
        self.q_perm = torch.argsort(self.q_invperm).short()
        self.q_invperm_short = self.q_invperm.short()
        self.q4 = quantization_ops.make_q_matrix(
            self.q_weight,
            self.q_perm,
            self.q_invperm_short,
            self.q_scale,
            self.q_scale_max,
            self.q_groups,
            none_tensor,
            none_tensor,
            none_tensor,
            temp_dq,
        )

    def forward(self, input_):
        out_shape = input_.shape[:-1] + (self.q_weight.shape[-1], )
        reshaped_x = input_.reshape(-1, input_.shape[-1])
        output = torch.empty((reshaped_x.shape[0], self.q_weight.shape[-1]),
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


class EXL2ColumnParallelLinear(ColumnParallelLinear):

    def create_weights(self, dtype: torch.dtype) -> None:
        self.use_exllama = True

        self.q_weight = Parameter(
            torch.empty(
                (1, self.output_size_per_partition),
                device="cuda", dtype=torch.int32),
            requires_grad=False,
        )
        self.q_scale = Parameter(
            torch.empty(
                (1, self.output_size_per_partition // 8),
                device="cuda", dtype=torch.int32),
            requires_grad=False,
        )
        self.q_groups = Parameter(
            torch.empty((1,), device="cuda", dtype=torch.short),
            requires_grad=False,
        )
        self.q_scale_max = Parameter(
            torch.empty((1,), device="cuda", dtype=torch.float16),
            requires_grad=False,
        )
        self.q_invperm = Parameter(
            torch.empty((1,), device="cuda", dtype=torch.int32),
            requires_grad=False,
        )

    def post_init(self, temp_dq):
        assert self.q_weight.device.type == "cuda"
        assert self.q_weight.device.index is not None

        none_tensor = torch.empty((1, 1), device="meta")
        temp_dq = temp_dq.get_scratch_slice(self.temp_dq_size())
        self.q_scale_max /= 256
        self.q_perm = torch.argsort(self.q_invperm).short()
        self.q_invperm_short = self.q_invperm.short()
        self.q4 = quantization_ops.make_q_matrix(
            self.q_weight,
            self.q_perm,
            self.q_invperm_short,
            self.q_scale,
            self.q_scale_max,
            self.q_groups,
            none_tensor,
            none_tensor,
            none_tensor,
            temp_dq,
        )

    def apply_weights(
        self,
        x: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        out_shape = x.shape[:-1] + (self.q_weight.shape[-1], )
        reshaped_x = x.reshape(-1, x.shape[-1])
        output = torch.empty((reshaped_x.shape[0], self.q_weight.shape[-1]),
                             dtype=torch.float16,
                             device=x.device)
        quantization_ops.gemm_half_q_half(reshaped_x, self.q4, output, False)
        if bias is not None:
            output = output + bias
        return output.reshape(out_shape)

    def temp_dq_size(self):
        return self.input_size * self.output_size_per_partition * 2 + 128

    def temp_fwd_size(self, max_tokens):
        return self.output_size_per_partition * max_tokens * 4 + 128

    def scratch_space_fixed(self, max_tokens):
        return self.temp_dq_size() + self.temp_fwd_size(max_tokens)


class EXL2RowParallelLinear(RowParallelLinear):

    def create_weights(self, dtype: torch.dtype) -> None:
        self.use_exllama = True
        self.q_weight = Parameter(
            torch.empty(
                (1, self.output_size), device="cuda", dtype=torch.int32),
            requires_grad=False,
        )
        self.q_scale = Parameter(
            torch.empty(
                (1, self.output_size // 8), device="cuda", dtype=torch.int32),
            requires_grad=False,
        )
        self.q_groups = Parameter(
            torch.empty((1,), device="cuda", dtype=torch.short),
            requires_grad=False,
        )
        self.q_scale_max = Parameter(
            torch.empty((1,), device="cuda", dtype=torch.float16),
            requires_grad=False,
        )
        self.q_invperm = Parameter(
            torch.empty((1,), device="cuda", dtype=torch.int32),
            requires_grad=False,
        )

    def post_init(self, temp_dq):
        assert self.q_weight.device.type == "cuda"
        assert self.q_weight.device.index is not None

        none_tensor = torch.empty((1, 1), device="meta")
        temp_dq = temp_dq.get_scratch_slice(self.temp_dq_size())
        self.q_scale_max /= 256
        self.q_perm = torch.argsort(self.q_invperm).short()
        self.q_invperm_short = self.q_invperm.short()
        self.q4 = quantization_ops.make_q_matrix(
            self.q_weight,
            self.q_perm,
            self.q_invperm_short,
            self.q_scale,
            self.q_scale_max,
            self.q_groups,
            none_tensor,
            none_tensor,
            none_tensor,
            temp_dq,
        )

    def apply_weights(self, x: torch.Tensor) -> torch.Tensor:
        out_shape = x.shape[:-1] + (self.q_weight.shape[-1], )
        reshaped_x = x.reshape(-1, x.shape[-1])

        output = torch.empty((reshaped_x.shape[0], self.q_weight.shape[-1]),
                             dtype=torch.float16,
                             device=x.device)
        quantization_ops.gemm_half_q_half(reshaped_x, self.q4, output,
                                          False)
        return output.reshape(out_shape)

    def forward(self, input_):
        if self.input_is_parallel:
            input_ = tensor_model_parallel_all_gather(input_)
        output_ = self.apply_weights(input_)
        if not self.reduce_results and self.world_size > 1:
            output_ = output_ / self.world_size

        if not self.skip_bias_add:
            output = output_ + self.bias if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return output, output_bias

    def temp_dq_size(self):
        return self.input_size * self.output_size * 2 + 128

    def temp_fwd_size(self, max_tokens):
        return self.output_size * max_tokens * 4 + 128

    def scratch_space_fixed(self, max_tokens):
        return self.temp_dq_size() + self.temp_fwd_size(max_tokens)