import aphrodite.modeling.megatron.parallel_state
import aphrodite.modeling.megatron.tensor_parallel

# Alias parallel_state as mpu, its legacy name
mpu = parallel_state

__all__ = [
    "parallel_state",
    "tensor_parallel",
]