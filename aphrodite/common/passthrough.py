from typing import Optional, Union
from typing_extensions import TypedDict


class Passthrough(TypedDict):
    """
    (development only) arguments passed through to the model.forward call from the API
    """
    foo: int
    bar: str

def try_get_passthrough(params: Optional[Union["SamplingParams", "PoolingParams"]]) -> Optional[Passthrough]:
    passthrough = None
    if isinstance(params, SamplingParams):
        passthrough = params.passthrough
    elif isinstance(params, PoolingParams) and isinstance(params.additional_data, dict):
        passthrough = params.additional_data.get("passthrough")
    return passthrough
from aphrodite.common.pooling_params import PoolingParams
from aphrodite.common.sampling_params import SamplingParams
