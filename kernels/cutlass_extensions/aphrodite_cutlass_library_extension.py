import enum
from typing import Dict, Union

from cutlass_library import *

#
#   Extend cutlass library with custom types, and missing values
#


class APHRODITEDataType(enum.Enum):
    u4b8 = enum_auto()
    u8b128 = enum_auto()


class MixedInputKernelScheduleType(enum.Enum):
    TmaWarpSpecializedMixedInput = enum_auto()
    TmaWarpSpecializedPingpongMixedInput = enum_auto()
    TmaWarpSpecializedCooperativeMixedInput = enum_auto()


APHRODITEDataTypeNames: Dict[Union[APHRODITEDataType, DataType], str] = {
    **DataTypeNames,  # type: ignore
    **{
        APHRODITEDataType.u4b8: "u4b8",
        APHRODITEDataType.u8b128: "u8b128",
    }
}

APHRODITEDataTypeTag: Dict[Union[APHRODITEDataType, DataType], str] = {
    **DataTypeTag,  # type: ignore
    **{
        APHRODITEDataType.u4b8: "cutlass::aphrodite_uint4b8_t",
        APHRODITEDataType.u8b128: "cutlass::aphrodite_uint8b128_t",
    }
}

APHRODITEKernelScheduleTag: Dict[Union[
    MixedInputKernelScheduleType, KernelScheduleType], str] = {
        **KernelScheduleTag,  # type: ignore
        **{
            MixedInputKernelScheduleType.TmaWarpSpecializedMixedInput:
            "cutlass::gemm::KernelTmaWarpSpecializedMixedInput",
            MixedInputKernelScheduleType.TmaWarpSpecializedPingpongMixedInput:
            "cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput",
            MixedInputKernelScheduleType.TmaWarpSpecializedCooperativeMixedInput:
            "cutlass::gemm::KernelTmaWarpSpecializedCooperativeMixedInput",
        }
    }
