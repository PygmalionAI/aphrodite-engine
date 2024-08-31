import functools
from typing import Dict


@functools.lru_cache
def _get_op_configs(op_type: str, batch: int, hidden_size: int):
    # TODO: add optimal configs
    return None


def check_divisbility(hidden_size: int):
    