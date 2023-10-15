# Mirostat is a special snowflake that gets it's own file
# In order to isolate it's state from the rest of the system
# And make hooks more tidy

from typing import Dict, List
from aphrodite.common.sequence import Sequence
from aphrodite.modeling.metadata import InputMetadata

_mirostat_mus: Dict[int, float] = dict()

def mirostat_get_mu_hook(input_metadata: InputMetadata) -> List[float]:
    mus: List[float] = []
    for seq_ids, params in input_metadata.seq_groups:
        tmp = [_mirostat_mus[id] if id in _mirostat_mus else 2*params.mirostat_tau for id in seq_ids]
        mus += tmp
    
    return mus


def mirostat_update_mu_hook(
    input_metadata: InputMetadata,
    mus: List[float]
) -> None:
    ids = [id for seq_ids, _ in input_metadata.seq_groups for id in seq_ids]
    for id, val in zip(ids, mus):
        _mirostat_mus[id] = val


def mirostat_delete_seq_hook(seq: Sequence) -> None:
    if seq.seq_id in _mirostat_mus:
        del _mirostat_mus[seq.seq_id]