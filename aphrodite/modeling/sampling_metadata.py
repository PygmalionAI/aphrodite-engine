from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, TypeVar, Callable
import random

import torch

from aphrodite.modeling.layers.ops.sample import (get_num_triton_sampler_splits
                                                  )
from aphrodite.common.sampling_params import SamplingParams, SamplingType
from aphrodite.common.sequence import SequenceData
from aphrodite.common.utils import is_pin_memory_available

_SEED_0_REPLACEMENT = 3403598558  # chosen by fair roll of a die


class PersistentMetadata:

    def __init__(self, metadata: Optional[Dict[int, dict]] = None):
        self._metadata: Dict[int, dict] = metadata or {}

    def get(self, seq_id: int, key, default=None):
        return self._metadata.get(seq_id, {}).get(key, default)


class OutputMetadata():
    """Not symmetrical with PersistentMetadata because the process of
    sampling can produce unique metadata per sample, per sequence.
    
    The appropriate conversion would be `output[seq][sample](dict)` to
    `persist[new_seq_for_sample](dict)`"""

    def __init__(self):
        self._metadata: Dict[int, Dict[int, dict]] = {}

    def add(self, seq_id: int, sample_id: int, key, val) -> None:
        (self._metadata.setdefault(seq_id, {}).setdefault(sample_id,
                                                          {})[key]) = val

    def get(self, seq_id: int, sample_id: int) -> dict:
        return self._metadata.get(seq_id, {}).get(sample_id, {})


class SamplingMetadata:
    """Metadata for input sequences. Used in sampler.

    Args:
        seq_groups: List of (seq_ids, sampling_params).
        seq_data: Seq_id -> SequenceData.
        prompt_lens: Lengths of prompts.
        selected_token_indices: Token indices selected for sampling.
        categorized_sample_indices: SamplingType -> token indices to sample.
        generators: List of torch.Generators to use for seeded sampling
        perform_sampling: Whether to perform sampling. This option is used to
            make the sampling only happens in the driver worker, and disable
            sampling in other worker processes.
        persistent_metadata: Metadata that persists across iterations.
        output_metadata: the output metadata.
    """

    def __init__(
        self,
        seq_groups: Optional[List[Tuple[List[int], SamplingParams]]],
        seq_data: Optional[Dict[int, SequenceData]],
        prompt_lens: Optional[List[int]],
        selected_token_indices: torch.Tensor,
        categorized_sample_indices: Optional[Dict[SamplingType, torch.Tensor]],
        generators: Optional[List[torch.Generator]] = None,
        perform_sampling: bool = True,
        persistent_metadata: Optional[PersistentMetadata] = None,
        output_metadata: Optional[OutputMetadata] = None,
    ) -> None:
        self.seq_groups = seq_groups
        self.seq_data = seq_data
        self.prompt_lens = prompt_lens
        self.selected_token_indices = selected_token_indices
        self.categorized_sample_indices = categorized_sample_indices
        self.generators = generators
        self.perform_sampling = perform_sampling
        self.persistent_metadata = persistent_metadata or PersistentMetadata()
        self.output_metadata = output_metadata or OutputMetadata()

        self.num_prompts = len(prompt_lens) if prompt_lens is not None else 0

    def __repr__(self) -> str:
        return (
            "SamplingMetadata("
            f"seq_groups={self.seq_groups}, "
            f"seq_data={self.seq_data}, "
            f"prompt_lens={self.prompt_lens}, "
            f"selected_token_indices={self.selected_token_indices}, "
            f"categorized_sample_indices={self.categorized_sample_indices}, "
            f"perform_sampling={self.perform_sampling}, "
            f"persistent_metadata={self.persistent_metadata}, "
            f"output_metadata={self.output_metadata}) ")


@dataclass
class SamplingTensors:
    """Tensors for sampling."""
    temperatures: torch.Tensor
    top_ps: torch.Tensor
    top_ks: torch.Tensor
    top_as: torch.Tensor
    min_ps: torch.Tensor
    pres_penalties: torch.Tensor
    freq_penalties: torch.Tensor
    rep_penalties: torch.Tensor
    tfss: torch.Tensor
    eta_cutoffs: torch.Tensor
    epsilon_cutoffs: torch.Tensor
    typical_ps: torch.Tensor
    miro_taus: torch.Tensor
    miro_etas: torch.Tensor
    miro_mus: torch.Tensor
    miro_indices: torch.Tensor
    miro_seqids: List[int]  # state writeback done CPU side
    dynatemp_mins: torch.Tensor
    dynatemp_maxs: torch.Tensor
    dynatemp_exps: torch.Tensor
    smoothing_indices: torch.Tensor
    smoothing_factors: torch.Tensor
    smoothing_curves: torch.Tensor

    seed_indices: torch.Tensor
    seed_transpose: torch.Tensor
    extra_seed_transpose: Optional[torch.Tensor]

    prompt_tokens: torch.Tensor
    output_tokens: torch.Tensor

    do_temperatures: bool
    do_dynatemps: bool
    do_penalties: bool
    do_top_ks: bool
    do_top_ps: bool
    do_top_as: bool
    do_min_ps: bool
    do_tfss: bool
    do_eta_cutoffs: bool
    do_epsilon_cutoffs: bool
    do_typical_ps: bool
    do_quadratic: bool
    do_mirostat: bool

    @classmethod
    def from_sampling_metadata(
            cls,
            sampling_metadata: "SamplingMetadata",
            vocab_size: int,
            tgt_device: torch.device,
            float_dtype: torch.dtype,
            *,
            extra_seeds_to_generate: int = 0,
            extra_entropy: Optional[Tuple[int,
                                          ...]] = None) -> "SamplingTensors":
        prompt_lens = sampling_metadata.prompt_lens or []
        groups = sampling_metadata.seq_groups or []
        seq_data = sampling_metadata.seq_data or {}
        persistent = sampling_metadata.persistent_metadata
        extra_entropy = extra_entropy or ()

        # Flattened list of (params, sid) matching the logits tensor.
        # `sid < 0` implies a prompt seq.
        unrolled_seqs: List[Tuple[SamplingParams, int]] = []
        group_plens = prompt_lens + [0] * (len(groups) - len(prompt_lens))
        for (ids, params), prompt_len in zip(groups, group_plens):
            if prompt_len and params.prompt_logprobs is not None:
                unrolled_seqs.extend([(params, -1)] * (prompt_len - 1))
            unrolled_seqs.extend([(params, sid) for sid in ids])

        T = TypeVar('T')

        def _unroll(fn_val: Callable[[SamplingParams], T],
                    prompt: Optional[T] = None) -> List[T]:
            """`fn_val` for every seq, with an override for prompt seqs."""
            return [
                prompt if sid < 0 and prompt is not None else fn_val(p)
                for p, sid in unrolled_seqs
            ]

        def _index(fn_mask: Callable[[SamplingParams], bool],
                   prompt: Optional[bool] = None) -> List[int]:
            """Index for every seq where `fn_mask` is true, with an override
            for prompt seqs."""
            return [
                i for i, (p, sid) in enumerate(unrolled_seqs)
                if (fn_mask(p) if prompt is None else (
                    prompt if sid < 0 else fn_mask(p)))
            ]

        def _filter(arr: List[T], indices: List[int]) -> List[T]:
            """Return only the elements of `arr` accessed by `indices`."""
            return [arr[i] for i in indices]

        miro_inds = _index(lambda p: p.mirostat_mode == 2, prompt=False)
        _miro_seqs = _filter(unrolled_seqs, miro_inds)

        quad_inds = _index(lambda p: p.smoothing_factor != 0)
        _quad_seqs = _filter(unrolled_seqs, quad_inds)

        # We need one base seed per Triton slice.
        triton_sampler_splits = get_num_triton_sampler_splits(vocab_size)
        n_seeds = triton_sampler_splits + extra_seeds_to_generate

        # Sequences get seeds. Prompt "sequences" do not.
        seed_indices = _index(lambda p: True, prompt=False)
        sampling_seeds = [
            cls._get_sequence_seeds(p.seed, n_seeds,
                                    p.sampling_type == SamplingType.GREEDY,
                                    seq_data[sid].get_len(), *extra_entropy,
                                    sid)
            for p, sid in _filter(unrolled_seqs, seed_indices)
        ]

        fvars = {  # noqa
            "temperatures": _unroll(lambda p: p.temperature),
            "top_ps": _unroll(lambda p: p.top_p),
            "top_as": _unroll(lambda p: p.top_a),
            "min_ps": _unroll(lambda p: p.min_p),
            "tfss": _unroll(lambda p: p.tfs, prompt=1),
            "eta_cutoffs": _unroll(lambda p: p.eta_cutoff * 1e-4, prompt=0),
            "epsilon_cutoffs": _unroll(lambda p: p.epsilon_cutoff * 1e-4, 0),
            "typical_ps": _unroll(lambda p: p.typical_p, prompt=1),
            "pres_penalties": _unroll(lambda p: p.presence_penalty, prompt=0),
            "freq_penalties": _unroll(lambda p: p.frequency_penalty, prompt=0),
            "rep_penalties": _unroll(lambda p: p.repetition_penalty, prompt=1),

            "dynatemp_mins": _unroll(lambda p: p.dynatemp_min),
            "dynatemp_maxs": _unroll(lambda p: p.dynatemp_max),
            "dynatemp_exps": _unroll(lambda p: p.dynatemp_exponent),

            "miro_taus": [p.mirostat_tau for p, _ in _miro_seqs],
            "miro_etas": [p.mirostat_eta for p, _ in _miro_seqs],
            "miro_mus": [persistent.get(sid, "miro_mu", p.mirostat_tau * 2)
                         for p, sid in _miro_seqs],

            "smoothing_factors": [p.smoothing_factor for p, _ in _quad_seqs],
            "smoothing_curves": [p.smoothing_curve for p, _ in _quad_seqs],
        }
        ivars = {  # noqa
            "top_ks": _unroll(lambda p: vocab_size
                              if p.top_k == -1 else min(p.top_k, vocab_size)),
            "miro_indices": miro_inds,
            "smoothing_indices": quad_inds,
            "seed_indices": seed_indices,
        }

        prompt_tokens = [[] if sid < 0 else seq_data[sid].prompt_token_ids
                         for _, sid in unrolled_seqs]
        output_tokens = [[] if sid < 0 else seq_data[sid].output_token_ids
                         for _, sid in unrolled_seqs]

        # need to transpose and make contiguous to copy the tensor correctly.
        # [batch_size, n_seeds] -> [n_seeds, batch_size]
        seeds_transpose = list(map(list, zip(*sampling_seeds)))
        seeds_gpu = seeds_transpose[:triton_sampler_splits]
        extra_seeds_gpu = seeds_transpose[triton_sampler_splits:] or None

        # Note that the performance will be very bad without pinned memory.
        # Pinned memory allows non-blocking transfers to device.
        pin_memory = is_pin_memory_available()

        def _tensor(contents: list, dtype) -> torch.Tensor:
            loc_t = torch.tensor(contents,
                                 dtype=dtype,
                                 device="cpu",
                                 pin_memory=pin_memory)
            return loc_t.to(device=tgt_device, non_blocking=True)

        def _unjagged(arrs: List[List[T]], padval: T) -> List[List[T]]:
            max_len = max(len(arr) for arr in arrs)
            return [arr + [padval] * (max_len - len(arr)) for arr in arrs]

        return cls(
            #  Flags and non-tensor fields
            do_temperatures=any(x != 1 for x in fvars["temperatures"]),
            do_dynatemps=(any(fvars["dynatemp_mins"])
                          or any(fvars["dynatemp_maxs"])),
            do_top_ks=any(x != vocab_size for x in ivars["top_ks"]),
            do_top_ps=any(x != 1 for x in fvars["top_ps"]),
            do_top_as=any(fvars["top_as"]),
            do_min_ps=any(fvars["min_ps"]),
            do_tfss=any(x != 1 for x in fvars["tfss"]),
            do_eta_cutoffs=any(fvars["eta_cutoffs"]),
            do_epsilon_cutoffs=any(fvars["epsilon_cutoffs"]),
            do_typical_ps=any(x != 1 for x in fvars["typical_ps"]),
            do_penalties=(any(fvars["pres_penalties"])
                          or any(fvars["freq_penalties"])
                          or any(x != 1 for x in fvars["rep_penalties"])),
            do_quadratic=len(quad_inds) > 0,
            do_mirostat=len(miro_inds) > 0,
            miro_seqids=_filter([s for _, s in unrolled_seqs], miro_inds),
            # Float tensors
            **{n: _tensor(vals, float_dtype)
               for n, vals in fvars.items()},
            # Integer tensors
            **{n: _tensor(vals, torch.int)
               for n, vals in ivars.items()},
            # Token ID tensors
            prompt_tokens=_tensor(_unjagged(prompt_tokens, vocab_size),
                                  torch.long),
            output_tokens=_tensor(_unjagged(output_tokens, vocab_size),
                                  torch.long),
            # Seeds (only for triton, though?)
            seed_transpose=_tensor(seeds_gpu, torch.long),
            extra_seed_transpose=(_tensor(extra_seeds_gpu, torch.long)
                                  if extra_seeds_gpu else None),
        )

    @staticmethod
    def _get_sequence_seeds(
        seed: Optional[int],
        seeds_to_generate: int,
        is_greedy: bool,
        *extra_entropy: int,
    ):
        """Get `seeds_to_generate` child seeds from `seed` and extra entropy."""
        if is_greedy:  # For the kernel, seed == 0 means greedy decoding.
            return [0] * seeds_to_generate

        if seed is None:
            randint_fn = random.randint
        else:
            randint_fn = random.Random(str((seed, ) + extra_entropy)).randint

        lo, hi = torch.iinfo(torch.long).min, torch.iinfo(torch.long).max
        # If the user/random sets seed = 0 but request should
        # have sampling, we need to change it to something
        # else. We use a constant in that case.
        # This way we don't need to create and load a bool
        # matrix in the sampling kernel, which reduces CPU
        # overhead and latency.
        return [
            randint_fn(lo, hi) or _SEED_0_REPLACEMENT
            for _ in range(seeds_to_generate)
        ]
