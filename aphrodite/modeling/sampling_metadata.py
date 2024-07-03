import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, TypeVar

import torch

from aphrodite.common.sampling_params import SamplingParams, SamplingType
from aphrodite.common.sequence import SequenceData, SequenceGroupMetadata
from aphrodite.common.utils import (async_tensor_h2d, is_pin_memory_available,
                                    maybe_expand_dim)
from aphrodite.modeling.layers.ops.sample import get_num_triton_sampler_splits

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


@dataclass
class SequenceGroupToSample:
    # Sequence ids for the sequence group in a previous step.
    seq_ids: List[int]
    sampling_params: SamplingParams
    # seq_id -> sequence data.
    seq_data: Dict[int, SequenceData]
    # The length of the prompt of the sequence group. None if it is in a decode
    # stage.
    prompt_len: Optional[int]
    # The length of the query tokens to compute in the current step. None if it
    # is in a decode stage. The length of subquery_len <= prompt_len.
    subquery_len: Optional[int]
    # A random number generator for sampling.
    generator: Optional[torch.Generator]
    # True if the sequence group is in prefill stage. False if it is in a
    # decode stage.
    is_prompt: bool
    # Query token indices from logits. to compute prompt logprob. Empty if
    # prompt logprob is not required.
    prompt_logprob_indices: List[int]
    # Sample token indices from logits. Empty if sampling is not required.
    sample_indices: List[int]

    @property
    def do_sample(self):
        return len(self.sample_indices) > 0

    def __post_init__(self):
        if len(self.prompt_logprob_indices) > 0:
            assert self.sampling_params.prompt_logprobs is not None
        if self.is_prompt:
            assert self.prompt_len is not None
            assert self.subquery_len is not None

class SamplingMetadata:
    """Metadata for input sequences. Used in sampler.
    The usage is as follow;
    ```
    hidden_states = execute_model(...)
    logits = hidden_states[sampling_metadata.selected_token_indices]
    sample(logits)
    def sample(logits):
        # Use categorized_sample_indices for sampling....
    ```

    Args:
        seq_groups: List of batched sequence groups.
        selected_token_indices: (num_query_tokens_to_logprob). Indices to find
            logits from the initial model output hidden states.
        categorized_sample_indices: SamplingType -> token indices to sample.
            Each token indices is 2D tensor of (num_indices, num_indices) where
            the first item means the sample index within the returned logit
            (before pruning padding), and the second item means the sample
            index after pruning using selected_token_indices.
            For example, if the returned logit is [1, 2, 3], and we select
            [1, 2] for sampling, the pruned logit will be [2, 3]. In this case,
            The first tuple is [1, 2] (sampled index within original logit),
            and the second tuple is [0, 1] (sampled index within pruned logit).
        num_prompts: Number of prompt sequence groups in seq_groups.
        persistent_metadata: Metadata that persists across iterations.
        output_metadata: the output metadata.
    """

    def __init__(
        self,
        seq_groups: List[SequenceGroupToSample],
        selected_token_indices: torch.Tensor,
        categorized_sample_indices: Dict[SamplingType, torch.Tensor],
        num_prompts: int,
        persistent_metadata: Optional[PersistentMetadata] = None,
        output_metadata: Optional[OutputMetadata] = None,
    ) -> None:
        self.seq_groups = seq_groups
        self.selected_token_indices = selected_token_indices
        self.categorized_sample_indices = categorized_sample_indices
        self.num_prompts = num_prompts
        self.persistent_metadata = persistent_metadata or PersistentMetadata()
        self.output_metadata = output_metadata or OutputMetadata()

    @staticmethod
    def prepare(
        seq_group_metadata_list: List[SequenceGroupMetadata],
        prompt_lens: List[int],
        subquery_lens: Optional[List[int]],
        device: str,
        pin_memory: bool,
    ) -> "SamplingMetadata":
        (
            seq_groups,
            selected_token_indices,
            categorized_sample_indices,
            num_prompts,
        ) = _prepare_seq_groups(seq_group_metadata_list, prompt_lens,
                                subquery_lens, device)
        selected_token_indices = async_tensor_h2d(selected_token_indices,
                                                  dtype=torch.long,
                                                  target_device=device,
                                                  pin_memory=pin_memory)
        categorized_sample_indices = {
            t: maybe_expand_dim(
                async_tensor_h2d(seq_ids,
                                 dtype=torch.int,
                                 target_device=device,
                                 pin_memory=pin_memory), 2, 2)
            for t, seq_ids in categorized_sample_indices.items()
        }

        sampling_metadata = SamplingMetadata(
            seq_groups=seq_groups,
            selected_token_indices=selected_token_indices,
            categorized_sample_indices=categorized_sample_indices,
            num_prompts=num_prompts,
        )
        return sampling_metadata

    def __repr__(self) -> str:
        return (
            "SamplingMetadata("
            f"seq_groups={self.seq_groups}, "
            f"selected_token_indices={self.selected_token_indices}, "
            f"categorized_sample_indices={self.categorized_sample_indices}, "
            f"persistent_metadata={self.persistent_metadata}, "
            f"output_metadata={self.output_metadata}) ")


def _prepare_seq_groups(
    seq_group_metadata_list: List[SequenceGroupMetadata],
    prompt_lens: List[int],
    subquery_lens: Optional[List[int]],
    device: str,
) -> Tuple[List[SequenceGroupToSample], List[int], Dict[
        SamplingType, List[Tuple[int, int]]], int]:
    """Prepare sequence groups and indices for sampling.
    Args:
        seq_group_metadata_list: A list of sequence group to batch.
        prompt_lens: A list of prompt lens per sequence group.
            Index of prompt len should match with seq_group_metadata_list.
        subquery_lens: A list of query lengths. Prompt lens include the length
            of entire prompt tokens, and it could be shorter.
        device: A device to use for random number generator,
            `SequenceGroupToSample.generator`.
    Returns:
        seq_groups: A list of sequence group to sample.
        selected_token_indices: See the definition from `SamplingMetadata`.
        categorized_sample_indices: See the definition from `SamplingMetadata`.
        num_prompts: Total number of prompts from `seq_group_metadata_list`.
    """
    # Batched sequence groups for the current model forward stsep.
    seq_groups: List[SequenceGroupToSample] = []
    # A list of token indices to sample/compute logprob. It is used to
    # prune the outcome logits from the model for the performance.
    selected_token_indices: List[int] = []
    # Used for selected_token_indices.
    model_output_idx = 0

    # Sampling type -> (
    # indices to sample/prompt logprob within pruned output logits,
    # indices to sample within pruned logits)
    categorized_sample_indices: Dict[SamplingType, List[Tuple[int, int]]] = {
        t: []
        for t in SamplingType
    }
    # Index of logits to compute logprob. Logits include both prompt logprob
    # and sample logprob indices.
    logit_idx = 0
    # Index to sample from a sample tensor. It is used by triton sample kernel.
    # See `_sample_with_triton_kernel` for more details.
    sample_idx = 0
    # Total number of prompts from given sequence groups.
    num_prompts = 0

    for i, seq_group_metadata in enumerate(seq_group_metadata_list):
        seq_ids = list(seq_group_metadata.seq_data.keys())
        sampling_params = seq_group_metadata.sampling_params
        is_prompt = seq_group_metadata.is_prompt
        generator: Optional[torch.Generator] = None
        # If the current seq group is in decode stage, it is None.
        prompt_len: Optional[int] = None
        subquery_len: Optional[int] = None
        prompt_logprob_indices: List[int] = []
        sample_indices: List[int] = []
        do_sample = seq_group_metadata.do_sample

        if seq_group_metadata.is_prompt:
            if sampling_params.seed is not None:
                seq_group_metadata.state.generator = torch.Generator(
                    device=device).manual_seed(sampling_params.seed)

            num_prompts += 1
            num_prefill_sample = len(seq_ids)
            assert num_prefill_sample == 1
            assert subquery_lens is not None and prompt_lens is not None
            subquery_len, prompt_len = subquery_lens[i], prompt_lens[i]
            # If we need sampling, exclude num_prefill_sample tokens from
            # prompt logprob.
            prompt_logprob_len = (subquery_len - num_prefill_sample
                                  if do_sample else subquery_len)
            sample_len = num_prefill_sample if do_sample else 0
        else:
            # Decode
            prompt_logprob_len = 0
            sample_len = len(seq_ids) if do_sample else 0

        # Update indices to select from the model output.
        """
        This blocks computes selected_token_indices which is used in the
        following way.
        hidden_states = model(...)
        logits = hidden_states[selected_token_indices]
        """

        if sampling_params.prompt_logprobs:
            selected_token_indices.extend(
                range(model_output_idx, model_output_idx + prompt_logprob_len))
        model_output_idx += prompt_logprob_len
        if do_sample:
            selected_token_indices.extend(
                range(model_output_idx, model_output_idx + sample_len))
        model_output_idx += sample_len

        # We now find indices for logprob computation and sampling.
        """
        This block computes categorized_sample_indices which is used in the
        following way.
        hidden_states = model(...)
        logits = hidden_states[selected_token_indices]
        def sample(logits):
           # Use categorized_sample_indices for sampling.
           # prompt_logprob_indices to find prompt logprob indices.
           # sample_indices to find sample indices.
        """

        if sampling_params.prompt_logprobs is not None:
            prompt_logprob_indices.extend(
                range(logit_idx, logit_idx + prompt_logprob_len))
            logit_idx += prompt_logprob_len
        if do_sample:
            sample_indices.extend(range(logit_idx, logit_idx + sample_len))
            categorized_sample_indices[sampling_params.sampling_type].extend(
                list(
                    zip(range(logit_idx, logit_idx + sample_len),
                        range(sample_idx, sample_idx + sample_len))))
            logit_idx += sample_len
            sample_idx += sample_len

        if sampling_params.seed is not None:
            generator = seq_group_metadata.state.generator

        seq_groups.append(
            SequenceGroupToSample(
                seq_ids=seq_ids,
                sampling_params=sampling_params,
                seq_data=seq_group_metadata.seq_data,
                prompt_len=prompt_len,
                subquery_len=subquery_len,
                generator=generator,
                is_prompt=is_prompt,
                prompt_logprob_indices=list(prompt_logprob_indices),
                sample_indices=list(sample_indices)))
    return (seq_groups, selected_token_indices, categorized_sample_indices,
            num_prompts)


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
