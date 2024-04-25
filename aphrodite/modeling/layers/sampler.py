"""A layer that samples the next tokens from the model's outputs."""
import itertools
import math
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn

from aphrodite.modeling.sampling_metadata import (SamplingMetadata,
                                                  OutputMetadata,
                                                  SamplingTensors)
from aphrodite.common.sampling_params import SamplingParams, SamplingType
from aphrodite.common.sequence import (Logprob, PromptLogprobs, SampleLogprobs,
                                       SamplerOutput, SequenceData,
                                       SequenceGroupOutput, SequenceOutput)
from aphrodite.modeling.layers.ops.sample import sample as sample_triton


class Sampler(nn.Module):
    """Samples the next tokens from the model's outputs.
    This layer does the following:
    1. Discard the hidden states that are not used for sampling (i.e., all
        tokens except the final one in each prompt).
    2. Compute the logits for the next tokens.
    3. Apply all the different sampler functions in the specified order.
    4. Sample the next tokens.
    Here, each sequence group within the batch can have different sampling
    parameters (e.g., sampling method, temperature, top-p, top-k, etc.).
    The structure of the logits tensor is coupled with the seq_groups in
    sampling_metadata. Typically, each sequence in each seq_group has one row in
    logits for the next token to be sampled; however, for a seq_group with a
    prompt request with the prompt_logprobs sampling parameter, there are rows
    in logits for each token in the input prompt.
    """

    def __init__(self):
        super().__init__()

        # Whether or not the SamplerOutput should have on-device tensors
        # containing the sampled token ids and probabilities. This is used by
        # speculative decoding.
        self.include_gpu_probs_tensor = False

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        assert logits is not None
        _, vocab_size = logits.shape
        output_metadata = OutputMetadata()
        # Apply min_tokens penalty which sets stop tokens to -inf if min_tokens
        # have not been generated yet
        logits = _apply_min_tokens_penalty(logits, sampling_metadata)

        # Prepare sampling tensors with pinned memory to avoid blocking.
        sampling_tensors = SamplingTensors.from_sampling_metadata(
            sampling_metadata, vocab_size, logits.device, logits.dtype)

        if sampling_tensors.do_penalties:
            logits = _apply_penalties(logits, sampling_tensors.prompt_tokens,
                                      sampling_tensors.output_tokens,
                                      sampling_tensors.pres_penalties,
                                      sampling_tensors.freq_penalties,
                                      sampling_tensors.rep_penalties)

        if sampling_tensors.do_temperatures or sampling_tensors.do_dynatemps:
            logits = _apply_temperature(logits, sampling_tensors.temperatures,
                                        sampling_tensors.dynatemp_mins,
                                        sampling_tensors.dynatemp_maxs,
                                        sampling_tensors.dynatemp_exps)

        if (sampling_tensors.do_top_ks or sampling_tensors.do_top_ps
                or sampling_tensors.do_top_as or sampling_tensors.do_min_ps):
            logits = _apply_alphabet_soup(logits, sampling_tensors.top_ps,
                                          sampling_tensors.top_ks,
                                          sampling_tensors.top_as,
                                          sampling_tensors.min_ps)
        if sampling_tensors.do_tfss:
            logits = _apply_tfs(logits, sampling_tensors.tfss)
        if sampling_tensors.do_eta_cutoffs:
            logits = _apply_eta_cutoff(logits, sampling_tensors.eta_cutoffs)
        if sampling_tensors.do_epsilon_cutoffs:
            logits = _apply_epsilon_cutoff(logits,
                                           sampling_tensors.epsilon_cutoffs)
        if sampling_tensors.do_typical_ps:
            logits = _apply_typical_sampling(logits,
                                             sampling_tensors.typical_ps)

        if sampling_tensors.do_quadratic:
            logits = _apply_quadratic_sampling(
                logits, sampling_tensors.smoothing_indices,
                sampling_tensors.smoothing_factors,
                sampling_tensors.smoothing_curves)

        banned_tokens = _get_custom_token_bans(sampling_metadata)
        assert len(banned_tokens) == logits.shape[0]
        logits = _apply_token_bans(logits, banned_tokens)
        if sampling_tensors.do_mirostat:
            logits = _apply_mirostat_v2(logits, sampling_tensors)

        # We use float32 for probabilities and log probabilities.
        # Compute the probabilities.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # Compute the log probabilities.
        # Use log_softmax to ensure numerical stability.
        logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)

        # Sample the next tokens.
        # sample_results = _sample(probs, logprobs, sampling_metadata,
        #                          sampling_tensors)
        sample_results, maybe_sampled_tokens_tensor = _sample(
            probs,
            logprobs,
            sampling_metadata,
            sampling_tensors,
            include_gpu_probs_tensor=self.include_gpu_probs_tensor,
            modify_greedy_probs=self._should_modify_greedy_probs_inplace,
        )

        if self.include_gpu_probs_tensor:
            assert maybe_sampled_tokens_tensor is not None
            sampled_tokens_tensor = maybe_sampled_tokens_tensor
            on_device_tensors = (probs, sampled_tokens_tensor)
        else:
            on_device_tensors = None

        if sampling_tensors.do_mirostat:
            _mirostat_store_args(logits, sampling_tensors, sample_results,
                                 sampling_metadata, output_metadata)
        # Get the logprobs query results.
        prompt_logprobs, sample_logprobs = _get_logprobs(
            logprobs, sampling_metadata, sample_results)
        # return _build_sampler_output(sample_results, sampling_metadata,
        #                              prompt_logprobs, sample_logprobs,
        #                              output_metadata)
        return _build_sampler_output(sample_results, sampling_metadata,
                                     prompt_logprobs, sample_logprobs,
                                     output_metadata, on_device_tensors)

    @property
    def _should_modify_greedy_probs_inplace(self) -> bool:
        """Whether or not the sampler should modify the probability distribution
        of greedily-sampled tokens such that multinomial sampling would sample
        the greedily-sampled token.
        In other words, if True then we set the probability of the greedily-
        sampled token to 1.
        This is used by speculative decoding, which requires that the sampling
        method be encoded into the probability distribution.
        """
        # Modify greedy probs if include_gpu_probs_tensor is set.
        return self.include_gpu_probs_tensor


def _get_bin_counts_and_mask(
    tokens: torch.Tensor,
    vocab_size: int,
    num_seqs: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Compute the bin counts for the tokens.
    # vocab_size + 1 for padding.
    bin_counts = torch.zeros((num_seqs, vocab_size + 1),
                             dtype=torch.long,
                             device=tokens.device)
    bin_counts.scatter_add_(1, tokens, torch.ones_like(tokens))
    bin_counts = bin_counts[:, :vocab_size]
    mask = bin_counts > 0

    return bin_counts, mask


def _get_custom_token_bans(
        sampling_metadata: SamplingMetadata) -> List[List[int]]:
    assert sampling_metadata.seq_groups is not None
    assert sampling_metadata.prompt_lens is not None
    banned_tokens: List[List[int]] = []
    for i, seq_group in enumerate(sampling_metadata.seq_groups):
        seq_ids, sampling_params = seq_group
        custom_token_bans = sampling_params.custom_token_bans
        if (i < sampling_metadata.num_prompts
                and sampling_params.prompt_logprobs is not None):
            prompt_len = sampling_metadata.prompt_lens[i]
            banned_tokens += [custom_token_bans] * (prompt_len - 1)
        banned_tokens += [custom_token_bans] * len(seq_ids)
    return banned_tokens


def _apply_penalties(logits: torch.Tensor, prompt_tokens_tensor: torch.Tensor,
                     output_tokens_tensor: torch.Tensor,
                     presence_penalties: torch.Tensor,
                     frequency_penalties: torch.Tensor,
                     repetition_penalties: torch.Tensor) -> torch.Tensor:
    num_seqs, vocab_size = logits.shape
    _, prompt_mask = _get_bin_counts_and_mask(prompt_tokens_tensor, vocab_size,
                                              num_seqs)
    output_bin_counts, output_mask = _get_bin_counts_and_mask(
        output_tokens_tensor, vocab_size, num_seqs)

    repetition_penalties = repetition_penalties[:, None].repeat(1, vocab_size)
    repetition_penalties[~(prompt_mask | output_mask)] = 1.0
    logits = torch.where(logits > 0, logits / repetition_penalties,
                         logits * repetition_penalties)

    # We follow the definition in OpenAI API.
    # Refer to https://platform.openai.com/docs/api-reference/parameter-details
    logits -= frequency_penalties.unsqueeze_(dim=1) * output_bin_counts
    logits -= presence_penalties.unsqueeze_(dim=1) * output_mask
    return logits


def _apply_token_bans(logits: torch.Tensor,
                      banned_tokens: List[List[int]]) -> torch.Tensor:
    for i, banned_token_ids in enumerate(banned_tokens):
        if not banned_token_ids:
            continue
        logits[i, banned_token_ids] = -float("inf")
    return logits


def _apply_min_tokens_penalty(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    assert sampling_metadata.seq_groups is not None
    assert sampling_metadata.seq_data is not None
    # list of indices in logits that will be set to -inf
    logits_to_penalize = []
    start_idx = 0
    for seq_ids, sampling_params in sampling_metadata.seq_groups:
        min_tokens = sampling_params.min_tokens
        if min_tokens > 0:
            seqs_to_penalize = []
            for i, seq_id in enumerate(seq_ids):
                seq_data = sampling_metadata.seq_data[seq_id]
                if len(seq_data.output_token_ids) < min_tokens:
                    seqs_to_penalize.append(i)

            if seqs_to_penalize:
                # convert to the index into logits
                seqs_to_penalize = [start_idx + i for i in seqs_to_penalize]
                # use set() to remove any duplicates
                token_ids_to_penalize = set(sampling_params.stop_token_ids +
                                            [sampling_params.eos_token_id])
                # itertools.product pairs each seq index with every token id
                logits_to_penalize.extend(
                    itertools.product(seqs_to_penalize, token_ids_to_penalize))

        start_idx += len(seq_ids)

    if logits_to_penalize:
        # use zip and * to group indices along each dimension
        # eg. [ (1,2), (1,3), (5,6) ] -> ( (1,1,5), (2,3,6) )
        logits[tuple(zip(*logits_to_penalize))] = -float("inf")

    return logits


def _apply_alphabet_soup(
    logits: torch.Tensor,
    p: torch.Tensor,
    k: torch.Tensor,
    a: torch.Tensor,
    m: torch.Tensor,
) -> torch.Tensor:
    logits_sort, logits_idx = logits.sort(dim=-1, descending=True)

    # Apply top-p, min-p and top-a.
    probs_sort = logits_sort.softmax(dim=-1)
    probs_sum = probs_sort.cumsum(dim=-1).sub_(probs_sort)
    min_p_thresholds = probs_sort[:, 0] * m
    top_a_thresholds = torch.pow(probs_sort[:, 0], 2) * a
    threshold = torch.maximum(min_p_thresholds, top_a_thresholds)
    mask = (probs_sort < threshold.unsqueeze(1)
            )  # Cull logits below the top-a threshold
    mask.logical_or_(
        probs_sum >
        p.unsqueeze(dim=1))  # Cull logits above the top-p summation threshold
    mask[:, 0] = False  # Guarantee at least one token is pickable
    logits_sort[mask] = -float("inf")

    # Apply top-k.
    for i, topk in enumerate(k):
        logits_sort[i, topk:] = -float("inf")

    # Re-sort the probabilities.
    src = torch.arange(logits_idx.shape[-1],
                       device=logits_idx.device).expand_as(logits_idx)
    logits_idx_inv = torch.empty_like(logits_idx).scatter_(dim=-1,
                                                           index=logits_idx,
                                                           src=src)
    logits = torch.gather(logits_sort, dim=-1, index=logits_idx_inv)
    return logits


def _apply_tfs(
    logits: torch.Tensor,
    tfs: torch.Tensor,
) -> torch.Tensor:
    logits_sort, logits_idx = logits.sort(dim=-1, descending=True)
    d2 = logits_sort.softmax(dim=-1).diff().diff().abs()
    normalized_d2 = d2 / torch.sum(d2, dim=-1, keepdim=True)
    curvature_cdf = torch.cumsum(normalized_d2, dim=-1)

    tfs_mask = curvature_cdf > tfs.unsqueeze(dim=-1)

    tfs_mask = torch.cat(
        (
            torch.zeros(
                logits.shape[0], 1, dtype=torch.bool, device=logits.device),
            tfs_mask,
            torch.ones(
                logits.shape[0], 1, dtype=torch.bool, device=logits.device),
        ),
        dim=-1,
    )

    logits_sort[tfs_mask] = -float("inf")
    logits = torch.gather(logits_sort,
                          dim=-1,
                          index=torch.argsort(logits_idx, dim=-1))

    return logits


def _apply_eta_cutoff(
    logits: torch.Tensor,
    eta_cutoff: torch.Tensor,
) -> torch.Tensor:
    shifted_logits = torch.log_softmax(logits, dim=-1)
    probs = shifted_logits.exp()

    neg_entropy = (probs * shifted_logits).nansum(dim=-1)
    eps = torch.min(eta_cutoff,
                    torch.sqrt(eta_cutoff) *
                    torch.exp(neg_entropy)).unsqueeze(dim=1)

    eta_mask = probs < eps

    # guard against nulling out all the logits
    top_idx = torch.argmax(probs, dim=1, keepdim=True)
    eta_mask.scatter_(dim=1, index=top_idx, value=False)

    logits[eta_mask] = -float("inf")
    return logits


def _apply_epsilon_cutoff(
    logits: torch.Tensor,
    epsilon_cutoff: torch.Tensor,
) -> torch.Tensor:
    probs = logits.softmax(dim=-1)

    eps_mask = probs < epsilon_cutoff.unsqueeze(dim=1)

    # guard against nulling out all the logits
    top_idx = torch.argmax(probs, dim=1, keepdim=True)
    eps_mask.scatter_(dim=1, index=top_idx, value=False)

    logits[eps_mask] = -float("inf")
    return logits


def _apply_typical_sampling(
    logits: torch.Tensor,
    typical_p: torch.Tensor,
) -> torch.Tensor:
    shifted_logits = torch.log_softmax(logits, dim=-1)
    probs = shifted_logits.exp()

    neg_entropy = (probs * shifted_logits).nansum(dim=-1, keepdim=True)

    surprisal_deviations = (neg_entropy - shifted_logits).abs()
    _, indices = torch.sort(surprisal_deviations)
    reordered_probs = probs.gather(-1, indices)
    typ_mask_sorted = reordered_probs.cumsum(dim=-1) >= typical_p.unsqueeze(
        dim=1)

    min_tokens_to_keep = 1
    # Keep at least min_tokens_to_keep
    typ_mask_sorted[..., :min_tokens_to_keep] = 0

    typ_mask = typ_mask_sorted.scatter(1, indices, typ_mask_sorted)
    logits[typ_mask] = -float("inf")
    return logits


# pulls double duty for temperature and dynatemp
def _apply_temperature(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    dynatemp_mins: torch.Tensor,
    dynatemp_maxs: torch.Tensor,
    dynatemp_exps: torch.Tensor,
) -> torch.Tensor:
    dynatemp_mask = torch.logical_or(dynatemp_mins > 0, dynatemp_maxs > 0)
    dynatemp_mins = dynatemp_mins[dynatemp_mask]
    dynatemp_maxs = dynatemp_maxs[dynatemp_mask]
    dynatemp_exps = dynatemp_exps[dynatemp_mask]
    dynatemp_mins = dynatemp_mins.clamp_(min=0)

    dynatemp_logits = logits[dynatemp_mask]
    dynatemp_shifted_logits = torch.log_softmax(dynatemp_logits, dim=-1)
    dynatemp_probs = dynatemp_shifted_logits.exp()
    dynatemp_entropies = -(dynatemp_probs *
                           dynatemp_shifted_logits).nansum(dim=-1)
    dynatemp_max_entropies = torch.log_(
        (dynatemp_logits > float("-inf")).sum(dim=-1).float())
    normalized_entropies = dynatemp_entropies.div_(dynatemp_max_entropies)
    dyn_temp = (dynatemp_mins + (dynatemp_maxs - dynatemp_mins) *
                normalized_entropies.pow_(dynatemp_exps))

    temperatures[dynatemp_mask] = dyn_temp
    temperatures[temperatures == 0.0] = 1.0
    logits.div_(temperatures.unsqueeze_(dim=1))
    return logits


def _apply_quadratic_sampling(
    logits: torch.Tensor,
    indices: torch.Tensor,
    factors: torch.Tensor,
    curves: torch.Tensor,
) -> torch.Tensor:
    """
    Applies a quadratic transformation to the logits based on the
    provided smoothing factors and curves. The transformation is
    centered around the maximum logit value in the batch.

    The transformation involves a quadratic and cubic term, with the
    cubic term controlled by the smoothing curve. The quadratic term is
    scaled by the smoothing factor, and the cubic term is scaled by the
    product of the smoothing factor and the smoothing curve.

    params:
        logits (torch.Tensor): The logits to be transformed.
        indices (torch.Tensor): Indices to project `logits` down to 
            the other tensor's lengths.
        factors (torch.Tensor): The factors to scale the quadratic
            term in the transformation.
        curves (torch.Tensor): The factors to scale the cubic term
            in the transformation.

    returns:
        torch.Tensor: The transformed logits.

    Credits: @kalomaze
    """
    factors.unsqueeze_(dim=1)
    curves.unsqueeze_(dim=1)
    k = factors * (3 - curves) / 2
    s = factors * (curves - 1) / 2

    quadlogits = logits[indices]  # project to only relevant logits
    max_logits = quadlogits.max(dim=-1, keepdim=True).values

    # Construct the delta from each logit to its new value
    diff = quadlogits - max_logits
    diff -= diff**2 * (s * diff - k)
    diff[diff != diff] = 0  # Eliminate NaNs from infs

    logits[indices] -= diff
    return logits


def _greedy_sample(
    selected_seq_groups: List[Tuple[List[int], SamplingParams]],
    samples: torch.Tensor,
) -> List[Tuple[List[int], List[int]]]:
    sample_idx = 0
    results = []
    for seq_group in selected_seq_groups:
        seq_ids, _ = seq_group
        num_parent_seqs = len(seq_ids)
        assert num_parent_seqs == 1, (
            "Greedy sampling should have only one seq.")
        parent_ids = list(range(num_parent_seqs))
        next_token_ids = [samples[sample_idx].item()]
        results.append((next_token_ids, parent_ids))
        sample_idx += num_parent_seqs
    return results


def _random_sample(
    selected_seq_groups: List[Tuple[List[int], SamplingParams]],
    is_prompts: List[bool],
    random_samples: torch.Tensor,
) -> List[Tuple[List[int], List[int]]]:
    # Find the maximum best_of value of the prompt phase requests.
    random_samples = random_samples.cpu()
    sample_idx = 0
    results = []
    for seq_group, is_prompt in zip(selected_seq_groups, is_prompts):
        seq_ids, sampling_params = seq_group
        num_parent_seqs = len(seq_ids)
        if is_prompt:
            # Prompt phase.
            parent_ids = [0] * sampling_params.best_of
            next_token_ids = random_samples[
                sample_idx, :sampling_params.best_of].tolist()
        else:
            # Generation phase.
            parent_ids = list(range(num_parent_seqs))
            next_token_ids = random_samples[sample_idx:sample_idx +
                                            num_parent_seqs, 0].tolist()
        results.append((next_token_ids, parent_ids))
        sample_idx += num_parent_seqs
    return results


def _beam_search_sample(
    selected_seq_groups: List[Tuple[List[int], SamplingParams]],
    is_prompts: List[bool],
    seq_data: Dict[int, SequenceData],
    logprobs: torch.Tensor,
) -> List[Tuple[List[int], List[int]]]:
    # We sample 2 * beam_width candidates to make sure that with high
    # probability we can get `beam_width` candidates in addition to
    # the finished sequences for the next iteration. See
    # https://github.com/tensorflow/tensor2tensor/blob/bafdc1b67730430d38d6ab802cbd51f9d053ba2e/tensor2tensor/utils/beam_search.py#L557-L563
    # for details. See also HF reference:
    # https://github.com/huggingface/transformers/blob/a4dd53d88e4852f023332d284ff07a01afcd5681/src/transformers/generation/utils.py#L3063-L3065
    #
    # Note: Beam search is not vectorized, so its speed can be slower than
    # other sampling methods.
    sample_idx = 0
    results = []
    for seq_group, is_prompt in zip(selected_seq_groups, is_prompts):
        seq_ids, sampling_params = seq_group
        num_parent_seqs = len(seq_ids)
        beam_width = sampling_params.best_of
        seq_group_logprobs = logprobs[sample_idx:sample_idx + num_parent_seqs]
        if is_prompt:
            # Prompt phase.
            assert num_parent_seqs == 1, (
                "Prompt input should have only one seq.")
            parent_ids = [0] * (2 * beam_width)
            _, next_token_ids = torch.topk(seq_group_logprobs[0],
                                           2 * beam_width)
            next_token_ids = next_token_ids.tolist()
        else:
            # Generation phase.
            cumulative_logprobs = [
                seq_data[seq_id].cumulative_logprob for seq_id in seq_ids
            ]
            cumulative_logprobs = torch.tensor(
                cumulative_logprobs,
                dtype=torch.float,
                device=seq_group_logprobs.device)
            seq_group_logprobs = (seq_group_logprobs +
                                  cumulative_logprobs.unsqueeze(dim=1))
            _, topk_ids = torch.topk(seq_group_logprobs.flatten(),
                                     2 * beam_width)
            topk_ids = topk_ids.tolist()
            vocab_size = seq_group_logprobs.size(-1)
            parent_ids = [i // vocab_size for i in topk_ids]
            next_token_ids = [i % vocab_size for i in topk_ids]
        results.append((next_token_ids, parent_ids))
        sample_idx += num_parent_seqs
    assert sample_idx == logprobs.size(0)
    return results


# torch.multinomial forces a GPU<->CPU sync.
# Therefore, we use an optimized implementation instead.
# Note that we always sample with replacement.
# probs will be modified in place, but this is fine, as we pass
# in a copy already.
def _multinomial(
    probs: torch.Tensor,
    num_samples: int,
    seq_groups: Optional[List[Tuple[List[int], SamplingParams]]] = None,
    generators: Optional[List[torch.Generator]] = None,
) -> torch.Tensor:
    if num_samples > 1:
        # This is equivalent to torch.repeat_interleaved (which also
        # forces a GPU<->CPU sync).
        # This allows us to do sampling with replacement by creating
        # num_samples copies of each row in the tensor, and then
        # batch sampling the resulting tensor.
        probs = probs[:, None, :].expand(probs.shape[0], num_samples,
                                         probs.shape[1]).contiguous().view(
                                             -1, probs.shape[1])
    q = torch.empty_like(probs)
    if seq_groups is None:
        q.exponential_()
    else:
        assert generators is not None
        sample_idx = 0
        for (seq_ids, _), generator in zip(seq_groups, generators):
            next_sample_idx = sample_idx + len(seq_ids) * num_samples
            q[sample_idx:next_sample_idx].exponential_(generator=generator)
            sample_idx = next_sample_idx
    return probs.div_(q).argmax(dim=1).view(-1, num_samples)


def _sample_with_torch(
    probs: torch.Tensor,
    logprobs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    include_gpu_probs_tensor: bool,
    modify_greedy_probs: bool,
) -> Tuple[List[Tuple[List[int], List[int]]], Optional[torch.Tensor]]:
    """Returns list of (selected_tokens, parent_seq_ids) tuples
    corresponding to sampling_metadata.seq_groups."""
    assert sampling_metadata.seq_groups is not None
    assert sampling_metadata.categorized_sample_indices is not None
    assert sampling_metadata.seq_data is not None
    categorized_seq_group_ids = {t: [] for t in SamplingType}
    categorized_sample_indices = sampling_metadata.categorized_sample_indices
    for i, seq_group in enumerate(sampling_metadata.seq_groups):
        _, sampling_params = seq_group
        sampling_type = sampling_params.sampling_type
        categorized_seq_group_ids[sampling_type].append(i)

    sample_results_dict: Dict[int, Tuple[List[int], List[int]]] = {}
    sample_metadata = {}
    multinomial_samples = {}

    # Create output tensor for sampled token ids.
    if include_gpu_probs_tensor:
        sampled_token_ids_tensor = torch.empty(logprobs.shape[0],
                                               1,
                                               dtype=torch.long,
                                               device=logprobs.device)
    else:
        sampled_token_ids_tensor = None

    # Counterintuitively, having two loops here is actually faster.
    # The first loop can run without waiting on GPU<->CPU sync.
    for sampling_type, sample_indices in categorized_sample_indices.items():
        sample_indices = sample_indices[:, 0]
        if len(sample_indices) == 0:
            continue
        seq_group_ids = categorized_seq_group_ids[sampling_type]
        seq_groups = [sampling_metadata.seq_groups[i] for i in seq_group_ids]
        is_prompts = [i < sampling_metadata.num_prompts for i in seq_group_ids]
        sample_metadata[sampling_type] = (seq_group_ids, seq_groups,
                                          is_prompts, sample_indices)
        long_sample_indices = sample_indices.long()
        if sampling_type == SamplingType.GREEDY:
            greedy_samples = torch.argmax(logprobs[long_sample_indices],
                                          dim=-1)

            if include_gpu_probs_tensor:
                # Store sampled tokens in output tensor.
                sampled_token_ids_tensor[
                    long_sample_indices] = greedy_samples.unsqueeze(-1)

            if modify_greedy_probs:
                # If required, modify the probabilities such that sampling from
                # the modified distribution would always sample the argmax
                # token id.
                _modify_greedy_probs_inplace(logprobs, probs,
                                             long_sample_indices,
                                             greedy_samples)
        elif sampling_type in (SamplingType.RANDOM, SamplingType.RANDOM_SEED):
            max_best_of_in_batch = 1
            for seq_group, is_prompt in zip(seq_groups, is_prompts):
                if is_prompt:
                    _, sampling_params = seq_group
                    max_best_of_in_batch = max(max_best_of_in_batch,
                                               sampling_params.best_of)
            seeded_args = {} if sampling_type == SamplingType.RANDOM else {
                "seq_groups": seq_groups,
                "generators": sampling_metadata.generators,
            }
            multinomial_samples[sampling_type] = _multinomial(
                probs[long_sample_indices], max_best_of_in_batch,
                **seeded_args)

            if include_gpu_probs_tensor:
                # Store sampled tokens in output tensor.
                sampled_token_ids_tensor[
                    long_sample_indices] = multinomial_samples[sampling_type]
        elif sampling_type == SamplingType.BEAM:
            beam_search_logprobs = logprobs[sample_indices]
        else:
            raise ValueError(f"Unsupported sampling type: {sampling_type}")

    # GPU<->CPU sync happens in the loop below.
    # This also converts the sample output to Python objects.

    for sampling_type, metadata in sample_metadata.items():
        seq_group_ids, seq_groups, is_prompts, sample_indices = metadata
        if sampling_type == SamplingType.GREEDY:
            sample_results = _greedy_sample(seq_groups, greedy_samples)
        elif sampling_type in (SamplingType.RANDOM, SamplingType.RANDOM_SEED):
            sample_results = _random_sample(seq_groups, is_prompts,
                                            multinomial_samples[sampling_type])
        elif sampling_type == SamplingType.BEAM:
            sample_results = _beam_search_sample(seq_groups, is_prompts,
                                                 sampling_metadata.seq_data,
                                                 beam_search_logprobs)
        sample_results_dict.update(zip(seq_group_ids, sample_results))

    sample_results = [
        sample_results_dict[i]
        for i in range(len(sampling_metadata.seq_groups))
    ]
    return sample_results, sampled_token_ids_tensor


def _sample_with_triton_kernel(
    probs: torch.Tensor,
    logprobs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    sampling_tensors: SamplingTensors,
) -> List[Tuple[List[int], List[int]]]:
    assert sampling_metadata.seq_groups is not None
    assert sampling_metadata.categorized_sample_indices is not None
    assert sampling_metadata.seq_data is not None
    categorized_seq_group_ids = {t: [] for t in SamplingType}
    categorized_sample_indices = sampling_metadata.categorized_sample_indices
    for i, seq_group in enumerate(sampling_metadata.seq_groups):
        _, sampling_params = seq_group
        sampling_type = sampling_params.sampling_type
        categorized_seq_group_ids[sampling_type].append(i)

    sample_results_dict: Dict[int, Tuple[List[int], List[int]]] = {}
    sample_metadata = {}
    max_best_of_in_batch = 1

    # Counterintuitively, having two loops here is actually faster.
    # The first loop can run without waiting on GPU<->CPU sync.
    for sampling_type, sample_indices in categorized_sample_indices.items():
        sampled_token_indices = sample_indices[:, 1]
        sample_indices = sample_indices[:, 0]
        if len(sample_indices) == 0:
            continue
        seq_group_ids = categorized_seq_group_ids[sampling_type]
        seq_groups = [sampling_metadata.seq_groups[i] for i in seq_group_ids]
        is_prompts = [i < sampling_metadata.num_prompts for i in seq_group_ids]
        sample_metadata[sampling_type] = (seq_group_ids, seq_groups,
                                          is_prompts, sample_indices,
                                          sampled_token_indices)
        if sampling_type in (SamplingType.GREEDY, SamplingType.RANDOM,
                             SamplingType.RANDOM_SEED):
            for seq_group, is_prompt in zip(seq_groups, is_prompts):
                if is_prompt:
                    _, sampling_params = seq_group
                    max_best_of_in_batch = max(max_best_of_in_batch,
                                               sampling_params.best_of)
        elif sampling_type == SamplingType.BEAM:
            beam_search_logprobs = logprobs[sample_indices]
        else:
            raise ValueError(f"Unsupported sampling type: {sampling_type}")

    sampled_tokens, _, _ = sample_triton(
        probs=probs,
        seeds=sampling_tensors.seed_transpose,
        max_best_of=max_best_of_in_batch,
        sample_indices=sampling_tensors.seed_indices,
        logprobs=logprobs,
        # don't save logprobs because we have logic for that below
        # TODO: use this instead of the CPU-based logic below
        save_logprobs=False,
    )

    # GPU<->CPU sync happens in the loop below.

    for sampling_type in SamplingType:
        if sampling_type not in sample_metadata:
            continue
        (seq_group_ids, seq_groups, is_prompts, sample_indices,
         sampled_token_indices) = sample_metadata[sampling_type]
        if sampling_type == SamplingType.GREEDY:
            sample_results = _greedy_sample(
                seq_groups, sampled_tokens[sampled_token_indices][:, 0])
        elif sampling_type in (SamplingType.RANDOM, SamplingType.RANDOM_SEED):
            sample_results = _random_sample(
                seq_groups, is_prompts, sampled_tokens[sampled_token_indices])
        elif sampling_type == SamplingType.BEAM:
            sample_results = _beam_search_sample(seq_groups, is_prompts,
                                                 sampling_metadata.seq_data,
                                                 beam_search_logprobs)
        sample_results_dict.update(zip(seq_group_ids, sample_results))

    sample_results = [
        sample_results_dict[i]
        for i in range(len(sampling_metadata.seq_groups))
    ]
    return sample_results


def _sample(
    probs: torch.Tensor, logprobs: torch.Tensor,
    sampling_metadata: SamplingMetadata, sampling_tensors: SamplingTensors,
    include_gpu_probs_tensor: bool, modify_greedy_probs: bool
) -> Tuple[List[Tuple[List[int], List[int]]], Optional[torch.Tensor]]:
    return _sample_with_torch(
        probs,
        logprobs,
        sampling_metadata,
        include_gpu_probs_tensor=include_gpu_probs_tensor,
        modify_greedy_probs=modify_greedy_probs,
    )

    # TODO: Enable once Triton kernel & associated code is faster.
    # return _sample_with_triton_kernel(probs, logprobs, sampling_metadata,
    #                                   sampling_tensors)


def _get_ranks(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    This function calculates the ranks of the chosen tokens in a logprob tensor.
    Args:
        x (torch.Tensor): 2D logprob tensor of shape (N, M)
                        where N is the no. of tokens and M is the vocab dim.
        indices (torch.Tensor): List of chosen token indices.
    Returns:
        torch.Tensor: 1D tensor of shape (N,) where N is the no. of tokens.
                    Each element in the returned tensor represents the rank 
                    of the chosen token in the input logprob tensor.
    """
    vals = x[torch.arange(0, len(x), device=x.device, dtype=indices.dtype),
             indices]
    return (x > vals[:, None]).long().sum(1).add_(1)


def _get_logprobs(
    logprobs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    sample_results: List[Tuple[List[int], List[int]]],
) -> Tuple[List[Optional[PromptLogprobs]], List[SampleLogprobs]]:
    assert sampling_metadata.seq_groups is not None
    assert sampling_metadata.prompt_lens is not None
    assert sampling_metadata.seq_data is not None

    # Prepare query indices
    batched_logprobs_query_seq_indices: List[int] = []
    batched_logprobs_query_token_indices: List[int] = []
    # at least get one logprob for each token
    largest_num_logprobs = 1
    sample_idx = 0
    for i, (seq_group, sample_result) in enumerate(
            zip(sampling_metadata.seq_groups, sample_results)):
        seq_ids, sampling_params = seq_group
        next_token_ids, parent_ids = sample_result
        num_parent_seqs = len(seq_ids)
        if (i < sampling_metadata.num_prompts
                and sampling_params.prompt_logprobs is not None):
            largest_num_logprobs = max(largest_num_logprobs,
                                       sampling_params.prompt_logprobs)
            prompt_len = sampling_metadata.prompt_lens[i]
            prompt_tokens = sampling_metadata.seq_data[
                seq_ids[0]].prompt_token_ids
            batched_logprobs_query_seq_indices.extend(
                sample_idx + j for j in range(prompt_len - 1))
            batched_logprobs_query_token_indices.extend(
                token_id for token_id in prompt_tokens[1:])
            sample_idx += prompt_len - 1
        batched_logprobs_query_seq_indices.extend(
            [sample_idx + parent_id for parent_id in parent_ids])
        batched_logprobs_query_token_indices.extend(next_token_ids)
        if sampling_params.logprobs is not None:
            largest_num_logprobs = max(largest_num_logprobs,
                                       sampling_params.logprobs)
        sample_idx += num_parent_seqs
    assert sample_idx == logprobs.size(0)

    batched_logprobs_query_seq_indices_gpu = torch.tensor(
        batched_logprobs_query_seq_indices, device=logprobs.device)
    batched_logprobs_query_token_indices_gpu = torch.tensor(
        batched_logprobs_query_token_indices, device=logprobs.device)
    # Batched query for logprobs of selected token
    batched_logprobs_query_result = logprobs[[
        batched_logprobs_query_seq_indices_gpu,
        batched_logprobs_query_token_indices_gpu
    ]]
    batched_ranks_query_result = _get_ranks(
        logprobs[batched_logprobs_query_seq_indices_gpu],
        batched_logprobs_query_token_indices_gpu)
    # Batched query for logprobs of topk tokens
    if largest_num_logprobs > 0:
        top_logprobs, top_token_ids = torch.topk(logprobs,
                                                 largest_num_logprobs,
                                                 dim=-1)
        top_logprobs = top_logprobs.cpu()
        top_token_ids = top_token_ids.cpu()
    else:
        top_logprobs, top_token_ids = None, None

    batched_logprobs_query_result = batched_logprobs_query_result.cpu()

    batched_ranks_query_result = batched_ranks_query_result.cpu()

    # Gather results
    result_prompt_logprobs: List[Optional[PromptLogprobs]] = []
    result_sample_logprobs: List[SampleLogprobs] = []
    sample_idx = 0
    query_result_idx = 0
    for i, (seq_group, sample_result) in enumerate(
            zip(sampling_metadata.seq_groups, sample_results)):
        seq_ids, sampling_params = seq_group
        next_token_ids, parent_ids = sample_result
        # Prompt logprobs
        if (i < sampling_metadata.num_prompts
                and sampling_params.prompt_logprobs is not None):
            num_logprobs = sampling_params.prompt_logprobs
            prompt_tokens = sampling_metadata.seq_data[
                seq_ids[0]].prompt_token_ids
            group_prompt_logprobs: PromptLogprobs = [None]
            for token_id in prompt_tokens[1:]:
                prompt_logprobs_dict = {
                    token_id:
                    (batched_logprobs_query_result[query_result_idx].item(),
                     batched_ranks_query_result[query_result_idx].item())
                }
                if num_logprobs > 0:
                    prompt_logprobs_dict.update(
                        zip(
                            top_token_ids[sample_idx, :num_logprobs].tolist(),
                            zip(
                                top_logprobs[
                                    sample_idx, :num_logprobs].tolist(),
                                range(1, num_logprobs + 1))))
                group_prompt_logprobs.append({
                    token_id: Logprob(*logprob_rank)
                    for token_id, logprob_rank in prompt_logprobs_dict.items()
                })
                sample_idx += 1
                query_result_idx += 1
            result_prompt_logprobs.append(group_prompt_logprobs)
        else:
            result_prompt_logprobs.append(None)
        # Sample logprobs
        num_logprobs = sampling_params.logprobs
        if num_logprobs is None:
            num_logprobs = 0
        group_sample_logprobs: SampleLogprobs = []
        for next_token_id, parent_id in zip(next_token_ids, parent_ids):
            sample_logprobs_dict = {
                next_token_id:
                (batched_logprobs_query_result[query_result_idx].item(),
                 batched_ranks_query_result[query_result_idx].item())
            }
            query_result_idx += 1
            if num_logprobs >= 0:
                sample_logprobs_dict.update(
                    zip(
                        top_token_ids[sample_idx +
                                      parent_id, :num_logprobs].tolist(),
                        zip(
                            top_logprobs[sample_idx +
                                         parent_id, :num_logprobs].tolist(),
                            range(1, num_logprobs + 1))))
            group_sample_logprobs.append({
                token_id: Logprob(*logprob_rank)
                for token_id, logprob_rank in sample_logprobs_dict.items()
            })
        result_sample_logprobs.append(group_sample_logprobs)
        sample_idx += len(seq_ids)
    return result_prompt_logprobs, result_sample_logprobs


def _modify_greedy_probs_inplace(logprobs: torch.Tensor, probs: torch.Tensor,
                                 sample_indices: torch.Tensor,
                                 greedy_samples: torch.Tensor) -> None:
    """Modify the probability distributions of the greedily-sampled tokens such
    that each sampled token has a "probability" of 1.0. This is required by
    speculative decoding, which depends on the sampling method being encoded
    within the probability distribution for correctness.
    # Why do we only need to do this for greedy sampling?
    Aphrodite's sampler performs the following steps for greedy or multinomial
    (random) sampling:
        1. Get logits from model.
        2. Modify logits according to per-sequence sampling parameters.
            - Multiply by temperature, top-k and top-p masking, penalize tokens
                according to their frequency, etc.
        3. Sample a token.
            - Random sampling simply samples from the modified probability
                distribution.
            - Greedy sampling performs `argmax` to obtain the token with the
                highest likelihood.
    
    Ignoring greedy sampling for a moment, we find that the computed probability
    distribution has the following property: we can sample from it independently
    and find that the token sampled by the Sampler has a frequency corresponding
    to how often we see it in our sampling. In other words, for tokens sampled
    with Aphrodite's random SamplingType, the computed probability distribution
    encodes the sampling methodology completely.
    Greedy sampling does not normally have this property. Aphrodite modifies
    logits according to sampling params, then performs `argmax`, then returns
    the sampled token and the computed probability distribution. If we sample
    from the distribution, we'll find the likelihood of the greedily-sampled
    token is not always 1.0.
    Since lossless speculative decoding requires that the sampling methodology
    be encoded within the probability distribution, we are motivated to modify
    the probability distribution such that the sampled token has probability 1
    when speculative decoding is used.
    NOTE: Alternatively, we could use an extremely low temperature to achieve
    greedy sampling using multinomial computation and unite the codepaths. This
    has implications on the overall design of the sampler, e.g. how to record
    accurate logprobs for the user, so this improvement is deferred to later.
    """
    logprobs[sample_indices, :] = -float('inf')
    logprobs[sample_indices, greedy_samples] = 0.0
    probs[sample_indices, :] = 0
    probs[sample_indices, greedy_samples] = 1.0


def _build_sampler_output(
    sample_results: List[Tuple[List[int], List[int]]],
    sampling_metadata: SamplingMetadata,
    prompt_logprobs: List[Optional[PromptLogprobs]],
    sample_logprobs: List[SampleLogprobs],
    output_metadata: OutputMetadata,
    on_device_tensors: Optional[Tuple[torch.Tensor, torch.Tensor]],
) -> SamplerOutput:
    """Construct Python objects with the output of sampling.
    Args:
        on_device_tensors: Tuple containing on-device tensors with the
            probabilities used in sampling and the sampled token ids. This
            allows post-processing without copies to CPU/serialization, e.g. in
            speculative decoding rejection sampling.
    """
    assert sampling_metadata.seq_groups is not None
    sampler_output = []
    for (seq_group, sample_result, group_prompt_logprobs,
         group_sample_logprobs) in zip(sampling_metadata.seq_groups,
                                       sample_results, prompt_logprobs,
                                       sample_logprobs):
        seq_ids, _ = seq_group
        seq_outputs = [
            SequenceOutput(seq_ids[parent_id], token_id, logprobs,
                           output_metadata.get(seq_ids[parent_id], idx))
            for idx, (token_id, parent_id, logprobs) in enumerate(
                zip(*sample_result, group_sample_logprobs))
        ]

        sampler_output.append(
            SequenceGroupOutput(seq_outputs, group_prompt_logprobs))
    # If not specified, store None values in SamplerOutput.
    if on_device_tensors is not None:
        sampled_token_probs, sampled_token_ids = on_device_tensors
    else:
        sampled_token_probs, sampled_token_ids = (None, None)

    return SamplerOutput(
        outputs=sampler_output,
        sampled_token_probs=sampled_token_probs,
        sampled_token_ids=sampled_token_ids,
    )


def _apply_mirostat_v2(logits: torch.Tensor,
                       sampling_tensors: SamplingTensors) -> torch.Tensor:
    # Reduce our view to just the affected logits
    logit_view = logits[sampling_tensors.miro_indices]

    # Calculate surprise value per token
    #  Convert nats to bits for compatibility with ooba/kobold parameters.
    logit_surprise = torch.log_softmax(logit_view, dim=-1) / -math.log(2)

    # Mask out "too-surprising" tokens (surprisal > mu)
    mus = sampling_tensors.miro_mus
    miro_mask = logit_surprise > mus.unsqueeze(dim=-1)

    # Unmask most-likely logit to guarantee a selection.
    maxinds = torch.argmax(logit_view, dim=-1, keepdim=True)
    miro_mask.scatter_(dim=1, index=maxinds, value=False)

    # Apply logit mask (effectively a top-k filter).
    logit_view[miro_mask] = -float("inf")

    # Project logit changes made to the view onto the original.
    # I think this step might be redundant.
    logits[sampling_tensors.miro_indices] = logit_view
    return logits


def _mirostat_store_args(logits: torch.Tensor, args: SamplingTensors,
                         sample_results: List[Tuple[List[int], List[int]]],
                         sampling_metadata: SamplingMetadata,
                         output_metadata: OutputMetadata) -> None:
    """Based on whichever token was finally sampled, we calculate the
    final surprisal values to update the mus.
    
    Because a single sequence can have multiple samples, we must fork
    the mu accordingly."""
    assert sampling_metadata.seq_groups is not None
    seqid_to_tokens = {}
    seqid_to_indices = {}
    for (sids, _), (toks, parents) in zip(sampling_metadata.seq_groups,
                                          sample_results):
        for idx, (token, parent) in enumerate(zip(toks, parents)):
            seqid_to_tokens.setdefault(sids[parent], []).append(token)
            seqid_to_indices.setdefault(sids[parent], []).append(idx)

    seqids = args.miro_seqids

    picked_tokens = torch.tensor([seqid_to_tokens[x] for x in seqids],
                                 device=logits.device,
                                 dtype=torch.long)

    # Clumsily, we recalculate token surprisals.
    logits_view = logits[args.miro_indices]
    picked_surprise = torch.gather(torch.log_softmax(logits_view, dim=-1),
                                   dim=-1,
                                   index=picked_tokens) / -math.log(2)

    taus = args.miro_taus.unsqueeze(dim=-1)  # AKA target surprisals
    etas = args.miro_etas.unsqueeze(dim=-1)  # AKA accumulation rates
    mus = args.miro_mus.unsqueeze(dim=-1)  # AKA surprisal accumulators
    nu_mus = mus - (picked_surprise - taus) * etas

    # Record updated mu values for use in the next iteration
    # Note how each mu is split into multiple based on the number of samples.
    for seqid, seq_mus in zip(seqids, nu_mus):
        for sample_idx, mu in zip(seqid_to_indices[seqid], seq_mus):
            output_metadata.add(seqid, sample_idx, "miro_mu", mu)
