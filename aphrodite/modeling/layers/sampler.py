"""A layer that samples the next tokens from the model's outputs."""
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn

from aphrodite.modeling.sampling_metadata import (SamplingMetadata,
                                                  OutputMetadata,
                                                  SamplingTensors)
from aphrodite.modeling.megatron.communication_op import (
    tensor_model_parallel_gather)
from aphrodite.common.sampling_params import SamplingParams, SamplingType
from aphrodite.common.sequence import (Logprob, PromptLogprobs, SampleLogprobs,
                                       SamplerOutput, SequenceData,
                                       SequenceGroupOutput, SequenceOutput)


class Sampler(nn.Module):
    """Samples the next tokens from the model's outputs.

    This layer does the following:
    1. Discard the hidden states that are not used for sampling (i.e., all
        tokens except the final one in each prompt).
    2. Compute the logits for the next tokens.
    3. Apply presence and frequency penalties.
    4. Apply temperature scaling.
    5. Apply top-p and top-k truncation.
    6. Sample the next tokens.
    Here, each sequence group within the batch can have different sampling
    parameters (e.g., sampling method, temperature, top-p, top-k, etc.).
    """

    def __init__(self,
                 vocab_size: int,
                 org_vocab_size: Optional[int] = None) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        # original vocabulary size (without LoRA).
        self.org_vocab_size = org_vocab_size or vocab_size

    def _get_logits(self, hidden_states: torch.Tensor, embedding: torch.Tensor,
                    embedding_bias: Optional[torch.Tensor]) -> torch.Tensor:
        # Get the logits for the next tokens.
        logits = torch.matmul(hidden_states, embedding.t())
        if embedding_bias is not None:
            logits += embedding_bias
        logits = tensor_model_parallel_gather(logits)
        # Remove paddings in vocab (if any).
        if logits is not None:
            logits = logits[:, :self.org_vocab_size]
        return logits

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        # Get the hidden states that we use for sampling.
        logits = _prune_hidden_states(logits, sampling_metadata)
        logits = tensor_model_parallel_gather(logits)
        # Remove paddings in vocab (if any).
        if logits is not None:
            logits = logits[:, :self.vocab_size]

        # Only perform sampling in the driver worker.
        # Note: `_get_logits` is still distributed across TP workers because
        # the `embedding` weight is distributed across TP workers.
        # TODO: Change the get_logits part to a separate stage.
        if not sampling_metadata.perform_sampling:
            return None

        assert logits is not None
        _, vocab_size = logits.shape

        output_metadata = OutputMetadata()

        # Apply logits processors (if any)
        logits = _apply_logits_processors(logits, sampling_metadata)

        # Prepare sampling tensors with pinned memory to avoid blocking.
        (sampling_tensors, do_temperatures, do_penalties, do_topks, do_topps,
         do_topas, do_minps, do_tfss, do_eta_cutoffs, do_epsilon_cutoffs,
         do_typical_ps, do_quadratic,
         do_mirostat) = (SamplingTensors.from_sampling_metadata(
             sampling_metadata, vocab_size, logits.device, logits.dtype))

        if do_penalties:
            logits = _apply_penalties(logits, sampling_tensors.prompt_tokens,
                                      sampling_tensors.output_tokens,
                                      sampling_tensors.presence_penalties,
                                      sampling_tensors.frequency_penalties,
                                      sampling_tensors.repetition_penalties)

        if do_temperatures:
            logits = _apply_temperature(logits, sampling_tensors.temperatures,
                                        sampling_tensors.dynatemp_mins,
                                        sampling_tensors.dynatemp_maxs,
                                        sampling_tensors.dynatemp_exps)

        if do_topks or do_topps or do_topas or do_minps:
            logits = _apply_alphabet_soup(logits, sampling_tensors.top_ps,
                                          sampling_tensors.top_ks,
                                          sampling_tensors.top_as,
                                          sampling_tensors.min_ps)
        if do_tfss:
            logits = _apply_tfs(logits, sampling_tensors.tfss)
        if do_eta_cutoffs:
            logits = _apply_eta_cutoff(logits, sampling_tensors.eta_cutoffs)
        if do_epsilon_cutoffs:
            logits = _apply_epsilon_cutoff(logits,
                                           sampling_tensors.epsilon_cutoffs)
        if do_typical_ps:
            logits = _apply_typical_sampling(logits,
                                             sampling_tensors.typical_ps)
        if do_quadratic:
            logits = _apply_quadratic_sampling(
                logits, sampling_tensors.smoothing_factors,
                sampling_tensors.smoothing_curves)

        banned_tokens = _get_custom_token_bans(sampling_metadata)
        assert len(banned_tokens) == logits.shape[0]
        logits = _apply_token_bans(logits, banned_tokens)
        if do_mirostat:
            logits = _mirostat(logits, sampling_tensors, output_metadata)

        # We use float32 for probabilities and log probabilities.
        # Compute the probabilities.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # Compute the log probabilities.
        # Use log_softmax to ensure numerical stability.
        logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)

        # Sample the next tokens.
        sample_results = _sample(probs, logprobs, sampling_metadata)
        # Get the logprobs query results.
        prompt_logprobs, sample_logprobs = _get_logprobs(
            logprobs, sampling_metadata, sample_results)
        return _build_sampler_output(sample_results, sampling_metadata,
                                     prompt_logprobs, sample_logprobs,
                                     output_metadata)


def _prune_hidden_states(
    hidden_states: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    return hidden_states.index_select(0,
                                      sampling_metadata.selected_token_indices)


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


# def _apply_logits_processors(
#     logits: torch.Tensor,
#     metadata: SamplingMetadata,
# ) -> torch.Tensor:
#     seq_offset = 0
#     for i, (seq_ids, sampling_params) in enumerate(metadata.seq_groups):
#         seq_size = len(seq_ids)
#         output_tokens = []
#         if (i < metadata.num_prompts
#                 and sampling_params.prompt_logprobs is not None):
#             prompt_seqs = metadata.prompt_lens[i] - 1
#             seq_size += prompt_seqs
#             output_tokens.extend([[]] * prompt_seqs)
#         seq_end = seq_offset + seq_size

#         if sampling_params.logits_processors:
#             output_tokens.extend(metadata.seq_data[sid].output_token_ids
#                                  for sid in seq_ids)
#             for proc in sampling_params.logits_processors:
#                 proc(logits[seq_offset:seq_end], output_tokens)

#         seq_offset = seq_end

#     return logits


def _apply_logits_processors(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    logits_row_idx = 0
    found_logits_processors = False
    for seq_ids, sampling_params in sampling_metadata.seq_groups:
        logits_processors = sampling_params.logits_processors
        if logits_processors:
            found_logits_processors = True
            for seq_id in seq_ids:
                logits_row = logits[logits_row_idx]
                token_ids = sampling_metadata.seq_data[seq_id].output_token_ids
                for logits_processor in logits_processors:
                    logits_row = logits_processor(token_ids, logits_row)
                logits[logits_row_idx] = logits_row
                logits_row_idx += 1
        else:
            logits_row_idx += len(seq_ids)
    if found_logits_processors:
        assert logits_row_idx == logits.shape[0]
    return logits


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
    treshold = torch.maximum(min_p_thresholds, top_a_thresholds)
    mask = (probs_sort < treshold.unsqueeze(1)
            )  # Cull logits below the top-a threshold
    mask.logical_or_(
        probs_sum >
        p.unsqueeze(dim=1))  # Cull logits above the top-p summation threshold
    mask[:, 0] = False  # Guarantee at least one token is pickable
    logits_sort[mask] = -float("inf")

    # Apply top-k.
    # Create a mask for the top-k elements.
    top_k_mask = torch.arange(logits_idx.shape[-1], device=logits_idx.device)
    top_k_mask = top_k_mask.expand(logits_idx.shape[0], -1)
    top_k_mask = top_k_mask >= k.unsqueeze_(dim=1)

    # Final mask.
    mask = (mask | top_k_mask)
    logits_sort.masked_fill_(mask, -float("inf"))

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
    eta = torch.tensor(eta_cutoff, dtype=logits.dtype,
                       device=logits.device) * 1e-4
    shifted_logits = torch.log_softmax(logits, dim=-1)
    probs = shifted_logits.exp()

    neg_entropy = (probs * shifted_logits).nansum(dim=-1)
    eps = torch.min(eta,
                    torch.sqrt(eta) * torch.exp(neg_entropy)).unsqueeze(dim=1)

    eta_mask = probs < eps

    if torch.all(eta_mask):  # guard against nulling out all the logits
        topk_prob, _ = torch.max(probs, dim=-1)
        eta_mask = probs < topk_prob

    logits[eta_mask] = -float("inf")
    return logits


def _apply_epsilon_cutoff(
    logits: torch.Tensor,
    epsilon_cutoff: torch.Tensor,
) -> torch.Tensor:
    eps = torch.tensor(epsilon_cutoff,
                       dtype=logits.dtype,
                       device=logits.device).unsqueeze(dim=1)
    probs = logits.softmax(dim=-1)

    eps_mask = probs < (eps * 1e-4)

    if torch.all(eps_mask):  # guard against nulling out all the logits
        topk_prob, _ = torch.max(probs, dim=-1)
        eps_mask = probs < topk_prob

    logits[eps_mask] = -float("inf")
    return logits


def _apply_typical_sampling(
    logits: torch.Tensor,
    typical_p: torch.Tensor,
) -> torch.Tensor:
    typ_p = torch.tensor(typical_p, dtype=logits.dtype, device=logits.device)
    shifted_logits = torch.log_softmax(logits, dim=-1)
    probs = shifted_logits.exp()

    neg_entropy = (probs * shifted_logits).nansum(dim=-1, keepdim=True)

    surprisal_deviations = (neg_entropy - shifted_logits).abs()
    _, indices = torch.sort(surprisal_deviations)
    reordered_probs = probs.gather(-1, indices)
    typ_mask_sorted = reordered_probs.cumsum(dim=-1) >= typ_p.unsqueeze(dim=1)

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
    smoothing_factors: torch.Tensor,
    smoothing_curves: torch.Tensor,
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
        smoothing_factors (torch.Tensor): The factors to scale the quadratic
            term in the transformation.
        smoothing_curves (torch.Tensor): The factors to scale the cubic term
            in the transformation.

    returns:
        torch.Tensor: The transformed logits.

    Credits: @kalomaze
    """
    max_logits = logits.max(dim=-1, keepdim=True).values
    diff = logits - max_logits
    smoothing_factors.unsqueeze_(dim=1)
    smoothing_curves.unsqueeze_(dim=1)

    k = (3 - smoothing_curves) / 2
    s = (smoothing_curves - 1) / 2

    mask = smoothing_factors > 0
    mask = mask.flatten()
    transformed_logits = torch.where(
        logits != float('-inf'), -(k * smoothing_factors * diff**2) +
        (s * smoothing_factors * diff**3) + max_logits, logits)
    logits[mask, :] = transformed_logits[mask, :]
    return logits


def _greedy_sample(
    selected_seq_groups: List[Tuple[List[int], SamplingParams]],
    samples: torch.Tensor,
) -> List[Tuple[List[int], List[int]]]:
    samples = samples.tolist()
    sample_idx = 0
    results = []
    for seq_group in selected_seq_groups:
        seq_ids, _ = seq_group
        num_parent_seqs = len(seq_ids)
        assert num_parent_seqs == 1, (
            "Greedy sampling should have only one seq.")
        parent_ids = list(range(num_parent_seqs))
        next_token_ids = [samples[sample_idx]]
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
        sample_idx = 0
        for (seq_ids, _), generator in zip(seq_groups, generators):
            next_sample_idx = sample_idx + len(seq_ids) * num_samples
            q[sample_idx:next_sample_idx].exponential_(generator=generator)
            sample_idx = next_sample_idx
    return probs.div_(q).argmax(dim=1).view(-1, num_samples)


def _sample(
    probs: torch.Tensor,
    logprobs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> List[Tuple[List[int], List[int]]]:
    categorized_seq_group_ids = {t: [] for t in SamplingType}
    categorized_sample_indices = sampling_metadata.categorized_sample_indices
    for i, seq_group in enumerate(sampling_metadata.seq_groups):
        _, sampling_params = seq_group
        sampling_type = sampling_params.sampling_type
        categorized_seq_group_ids[sampling_type].append(i)

    sample_results_dict: Dict[int, Tuple[List[int], List[int]]] = {}
    sample_metadata = {}
    multinomial_samples = {}

    # Counterintuitively, having two loops here is actually faster.
    # The first loop can run without waiting on GPU<->CPU sync.
    for sampling_type, sample_indices in categorized_sample_indices.items():
        if len(sample_indices) == 0:
            continue
        seq_group_ids = categorized_seq_group_ids[sampling_type]
        seq_groups = [sampling_metadata.seq_groups[i] for i in seq_group_ids]
        is_prompts = [i < sampling_metadata.num_prompts for i in seq_group_ids]
        sample_metadata[sampling_type] = (seq_group_ids, seq_groups,
                                          is_prompts, sample_indices)
        if sampling_type == SamplingType.GREEDY:
            greedy_samples = torch.argmax(logprobs[sample_indices], dim=-1)
        elif sampling_type in (SamplingType.RANDOM, SamplingType.RANDOM_SEED):
            max_best_of = 1
            for seq_group, is_prompt in zip(seq_groups, is_prompts):
                if is_prompt:
                    _, sampling_params = seq_group
                    max_best_of = max(max_best_of, sampling_params.best_of)
            seeded_args = {} if sampling_type == SamplingType.RANDOM else {
                "seq_groups": seq_groups,
                "generators": sampling_metadata.generators,
            }
            multinomial_samples[sampling_type] = _multinomial(
                probs[sample_indices], max_best_of, **seeded_args)
        elif sampling_type == SamplingType.BEAM:
            beam_search_logprobs = logprobs[sample_indices]
        else:
            raise ValueError(f"Unsupported sampling type: {sampling_type}")

    # GPU<->CPU sync happens in the loop below.

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
    return sample_results


def _get_logprobs(
    logprobs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    sample_results: List[Tuple[List[int], List[int]]],
) -> Tuple[List[Optional[List[Optional[Dict[int, float]]]]], List[List[Dict[
        int, float]]]]:
    # Prepare query indices
    batched_logprobs_query_seq_indices: List[int] = []
    batched_logprobs_query_token_indices: List[int] = []
    largest_num_logprobs = 0
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

    # Batched query for logprobs of selected token
    batched_logprobs_query_result = logprobs[[
        batched_logprobs_query_seq_indices,
        batched_logprobs_query_token_indices
    ]]

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
            prompt_len = sampling_metadata.prompt_lens[i]
            prompt_tokens = sampling_metadata.seq_data[
                seq_ids[0]].prompt_token_ids
            group_prompt_logprobs: PromptLogprobs = [None]
            for token_id in prompt_tokens[1:]:
                prompt_logprobs_dict = {
                    token_id:
                    batched_logprobs_query_result[query_result_idx].item()
                }
                if num_logprobs > 0:
                    prompt_logprobs_dict.update(
                        zip(top_token_ids[sample_idx, :num_logprobs].tolist(),
                            top_logprobs[sample_idx, :num_logprobs].tolist()))
                group_prompt_logprobs.append({
                    token_id: Logprob(logprob)
                    for token_id, logprob in prompt_logprobs_dict.items()
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
                batched_logprobs_query_result[query_result_idx].item()
            }
            query_result_idx += 1
            if num_logprobs > 0:
                sample_logprobs_dict.update(
                    zip(
                        top_token_ids[sample_idx +
                                      parent_id, :num_logprobs].tolist(),
                        top_logprobs[sample_idx +
                                     parent_id, :num_logprobs].tolist()))
            group_sample_logprobs.append({
                token_id: Logprob(logprob)
                for token_id, logprob in sample_logprobs_dict.items()
            })
        result_sample_logprobs.append(group_sample_logprobs)
        sample_idx += len(seq_ids)

    return result_prompt_logprobs, result_sample_logprobs


def _build_sampler_output(
    sample_results: List[Tuple[List[int], List[int]]],
    sampling_metadata: SamplingMetadata,
    prompt_logprobs: List[Optional[PromptLogprobs]],
    sample_logprobs: List[SampleLogprobs],
    output_metadata: OutputMetadata,
) -> SamplerOutput:
    sampler_output = []
    for (seq_group, sample_result, group_prompt_logprobs,
         group_sample_logprobs) in zip(sampling_metadata.seq_groups,
                                       sample_results, prompt_logprobs,
                                       sample_logprobs):
        seq_ids, _ = seq_group
        next_token_ids, parent_ids = sample_result
        seq_outputs = []
        for parent_id, next_token_id, logprobs in zip(parent_ids,
                                                      next_token_ids,
                                                      group_sample_logprobs):
            seq_outputs.append(
                SequenceOutput(seq_ids[parent_id], next_token_id, logprobs,
                               output_metadata.get(seq_ids[parent_id])))
        sampler_output.append(
            SequenceGroupOutput(seq_outputs, group_prompt_logprobs))
    return sampler_output


def _miro_store_args(seqids: List[int], mus: List[float],
                     output_metadata: OutputMetadata) -> None:
    for sid, mu in zip(seqids,
                       mus.tolist()):  # tolist might be premature optimization
        output_metadata.add(sid, "miro_mu", mu)


def _apply_mirostat_v2(
        logits: torch.Tensor,
        taus: torch.Tensor,  # AKA the targeted surprise
        etas: torch.Tensor,  # AKA the learning rate
        mus: torch.
    Tensor,  # AKA the accumulator that always tries to approach [tau]
) -> torch.Tensor:

    logit_surprise = torch.softmax(
        logits, dim=-1).log2_().neg_()  # Calculate surprise value per token
    # For compatibility with ooba/kobold, done in unit of bits(log base 2)
    # not nats(ln).
    # Ideally this would be a log_softmax, for numerical stability and
    # elegance purposes.
    # logit_surprise = torch.log_softmax(logits, dim=-1).neg_()

    miro_mask = logit_surprise > mus.unsqueeze(
        dim=-1)  # Mask out "too-surprising" tokens (above mu)
    mininds = torch.argmin(logit_surprise, dim=-1)
    miro_mask.scatter_(
        1, mininds.unsqueeze(dim=-1), False
    )  # Force at least one outcome to be possible, ideally the most likely one

    logits[miro_mask] = -float("inf")

    probs = torch.softmax(logits, dim=-1,
                          dtype=logits.dtype)  # Get probs, post-mask

    # NOTE: Mirostat updates its `mu` values based on the sample chosen.
    # The silly approach here is to just sample it and make the logits one-hot.
    # This breaks fine grained seeding, but we don't have that yet.
    # TODO: FIX when it gets added
    next_token_ids = _multinomial(probs, num_samples=1)

    # Calculation new `mu` values
    # NOTE: If we can know the logit values of the PREVIOUS iteration,
    # it should be possible to update `mu` before applying mirostat each
    # iteration, thus letting us keep _sample as the last thing that happens.
    picked_surprises = torch.gather(logit_surprise,
                                    dim=-1,
                                    index=next_token_ids)
    eps = picked_surprises.squeeze() - taus
    mus.sub_(etas * eps)

    logits.fill_(-float("inf"))
    # This value doesn't actually matter, so long as it's not -inf.
    # Vectors are now one-hot, after all.
    logits.scatter_(1, next_token_ids, 1.0)
    return logits


def _mirostat(logits: torch.Tensor, sampling_tensors: SamplingTensors,
              output_metadata: OutputMetadata) -> torch.Tensor:
    idx = sampling_tensors.miro_indices
    seqids = sampling_tensors.miro_seqids
    taus = sampling_tensors.miro_taus
    etas = sampling_tensors.miro_etas
    mus = sampling_tensors.miro_mus

    logits[idx] = _apply_mirostat_v2(logits[idx], taus, etas,
                                     mus)  # mus is an inout param, :vomit:
    _miro_store_args(seqids, mus, output_metadata)
    return logits
