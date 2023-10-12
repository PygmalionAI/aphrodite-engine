"""A layer that samples the next tokens from the model's outputs."""
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

from aphrodite.modeling.metadata import InputMetadata
from aphrodite.modeling.megatron.communication_op import (
    tensor_model_parallel_all_gather
)
from aphrodite.common.sampling_params import SamplingParams, SamplingType
from aphrodite.common.sequence import SamplerOutput, SequenceOutputs, SequenceData

_SAMPLING_EPS = 1e-5


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

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(
        self,
        embedding: torch.Tensor,
        hidden_states: torch.Tensor,
        input_metadata: InputMetadata,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> SamplerOutput:
        # Get the hidden states that we use for sampling.
        hidden_states = _prune_hidden_states(hidden_states, input_metadata)

        # Get the logits for the next tokens.
        logits = _get_logits(hidden_states, embedding, embedding_bias,
                             self.vocab_size)

        # Apply presence and frequency penalties.
        output_tokens = _get_output_tokens(input_metadata)
        assert len(output_tokens) == logits.shape[0]
        presence_penalties, frequency_penalties, repetition_penalties = _get_penalties(input_metadata)
        assert len(presence_penalties) == logits.shape[0]
        assert len(frequency_penalties) == logits.shape[0]
        logits = _apply_penalties(logits, output_tokens,
                                  presence_penalties, frequency_penalties, repetition_penalties,
                                  self.vocab_size)
        
        logits = _apply_logits_processors(input_metadata, logits, output_tokens)

        # Apply Eta sampling, as described in https://arxiv.org/abs/2210.15191
        eta_cutoffs = _get_eta_cutoffs(input_metadata)
        assert len(eta_cutoffs) == logits.shape[0]
        if any(eta > _SAMPLING_EPS for eta in eta_cutoffs):
            logits = _apply_eta_cutoff(logits, eta_cutoffs)

        # Apply Locally typical sampling, as described in https://arxiv.org/abs/2202.00666
        typical_ps = _get_typical_ps(input_metadata)
        assert len(typical_ps) == logits.shape[0]
        if any(typ_p < 1.0 - _SAMPLING_EPS for typ_p in typical_ps):
            logits = _apply_typical_sampling(logits, typical_ps)

        # Apply Tail Free Sampling, as described in https://www.trentonbricken.com/Tail-Free-Sampling/
        tfss = _get_tfs(input_metadata)
        assert len(tfss) == logits.shape[0]
        if any(z < 1.0 - _SAMPLING_EPS for z in tfss):
            logits = _apply_tfs(logits, tfss)

        epsilon_cutoffs = _get_epsilon_cutoffs(input_metadata)
        assert len(epsilon_cutoffs) == logits.shape[0]
        if any(epsilon > _SAMPLING_EPS for epsilon in epsilon_cutoffs):
            logits = _apply_epsilon_cutoff(logits, epsilon_cutoffs)

        # Apply temperature scaling.
        temperatures = _get_temperatures(input_metadata)
        assert len(temperatures) == logits.shape[0]
        if any(t != 1.0 for t in temperatures):
            t = torch.tensor(temperatures,
                             dtype=logits.dtype,
                             device=logits.device)
            # Use in-place division to avoid creating a new tensor.
            logits.div_(t.unsqueeze(dim=1))

        # Apply top-p, top-k, and top-a truncation.
        top_ps, top_ks, top_as = _get_top_a_top_p_top_k(input_metadata, self.vocab_size)
        assert len(top_ps) == len(top_ks) == logits.shape[0]
        do_top_p = any(p < 1.0 - _SAMPLING_EPS for p in top_ps)
        do_top_k = any(k != self.vocab_size for k in top_ks)
        do_top_a = any(a > _SAMPLING_EPS for a in top_as)
        if do_top_p or do_top_k or do_top_a:
            logits = _apply_top_a_top_p_top_k(logits, top_ps, top_ks, top_as)

        # We use float32 for probabilities and log probabilities.
        # Compute the probabilities.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # Compute the log probabilities.
        # Use log_softmax to ensure numerical stability.
        logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)

        # Sample the next tokens.
        return _sample(probs, logprobs, input_metadata)


def _get_logits(hidden_states: torch.Tensor, embedding: torch.Tensor,
                embedding_bias: Optional[torch.Tensor],
                vocab_size: int) -> torch.Tensor:
    # Get the logits for the next tokens.
    logits = torch.matmul(hidden_states, embedding.t())
    if embedding_bias is not None:
        logits += embedding_bias
    logits = tensor_model_parallel_all_gather(logits)
    # Remove paddings in vocab (if any).
    logits = logits[:, :vocab_size]
    return logits


def _prune_hidden_states(
    hidden_states: torch.Tensor,
    input_metadata: InputMetadata,
) -> torch.Tensor:
    last_token_indices = []
    start_idx = 0
    for i, seq_group in enumerate(input_metadata.seq_groups):
        seq_ids, _ = seq_group
        if i < input_metadata.num_prompts:
            assert len(seq_ids) == 1, "Prompt input should have only one seq."
            prompt_len = input_metadata.prompt_lens[i]
            last_token_indices.append(start_idx + prompt_len - 1)
            start_idx += prompt_len
        else:
            num_seqs = len(seq_ids)
            last_token_indices.extend(range(start_idx, start_idx + num_seqs))
            start_idx += num_seqs

    last_token_indices = torch.tensor(last_token_indices,
                                      dtype=torch.long,
                                      device=hidden_states.device)
    return hidden_states.index_select(0, last_token_indices)


def _get_penalties(
        input_metadata: InputMetadata) -> Tuple[List[float], List[float]]:
    # Collect the presence and frequency penalties.
    presence_penalties: List[float] = []
    frequency_penalties: List[float] = []
    repetition_penalties: List[float] = []
    for seq_group in input_metadata.seq_groups:
        seq_ids, sampling_params = seq_group
        presence_penalties += [sampling_params.presence_penalty] * len(seq_ids)
        frequency_penalties += [sampling_params.frequency_penalty] * len(seq_ids)
        repetition_penalties += [sampling_params.repetition_penalty] * len(seq_ids)
    return presence_penalties, frequency_penalties, repetition_penalties


def _get_output_tokens(input_metadata: InputMetadata) -> List[List[int]]:
    output_tokens: List[List[int]] = []
    for seq_group in input_metadata.seq_groups:
        seq_ids, _ = seq_group
        for seq_id in seq_ids:
            seq_data = input_metadata.seq_data[seq_id]
            output_tokens.append(seq_data.output_token_ids)
    return output_tokens

def _apply_logits_processors(
    input_metadata: InputMetadata,
    logits: torch.Tensor,
    output_tokens: List[List[int]]
) -> torch.Tensor:
    for _, seq_group in enumerate(input_metadata.seq_groups):
        _, sampling_params = seq_group
        logits_processors = sampling_params.logits_processors

        if logits_processors is not None:
            for logits_processor in logits_processors:
                logits = logits_processor(logits, output_tokens)

    return logits

def _apply_penalties(
    logits: torch.Tensor,
    output_tokens: List[List[int]],
    presence_penalties: List[float],
    frequency_penalties: List[float],
    repetition_penalties: List[float],
    vocab_size: int,
) -> torch.Tensor:
    num_seqs, vocab_size = logits.shape
    for i in range(num_seqs):
        if not output_tokens[i]:
            continue
        if (abs(presence_penalties[i]) < _SAMPLING_EPS and
            abs(frequency_penalties[i]) < _SAMPLING_EPS and
            repetition_penalties[i] < 1.0 + _SAMPLING_EPS):
            continue
        break
    else:
        # Return early if all sequences have zero penalties.
        return logits

    max_output_len = max(len(tokens) for tokens in output_tokens)
    padded_output_tokens = [
        tokens + [vocab_size] * (max_output_len - len(tokens))
        for tokens in output_tokens
    ]
    output_tokens_tensor = torch.tensor(padded_output_tokens,
                                        dtype=torch.long,
                                        device=logits.device)

    # Compute the bin counts for the output tokens.
    # vocab_size + 1 for padding.
    bin_counts = torch.zeros((num_seqs, vocab_size + 1),
                             dtype=torch.long,
                             device=logits.device)
    bin_counts.scatter_add_(1, output_tokens_tensor,
                            torch.ones_like(output_tokens_tensor))
    bin_counts = bin_counts[:, :vocab_size]  # Remove the padding bin.

    frequency_penalties = torch.tensor(frequency_penalties,
                                       dtype=logits.dtype,
                                       device=logits.device)
    presence_penalties = torch.tensor(presence_penalties,
                                      dtype=logits.dtype,
                                      device=logits.device)
    repetition_penalties = torch.tensor(repetition_penalties,
                                      dtype=logits.dtype,
                                      device=logits.device)

    # We follow the definition in OpenAI API.
    # Refer to https://platform.openai.com/docs/api-reference/parameter-details
    logits -= frequency_penalties.unsqueeze(dim=1) * bin_counts
    presence_mask = (bin_counts > 0)
    logits -= presence_penalties.unsqueeze(dim=1) * presence_mask

    # Effectively: If token is present and logit is positive, divide logit by rep_pen.
    #              If token is present and logit is negative, multiply logit by rep_pen.
    logits += logits * (1 / repetition_penalties.unsqueeze(dim=1) - 1) * presence_mask * (logits > 0)
    logits += logits * (repetition_penalties.unsqueeze(dim=1) - 1) * presence_mask * (logits < 0)

    return logits


def _get_temperatures(input_metadata: InputMetadata) -> List[float]:
    # Collect the temperatures for the logits.
    temperatures: List[float] = []
    for seq_group in input_metadata.seq_groups:
        seq_ids, sampling_params = seq_group
        temperature = sampling_params.temperature
        if temperature < _SAMPLING_EPS:
            # NOTE: Zero temperature means deterministic sampling
            # (i.e., greedy sampling or beam search).
            # Set the temperature to 1 to avoid division by zero.
            temperature = 1.0
        temperatures += [temperature] * len(seq_ids)
    return temperatures


def _get_top_a_top_p_top_k(
    input_metadata: InputMetadata,
    vocab_size: int,
) -> Tuple[List[float], List[int], List[float]]:
    top_ps: List[float] = []
    top_ks: List[int] = []
    top_as: List[float] = []
    for seq_group in input_metadata.seq_groups:
        seq_ids, sampling_params = seq_group
        # k should not be greater than the vocab size.
        top_k = min(sampling_params.top_k, vocab_size)
        # k=-1 means no truncation.
        top_k = vocab_size if top_k == -1 else top_k

        top_ps += [sampling_params.top_p] * len(seq_ids)
        top_ks += [top_k] * len(seq_ids)
        top_as += [sampling_params.top_a] * len(seq_ids)

    return top_ps, top_ks, top_as



def _get_tfs(input_metadata: InputMetadata) -> List[float]:
    tfss: List[float] = []
    for seq_group in input_metadata.seq_groups:
        seq_ids, sampling_params = seq_group
        z = sampling_params.tfs
        tfss += [z] * len(seq_ids)
    return tfss


def _get_eta_cutoffs(input_metadata: InputMetadata) -> List[float]:
    eta_cutoffs: List[float] = []
    for seq_group in input_metadata.seq_groups:
        seq_ids, sampling_params = seq_group
        eta_cutoff = sampling_params.eta_cutoff
        eta_cutoffs += [eta_cutoff] * len(seq_ids)
    return eta_cutoffs


def _get_epsilon_cutoffs(input_metadata: InputMetadata) -> List[float]:
    epsilon_cutoffs: List[float] = []
    for seq_group in input_metadata.seq_groups:
        seq_ids, sampling_params = seq_group
        epsilon_cutoff = sampling_params.epsilon_cutoff
        epsilon_cutoffs += [epsilon_cutoff] * len(seq_ids)
    return epsilon_cutoffs


def _get_typical_ps(input_metadata: InputMetadata) -> List[float]:
    typical_ps: List[float] = []
    for seq_group in input_metadata.seq_groups:
        seq_ids, sampling_params = seq_group
        typical_p = sampling_params.typical_p
        typical_ps += [typical_p] * len(seq_ids)
    return typical_ps


def _apply_top_a_top_p_top_k(
    logits: torch.Tensor,
    top_ps: List[float],
    top_ks: List[int],
    top_as: List[float],
) -> torch.Tensor:
    ts_p = torch.tensor(top_ps, dtype=logits.dtype, device=logits.device)
    ts_k = torch.tensor(top_ks, dtype=torch.int, device=logits.device)
    ts_a = torch.tensor(top_as, dtype=logits.dtype, device=logits.device)
    logits_sort, logits_idx = logits.sort(dim=-1, descending=True)

    # Apply top-p and top-a.
    probs_sort = logits_sort.softmax(dim=-1)
    probs_sum = probs_sort.cumsum(dim=-1)
    top_a_thresholds = torch.pow(probs_sort[:, 0], 2) * ts_a
    top_ap_mask = (probs_sort < top_a_thresholds.unsqueeze(1)) # Cull logits below the top-a threshold
    top_ap_mask.logical_or_(probs_sum > ts_p.unsqueeze(dim=1)) # Cull logits above the top-p summation threshold
    top_ap_mask[:, 0] = False # Guarantee at least one token is pickable
    logits_sort[top_ap_mask] = -float("inf")
    
    # Apply top-k.
    # Create a mask for the top-k elements.
    top_k_mask = torch.arange(logits_idx.shape[-1], device=logits_idx.device)
    top_k_mask = top_k_mask.expand(logits_idx.shape[0], -1)
    top_k_mask = top_k_mask >= ts_k.unsqueeze(dim=1)
    logits_sort[top_k_mask] = -float("inf")

    # Re-sort the probabilities.
    logits = torch.gather(logits_sort,
                          dim=-1,
                          index=torch.argsort(logits_idx, dim=-1))
    return logits

def _apply_tfs(
    logits: torch.Tensor,
    tfss: List[float],
) -> torch.Tensor:
    z = torch.tensor(tfss, dtype=logits.dtype, device=logits.device)
    logits_sort, logits_idx = logits.sort(dim=-1, descending=True)
    d2 = logits_sort.softmax(dim=-1).diff().diff().abs()
    normalized_d2 = d2 / torch.sum(d2, dim=-1, keepdim=True)
    curvature_cdf = torch.cumsum(normalized_d2, dim=-1)

    tfs_mask = curvature_cdf > z.unsqueeze(dim=-1)

    tfs_mask = torch.cat(
            (
                torch.zeros(logits.shape[0], 1, dtype=torch.bool, device=logits.device),
                tfs_mask,
                torch.ones(logits.shape[0], 1, dtype=torch.bool, device=logits.device),
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
    eta_cutoffs: List[float],
) -> torch.Tensor:
    eta = torch.tensor(eta_cutoffs, dtype=logits.dtype, device=logits.device) * 1e-4
    shifted_logits = torch.log_softmax(logits, dim=-1)
    probs = shifted_logits.exp()

    neg_entropy = (probs * shifted_logits).nansum(dim=-1)
    eps = torch.min(eta, torch.sqrt(eta)*torch.exp(neg_entropy)).unsqueeze(dim=1)

    eta_mask = probs < eps

    if(torch.all(eta_mask)): # guard against nulling out all the logits
        topk_prob, _ = torch.max(probs, dim=-1)
        eta_mask = probs < topk_prob

    logits[eta_mask] = -float("inf")
    return logits


def _apply_epsilon_cutoff(
    logits: torch.Tensor,
    epsilon_cutoffs: List[float],
) -> torch.Tensor:
    eps = torch.tensor(epsilon_cutoffs, dtype=logits.dtype, device=logits.device).unsqueeze(dim=1)
    probs = logits.softmax(dim=-1)

    eps_mask = probs < (eps * 1e-4)

    if(torch.all(eps_mask)): # guard against nulling out all the logits
        topk_prob, _ = torch.max(probs, dim=-1)
        eps_mask = probs < topk_prob

    logits[eps_mask] = -float("inf")
    return logits


def _apply_typical_sampling(
    logits: torch.Tensor,
    typical_ps: List[float],
) -> torch.Tensor:
    typ_p = torch.tensor(typical_ps, dtype=logits.dtype, device=logits.device)
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

    typ_mask = typ_mask_sorted.scatter(
        1, indices, typ_mask_sorted
    )
    logits[typ_mask] = -float("inf")
    return logits


def _get_topk_logprobs(
    logprobs: torch.Tensor,
    num_logprobs: Optional[int],
) -> List[Dict[int, float]]:
    num_seqs = logprobs.size(0)
    if num_logprobs is None or num_logprobs == 0:
        return [{} for _ in range(num_seqs)]

    all_topk_logprobs, all_topk_ids = torch.topk(logprobs,
                                                 num_logprobs,
                                                 dim=-1)
    all_topk_logprobs = all_topk_logprobs.cpu()
    all_topk_ids = all_topk_ids.cpu()
    all_token_to_logprob = []
    for topk_logprobs, topk_ids in zip(all_topk_logprobs, all_topk_ids):
        token_to_logprob: Dict[int, float] = {}
        for token_id, logprob in zip(topk_ids, topk_logprobs):
            token_to_logprob[token_id.item()] = logprob.item()
        all_token_to_logprob.append(token_to_logprob)
    return all_token_to_logprob


def _build_sequence_outputs(
    parent_ids: List[int],
    next_token_ids: List[int],
    selected_token_logprobs: torch.Tensor,
    parent_seq_ids: List[int],
    parent_logprobs: torch.Tensor,
    num_output_logprobs: Optional[int],
) -> List[SequenceOutputs]:
    # Get top-k log probabilities for the next tokens.
    next_logprobs = _get_topk_logprobs(parent_logprobs, num_output_logprobs)
    seq_outputs: List[SequenceOutputs] = []
    for parent_id, next_token_id, token_logprob in zip(
            parent_ids, next_token_ids, selected_token_logprobs):
        output_logprobs = next_logprobs[parent_id].copy()
        output_logprobs[next_token_id] = token_logprob
        seq_outputs.append(
            SequenceOutputs(parent_seq_ids[parent_id], next_token_id,
                            output_logprobs))
    return seq_outputs


def _greedy_sample(
    selected_seq_groups: List[Tuple[List[int], SamplingParams]],
    logprobs: torch.Tensor,
) -> List[Tuple[List[int], List[int]]]:
    samples = torch.argmax(logprobs, dim=-1).cpu()
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
    assert sample_idx == logprobs.size(0)
    return results


def _random_sample(
    selected_seq_groups: List[Tuple[List[int], SamplingParams]],
    is_prompts: List[bool],
    probs: torch.Tensor,
) -> List[Tuple[List[int], List[int]]]:
    # Find the maximum best_of value of the prompt phase requests.
    max_best_of = 1
    for seq_group, is_prompt in zip(selected_seq_groups, is_prompts):
        if is_prompt:
            seq_ids, sampling_params = seq_group
            max_best_of = max(max_best_of, sampling_params.best_of)
    random_samples = torch.multinomial(probs,
                                       num_samples=max_best_of,
                                       replacement=True).cpu()
    sample_idx = 0
    results = []
    for seq_group, is_prompt in zip(selected_seq_groups, is_prompts):
        seq_ids, sampling_params = seq_group
        num_parent_seqs = len(seq_ids)
        if is_prompt:
            # Prompt phase.
            assert num_parent_seqs == 1, (
                "Prompt input should have only one seq.")
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
    assert sample_idx == probs.size(0)
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


def _sample(
    probs: torch.Tensor,
    logprobs: torch.Tensor,
    input_metadata: InputMetadata,
) -> SamplerOutput:
    categorized_seq_group_ids = {t: [] for t in SamplingType}
    start_idx = 0
    categorized_seq_ids = {t: [] for t in SamplingType}
    for i, seq_group in enumerate(input_metadata.seq_groups):
        seq_ids, sampling_params = seq_group
        sampling_type = sampling_params.sampling_type
        categorized_seq_group_ids[sampling_type].append(i)
        num_seqs = len(seq_ids)
        categorized_seq_ids[sampling_type].extend(
            range(start_idx, start_idx + num_seqs))
        start_idx += num_seqs

    seq_outputs_dict: Dict[int, List[SequenceOutputs]] = {}
    for sampling_type in SamplingType:
        seq_group_ids = categorized_seq_group_ids[sampling_type]
        seq_groups = [input_metadata.seq_groups[i] for i in seq_group_ids]
        is_prompts = [i < input_metadata.num_prompts for i in seq_group_ids]
        num_tokens = len(categorized_seq_ids[sampling_type])
        if num_tokens == 0:
            continue
        category_logprobs = logprobs[categorized_seq_ids[sampling_type]]
        category_probs = probs[categorized_seq_ids[sampling_type]]
        if sampling_type == SamplingType.GREEDY:
            sample_results = _greedy_sample(seq_groups, category_logprobs)
        elif sampling_type == SamplingType.RANDOM:
            sample_results = _random_sample(seq_groups, is_prompts,
                                            category_probs)
        elif sampling_type == SamplingType.BEAM:
            sample_results = _beam_search_sample(seq_groups, is_prompts,
                                                 input_metadata.seq_data,
                                                 category_logprobs)
        else:
            raise ValueError(f"Unsupported sampling type: {sampling_type}")

        # Batched query for logprobs of selected token
        batched_logprobs_query_seq_indices: List[int] = []
        batched_logprobs_query_token_indices: List[int] = []
        sample_idx = 0
        for seq_group_id, seq_group, sample_result in zip(
                seq_group_ids, seq_groups, sample_results):
            seq_ids, sampling_params = seq_group
            next_token_ids, parent_ids = sample_result
            num_parent_seqs = len(seq_ids)
            batched_logprobs_query_seq_indices.extend(
                [sample_idx + parent_id for parent_id in parent_ids])
            batched_logprobs_query_token_indices.extend(next_token_ids)
            sample_idx += num_parent_seqs
        assert sample_idx == num_tokens
        batched_logprobs_query_result = category_logprobs[[
            batched_logprobs_query_seq_indices,
            batched_logprobs_query_token_indices
        ]].tolist()

        # Build the sequence outputs.
        sample_idx = 0
        result_idx = 0
        for seq_group_id, seq_group, sample_result in zip(
                seq_group_ids, seq_groups, sample_results):
            seq_ids, sampling_params = seq_group
            next_token_ids, parent_ids = sample_result
            num_results = len(next_token_ids)
            num_parent_seqs = len(seq_ids)
            parent_logprobs = category_logprobs[sample_idx:sample_idx +
                                                num_parent_seqs]
            selected_token_logprobs = batched_logprobs_query_result[
                result_idx:result_idx + num_results]
            seq_output = _build_sequence_outputs(parent_ids, next_token_ids,
                                                 selected_token_logprobs,
                                                 seq_ids, parent_logprobs,
                                                 sampling_params.logprobs)
            seq_outputs_dict[seq_group_id] = seq_output
            sample_idx += num_parent_seqs
            result_idx += num_results
        assert sample_idx == num_tokens

    return [seq_outputs_dict[i] for i in range(len(input_metadata.seq_groups))]