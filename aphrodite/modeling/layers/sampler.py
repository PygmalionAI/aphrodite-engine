"""A layer that samples the next tokens from the model's outputs."""
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn

from aphrodite.modeling.metadata import InputMetadata, OutputMetadata
from aphrodite.modeling.megatron.communication_op import (
    tensor_model_parallel_all_gather)
from aphrodite.common.sampling_params import SamplingParams, SamplingType
from aphrodite.common.sequence import (PromptLogprobs, SampleLogprobs,
                                       SamplerOutput, SequenceData,
                                       SequenceGroupOutputs, SequenceOutputs)

import aphrodite.modeling.layers.sampler_mirostat as sampler_mirostat

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

        output_metadata = OutputMetadata()

        # Apply presence and frequency penalties.
        output_tokens = _get_output_tokens(input_metadata)
        assert len(output_tokens) == logits.shape[0]
        [presence_penalties, frequency_penalties,
         repetition_penalties] = _get_penalties(input_metadata)
        assert len(presence_penalties) == logits.shape[0]
        assert len(frequency_penalties) == logits.shape[0]
        logits = _apply_penalties(logits, output_tokens, presence_penalties,
                                  frequency_penalties, repetition_penalties,
                                  self.vocab_size)

        banned_tokens = _get_custom_token_bans(input_metadata)
        assert len(banned_tokens) == logits.shape[0]
        logits = _apply_token_bans(logits, banned_tokens)

        logits = _apply_logits_processors(input_metadata, logits,
                                          output_tokens)

        # Apply Mirostat
        # Note that we apply mirostat before temperature, not after
        # like it maybe should be.
        # To be fixed by implementing customizable sampling order
        if sampler_mirostat.is_applicable(input_metadata):
            sampler_mirostat.apply(logits, input_metadata, output_metadata)

        # Apply Eta sampling, as described in https://arxiv.org/abs/2210.15191
        eta_cutoffs = _get_eta_cutoffs(input_metadata)
        assert len(eta_cutoffs) == logits.shape[0]
        if any(eta > _SAMPLING_EPS for eta in eta_cutoffs):
            logits = _apply_eta_cutoff(logits, eta_cutoffs)

        # Apply Locally typical sampling, as described in
        # https://arxiv.org/abs/2202.00666
        typical_ps = _get_typical_ps(input_metadata)
        assert len(typical_ps) == logits.shape[0]
        if any(typ_p < 1.0 - _SAMPLING_EPS for typ_p in typical_ps):
            logits = _apply_typical_sampling(logits, typical_ps)

        # Apply Tail Free Sampling, as described in
        # https://www.trentonbricken.com/Tail-Free-Sampling/
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
        top_ps, top_ks, top_as = _get_top_a_top_p_top_k(
            input_metadata, self.vocab_size)
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
        sample_results = _sample(probs, logprobs, input_metadata)
        # Get the logprobs query results.
        prompt_logprobs, sample_logprobs = _get_logprobs(
            logprobs, input_metadata, sample_results)
        return _build_sampler_output(sample_results, input_metadata,
                                     prompt_logprobs, sample_logprobs,
                                     output_metadata)


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
    selected_token_indices: List[int] = []
    start_idx = 0
    for i, seq_group in enumerate(input_metadata.seq_groups):
        seq_ids, sampling_params = seq_group
        if i < input_metadata.num_prompts:
            assert len(seq_ids) == 1, "Prompt input should have only one seq."
            prompt_len = input_metadata.prompt_lens[i]
            if sampling_params.prompt_logprobs is not None:
                selected_token_indices.extend(
                    range(start_idx, start_idx + prompt_len - 1))
            selected_token_indices.append(start_idx + prompt_len - 1)
            start_idx += input_metadata.max_prompt_len
        else:
            num_seqs = len(seq_ids)
            selected_token_indices.extend(
                range(start_idx, start_idx + num_seqs))
            start_idx += num_seqs

    selected_token_indices = torch.tensor(selected_token_indices,
                                          dtype=torch.long,
                                          device=hidden_states.device)
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    return hidden_states.index_select(0, selected_token_indices)


def _get_penalties(
        input_metadata: InputMetadata) -> Tuple[List[float], List[float]]:
    # Collect the presence and frequency penalties.
    presence_penalties: List[float] = []
    frequency_penalties: List[float] = []
    repetition_penalties: List[float] = []
    for i, seq_group in enumerate(input_metadata.seq_groups):
        seq_ids, sampling_params = seq_group
        if (i < input_metadata.num_prompts
                and sampling_params.prompt_logprobs is not None):
            prompt_len = input_metadata.prompt_lens[i]
            presence_penalties += [0] * (prompt_len - 1)
            frequency_penalties += [0] * (prompt_len - 1)
            repetition_penalties += [0] * (prompt_len - 1)
        presence_penalties += [sampling_params.presence_penalty] * len(seq_ids)
        frequency_penalties += [sampling_params.frequency_penalty
                                ] * len(seq_ids)
        repetition_penalties += [sampling_params.repetition_penalty
                                 ] * len(seq_ids)
    return presence_penalties, frequency_penalties, repetition_penalties


def _get_output_tokens(input_metadata: InputMetadata) -> List[List[int]]:
    output_tokens: List[List[int]] = []
    for i, seq_group in enumerate(input_metadata.seq_groups):
        seq_ids, sampling_params = seq_group
        if (i < input_metadata.num_prompts
                and sampling_params.prompt_logprobs is not None):
            # NOTE: prompt token positions do not need output tokens to
            # compute penalties.
            prompt_len = input_metadata.prompt_lens[i]
            output_tokens.extend([] for _ in range(prompt_len - 1))
        for seq_id in seq_ids:
            seq_data = input_metadata.seq_data[seq_id]
            output_tokens.append(seq_data.output_token_ids)
    return output_tokens


def _get_custom_token_bans(input_metadata: InputMetadata) -> List[List[int]]:
    banned_tokens: List[List[int]] = []
    for i, seq_group in enumerate(input_metadata.seq_groups):
        seq_ids, sampling_params = seq_group
        custom_token_bans = sampling_params.custom_token_bans
        if (i < input_metadata.num_prompts
                and sampling_params.prompt_logprobs is not None):
            prompt_len = input_metadata.prompt_lens[i]
            banned_tokens += [custom_token_bans] * (prompt_len - 1)
        banned_tokens += [custom_token_bans] * len(seq_ids)
    return banned_tokens


def _apply_logits_processors(input_metadata: InputMetadata,
                             logits: torch.Tensor,
                             output_tokens: List[List[int]]) -> torch.Tensor:
    seq_offset = 0

    for seq_ids, sampling_params in input_metadata.seq_groups:
        seq_end = seq_offset + len(seq_ids)

        for proc in sampling_params.logits_processors:
            proc(logits[seq_offset:seq_end], output_tokens[seq_offset:seq_end])

        seq_offset = seq_end

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
        if (abs(presence_penalties[i]) < _SAMPLING_EPS
                and abs(frequency_penalties[i]) < _SAMPLING_EPS
                and repetition_penalties[i] < 1.0 + _SAMPLING_EPS):
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

    # Effectively:
    # If token is present and logit is positive, divide logit by rep_pen.
    # If token is present and logit is negative, multiply logit by rep_pen.
    logits += logits * (1 / repetition_penalties.unsqueeze(dim=1) -
                        1) * presence_mask * (logits > 0)
    logits += logits * (repetition_penalties.unsqueeze(dim=1) -
                        1) * presence_mask * (logits < 0)

    return logits


def _apply_token_bans(logits: torch.Tensor,
                      banned_tokens: List[List[int]]) -> torch.Tensor:
    for i, banned_token_ids in enumerate(banned_tokens):
        if not banned_token_ids:
            continue
        logits[i, banned_token_ids] = -float("inf")
    return logits


def _get_temperatures(input_metadata: InputMetadata) -> List[float]:
    # Collect the temperatures for the logits.
    temperatures: List[float] = []
    for i, seq_group in enumerate(input_metadata.seq_groups):
        seq_ids, sampling_params = seq_group
        temperature = sampling_params.temperature
        if temperature < _SAMPLING_EPS:
            # NOTE: Zero temperature means deterministic sampling
            # (i.e., greedy sampling or beam search).
            # Set the temperature to 1 to avoid division by zero.
            temperature = 1.0
        if (i < input_metadata.num_prompts
                and sampling_params.prompt_logprobs is not None):
            prompt_len = input_metadata.prompt_lens[i]
            temperatures += [temperature] * (prompt_len - 1)
        temperatures += [temperature] * len(seq_ids)
    return temperatures


def _get_top_a_top_p_top_k(
    input_metadata: InputMetadata,
    vocab_size: int,
) -> Tuple[List[float], List[int], List[float]]:
    top_ps: List[float] = []
    top_ks: List[int] = []
    top_as: List[float] = []
    for i, seq_group in enumerate(input_metadata.seq_groups):
        seq_ids, sampling_params = seq_group
        # k should not be greater than the vocab size.
        top_k = min(sampling_params.top_k, vocab_size)
        # k=-1 means no truncation.
        top_k = vocab_size if top_k == -1 else top_k
        if (i < input_metadata.num_prompts
                and sampling_params.prompt_logprobs is not None):
            prompt_len = input_metadata.prompt_lens[i]
            top_ps += [sampling_params.top_p] * (prompt_len - 1)
            top_ks += [top_k] * (prompt_len - 1)
            top_as += [sampling_params.top_a] * (prompt_len - 1)
        top_ps += [sampling_params.top_p] * len(seq_ids)
        top_ks += [top_k] * len(seq_ids)
        top_as += [sampling_params.top_a] * len(seq_ids)

    return top_ps, top_ks, top_as


def _get_tfs(input_metadata: InputMetadata) -> List[float]:
    tfss: List[float] = []
    for i, seq_group in enumerate(input_metadata.seq_groups):
        seq_ids, sampling_params = seq_group
        z = sampling_params.tfs
        if (i < input_metadata.num_prompts
                and sampling_params.prompt_logprobs is not None):
            prompt_len = input_metadata.prompt_lens[i]
            tfss += [z] * (prompt_len - 1)
        tfss += [z] * len(seq_ids)
    return tfss


def _get_eta_cutoffs(input_metadata: InputMetadata) -> List[float]:
    eta_cutoffs: List[float] = []
    for i, seq_group in enumerate(input_metadata.seq_groups):
        seq_ids, sampling_params = seq_group
        eta_cutoff = sampling_params.eta_cutoff
        if (i < input_metadata.num_prompts
                and sampling_params.prompt_logprobs is not None):
            prompt_len = input_metadata.prompt_lens[i]
            eta_cutoffs += [eta_cutoff] * (prompt_len - 1)
        eta_cutoffs += [eta_cutoff] * len(seq_ids)
    return eta_cutoffs


def _get_epsilon_cutoffs(input_metadata: InputMetadata) -> List[float]:
    epsilon_cutoffs: List[float] = []
    for i, seq_group in enumerate(input_metadata.seq_groups):
        seq_ids, sampling_params = seq_group
        epsilon_cutoff = sampling_params.epsilon_cutoff
        if (i < input_metadata.num_prompts
                and sampling_params.prompt_logprobs is not None):
            prompt_len = input_metadata.prompt_lens[i]
            epsilon_cutoffs += [epsilon_cutoff] * (prompt_len - 1)
        epsilon_cutoffs += [epsilon_cutoff] * len(seq_ids)
    return epsilon_cutoffs


def _get_typical_ps(input_metadata: InputMetadata) -> List[float]:
    typical_ps: List[float] = []
    for i, seq_group in enumerate(input_metadata.seq_groups):
        seq_ids, sampling_params = seq_group
        typical_p = sampling_params.typical_p
        if (i < input_metadata.num_prompts
                and sampling_params.prompt_logprobs is not None):
            prompt_len = input_metadata.prompt_lens[i]
            typical_ps += [typical_p] * (prompt_len - 1)
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
    top_ap_mask = (probs_sort < top_a_thresholds.unsqueeze(1)
                   )  # Cull logits below the top-a threshold
    top_ap_mask.logical_or_(probs_sum > ts_p.unsqueeze(
        dim=1))  # Cull logits above the top-p summation threshold
    top_ap_mask[:, 0] = False  # Guarantee at least one token is pickable
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
    eta_cutoffs: List[float],
) -> torch.Tensor:
    eta = torch.tensor(eta_cutoffs, dtype=logits.dtype,
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
    epsilon_cutoffs: List[float],
) -> torch.Tensor:
    eps = torch.tensor(epsilon_cutoffs,
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

    typ_mask = typ_mask_sorted.scatter(1, indices, typ_mask_sorted)
    logits[typ_mask] = -float("inf")
    return logits


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
) -> List[Tuple[List[int], List[int]]]:
    categorized_seq_group_ids = {t: [] for t in SamplingType}
    categorized_sample_indices = {t: [] for t in SamplingType}
    start_idx = 0
    for i, seq_group in enumerate(input_metadata.seq_groups):
        seq_ids, sampling_params = seq_group
        sampling_type = sampling_params.sampling_type
        if (i < input_metadata.num_prompts
                and sampling_params.prompt_logprobs is not None):
            prompt_len = input_metadata.prompt_lens[i]
            start_idx += prompt_len - 1
        categorized_seq_group_ids[sampling_type].append(i)
        num_seqs = len(seq_ids)
        categorized_sample_indices[sampling_type].extend(
            range(start_idx, start_idx + num_seqs))
        start_idx += num_seqs

    sample_results_dict: Dict[int, Tuple[List[int], List[int]]] = {}
    for sampling_type in SamplingType:
        seq_group_ids = categorized_seq_group_ids[sampling_type]
        seq_groups = [input_metadata.seq_groups[i] for i in seq_group_ids]
        is_prompts = [i < input_metadata.num_prompts for i in seq_group_ids]
        sample_indices = categorized_sample_indices[sampling_type]
        num_tokens = len(sample_indices)
        if num_tokens == 0:
            continue
        if sampling_type == SamplingType.GREEDY:
            category_logprobs = logprobs[sample_indices]
            sample_results = _greedy_sample(seq_groups, category_logprobs)
        elif sampling_type == SamplingType.RANDOM:
            category_probs = probs[sample_indices]
            sample_results = _random_sample(seq_groups, is_prompts,
                                            category_probs)
        elif sampling_type == SamplingType.BEAM:
            category_logprobs = logprobs[sample_indices]
            sample_results = _beam_search_sample(seq_groups, is_prompts,
                                                 input_metadata.seq_data,
                                                 category_logprobs)
        else:
            raise ValueError(f"Unsupported sampling type: {sampling_type}")

        sample_results_dict.update(zip(seq_group_ids, sample_results))

    sample_results = [
        sample_results_dict[i] for i in range(len(input_metadata.seq_groups))
    ]
    return sample_results


def _get_logprobs(
    logprobs: torch.Tensor,
    input_metadata: InputMetadata,
    sample_results: List[Tuple[List[int], List[int]]],
) -> Tuple[List[Optional[List[Optional[Dict[int, float]]]]], List[List[Dict[
        int, float]]]]:
    # Prepare query indices
    batched_logprobs_query_seq_indices: List[int] = []
    batched_logprobs_query_token_indices: List[int] = []
    largest_num_logprobs = 0
    sample_idx = 0
    for i, (seq_group, sample_result) in enumerate(
            zip(input_metadata.seq_groups, sample_results)):
        seq_ids, sampling_params = seq_group
        next_token_ids, parent_ids = sample_result
        num_parent_seqs = len(seq_ids)
        if (i < input_metadata.num_prompts
                and sampling_params.prompt_logprobs is not None):
            largest_num_logprobs = max(largest_num_logprobs,
                                       sampling_params.prompt_logprobs)
            prompt_len = input_metadata.prompt_lens[i]
            prompt_tokens = input_metadata.seq_data[
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
    ]].cpu()

    # Batched query for logprobs of topk tokens
    if largest_num_logprobs > 0:
        top_logprobs, top_token_ids = torch.topk(logprobs,
                                                 largest_num_logprobs,
                                                 dim=-1)
        top_logprobs = top_logprobs.cpu()
        top_token_ids = top_token_ids.cpu()
    else:
        top_logprobs, top_token_ids = None, None

    # Gather results
    result_prompt_logprobs: List[Optional[PromptLogprobs]] = []
    result_sample_logprobs: List[SampleLogprobs] = []
    sample_idx = 0
    query_result_idx = 0
    for i, (seq_group, sample_result) in enumerate(
            zip(input_metadata.seq_groups, sample_results)):
        seq_ids, sampling_params = seq_group
        next_token_ids, parent_ids = sample_result

        # Prompt logprobs
        if (i < input_metadata.num_prompts
                and sampling_params.prompt_logprobs is not None):
            num_logprobs = sampling_params.prompt_logprobs
            prompt_len = input_metadata.prompt_lens[i]
            prompt_tokens = input_metadata.seq_data[
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
                group_prompt_logprobs.append(prompt_logprobs_dict)
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
            group_sample_logprobs.append(sample_logprobs_dict)
        result_sample_logprobs.append(group_sample_logprobs)
        sample_idx += len(seq_ids)

    return result_prompt_logprobs, result_sample_logprobs


def _build_sampler_output(
    sample_results: List[Tuple[List[int], List[int]]],
    input_metadata: InputMetadata,
    prompt_logprobs: List[Optional[PromptLogprobs]],
    sample_logprobs: List[SampleLogprobs],
    output_metadata: OutputMetadata,
) -> SamplerOutput:
    sampler_output = []
    for (seq_group, sample_result, group_prompt_logprobs,
         group_sample_logprobs) in zip(input_metadata.seq_groups,
                                       sample_results, prompt_logprobs,
                                       sample_logprobs):
        seq_ids, _ = seq_group
        next_token_ids, parent_ids = sample_result
        seq_outputs = []
        for parent_id, next_token_id, logprobs in zip(parent_ids,
                                                      next_token_ids,
                                                      group_sample_logprobs):
            seq_outputs.append(
                SequenceOutputs(seq_ids[parent_id], next_token_id, logprobs,
                                output_metadata.get(seq_ids[parent_id])))
        sampler_output.append(
            SequenceGroupOutputs(seq_outputs, group_prompt_logprobs))
    return sampler_output
