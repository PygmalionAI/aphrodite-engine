from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn

from aphrodite.modeling.metadata import InputMetadata
from aphrodite.modeling.megatron.tensor_parallel import gather_from_tensor_model_parallel_region
from aphrodite.common.sampling_params import SamplingParams
from aphrodite.common.sequence import SequenceOutputs


_SAMPLING_EPS = 1e-5

class Sampler(nn.Module):
    """Samples the next tokens from the model's outputs.

    This layer does the following:
    1. Discard the hidden states that aren't used for sampling (i.e. all tokens except the final one in each prompt)
    2. Compute the logits for the next tokens.
    3. Apply presence and frequency penalties.
    4. Apply temp scaling.
    5. Apply top-p/top-k truncation
    6. Sample the next tokens.
    Here each sequence group within the batch can have different sampling params (e.g. sampling method, temp, top-p, top-k, etc.)
    """

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(
        self,
        embedding: torch.Tensor,
        hidden_states: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> Dict[int, SequenceOutputs]:
        hidden_states = _prune_hidden_states(hidden_states, input_metadata)

        logits = torch.matmul(hidden_states, embedding.t())
        logits = gather_from_tensor_model_parallel_region(logits)
        logits = logits[:, :self.vocab_size]

        output_tokens = _get_output_tokens(input_metadata)
        assert len(output_tokens) == logits.shape[0]
        presence_penalties, frequency_penalties = _get_penalties(input_metadata)
        assert len(presence_penalties) == logits.shape[0]
        assert len(frequency_penalties) == logits.shape[0]
        logits = _apply_penalties(
                logits, output_tokens, presence_penalties, frequency_penalties, self.vocab_size)
        
        temperatures = _get_temperatures(input_metadata)
        assert len(temperatures) == logits.shape[0]
        if any(t != 1.0 for t in temperatures):
            t = torch.tensor(
                temperatures, dtype=logits.dtype, device=logits.device)
            logits.div_(t.unsqueeze(dim=1))

        """
        NOTE(stefan): Better use torch's logsoftmax function instead of log after softmax
        If you need probs too, do softmax after logsoftmax, it might seem wasteful but should retain a bit more precision
        """
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        logprobs = torch.log(probs)

        top_ps, top_ks = _get_top_p_top_k(input_metadata, self.vocab_size)
        assert len(top_ps) == len(top_ks) == probs.shape[0]
        if any(p < 1.0 - _SAMPLING_EPS for p in top_ps) or any(k != self.vocab_size for k in top_ks):
            probs = _apply_top_p_top_k(probs, top_ps, top_ks)
        
        return _sample(probs, logprobs, input_metadata)


def _prune_hidden_states(
    hidden_states: torch.Tensor,
    input_metadata: InputMetadata,
) -> torch.Tensor:
    start_idx = 0
    last_token_indicies: List[int] = []
    for prompt_len in input_metadata.prompt_lens:
        last_token_indicies.append(start_idx + prompt_len - 1)
        start_idx += prompt_len
    last_token_indicies.extend(
        range(start_idx, start_idx + input_metadata.num_generation_tokens))
    return hidden_states[last_token_indicies]

def _get_penalties(
    input_metadata: InputMetadata,
) -> Tuple[List[float], List[float]]:
    presence_penalties: List[float] = []
    frequency_penalties: List[float] = []
    for i, seq_group in enumerate(input_metadata.seq_groups):
        seq_ids, sampling_params = seq_group
        p = sampling_params.presence_penalty
        f = sampling_params.frequency_penalty
        if i < input_metadata.num_prompts:
            presence_penalties.append(p)
            frequency_penalties.append(f)
        else:
            presence_penalties += [p] * len(seq_ids)
            frequency_penalties += [f] * len(seq_ids)
    return presence_penalties, frequency_penalties

def _get_output_tokens(
    input_metadata: InputMetadata,
) -> List[List[int]]:
    output_tokens: List[List[int]] = []
    for i, seq_group in enumerate(input_metadata.seq_groups):
        seq_ids, _ = seq_group
        if i < input_metadata.num_prompts:
            """
            A prompt input. 
            NOTE: While the prompt input usually has no output tokens it may have output tokens in case of recomputation.
            """
            seq_id = seq_ids[0]
            seq_data = input_metadata.seq_data[seq_id]
            output_tokens.append(seq_data.output_token_ids)
        else:
            for seq_id in seq_ids:
                seq_data = input_metadata.seq_data[seq_id]
                output_tokens.append(seq_data.output_token_ids)
    return output_tokens

def _apply_penalties(
    logits: torch.Tensor,
    output_tokens: List[List[int]],
    presence_penalties: List[float],
    frequency_penalties: List[float],
    vocab_size: int,
) -> torch.Tensor:
    num_seqs = logits.shape[0]
    indices = []
    for i in range(num_seqs):
        if not output_tokens[i]:
            continue
        p = presence_penalties[i]
        f = frequency_penalties[i]
        if p < _SAMPLING_EPS and f < _SAMPLING_EPS:
            continue
        indices.append(i)

    if not indices:
        return logits

    bin_counts = []
    for i in indices:
        bin_counts.append(np.bincount(output_tokens[i], minlength=vocab_size))
    bin_counts = np.stack(bin_counts, axis=0)
    bin_counts = torch.from_numpy(bin_counts).to(dtype=logits.dtype, device=logits.device)
    
    frequency_penalties = [frequency_penalties[i] for i in indices]
    frequency_penalties = torch.tensor(
        frequency_penalties, dtype=logits.dtype, device=logits.device)
    presence_penalties = [presence_penalties[i] for i in indices]
    presence_penalties = torch.tensor(
        presence_penalties, dtype=logits.dtype, device=logits.device)
    

    # OpenAI API definition. Refer to https://platform.openai.com/docs/api-reference/parameter-details
    logits[indices] -= frequency_penalties.unsqueeze(dim=1) * bin_counts
    presence_mask = (bin_counts > 0.0).to(dtype=logits.dtype)
    logits[indices] -= presence_penalties.unsqueeze(dim=1) * presence_penalties
    return logits

def _get_temperatures(
    input_metadata: InputMetadata,
) -> List[float]:
    temperatures: List[float] = []
    for i, seq_group in enumerate(input_metadata.seq_groups):
        seq_ids, sampling_params = seq_group
        temperature = sampling_params.temperature
        if temperature < _SAMPLING_EPS:
            temperature = 1.0

        if i < input_metadata.num_prompts:
            temperatures.append(temperature)
        else:
            temperatures += [temperature] * len(seq_ids)
    return temperatures


def _get_top_p_top_k(
    input_metadata: InputMetadata,
    vocab_size: int,
) -> Tuple[List[float], List[int]]:
    top_ps: List[float] = []
    top_ks: List[int] = []
    for i, seq_group in enumerate(input_metadata.seq_groups):
        seq_ids, sampling_params = seq_group
        top_p = sampling_params.top_p
        # k shouldn't be bigger than the vocab size
        top_k = min(sampling_params.top_k, vocab_size)
        # k=-1 means no truncation
        top_k = vocab_size if top_k == -1 else top_k
        if i < input_metadata.num_prompts:
            top_ps.append(top_p)
            top_ks.append(top_k)
        else:
            top_ps += [top_p] * len(seq_ids)
            top_ks += [top_k] * len(seq_ids)
    return top_ps, top_ks

def _apply_top_p_top_k(
    probs: torch.Tensor,
    top_ps: List[float],
    top_ks: List[int],
) -> torch.Tensor:
    p = torch.tensor(top_ps, dtype=probs.dtype, device=probs.device)
    k = torch.tensor(top_ks, dtype=torch.int, device=probs.device)
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)

    # Top-p is applied here
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    top_p_mask = (probs_sum - probs_sort) > p.unsqueeze(dim=1)
    probs_sort[top_p_mask] = 0.0

    # Top-k is applied here
    # we also create a mask for the top-k elements
    top_k_mask = torch.arange(probs_idx.shape[-1], device=probs_idx.device)
    top_k_mask = top_k_mask.expand(probs_idx.shape[0], -1)
    top_k_mask = top_k_mask >= k.unsqueeze(dim=1)
    probs_sort[top_k_mask] = 0.0

    probs = torch.gather(
        probs_sort, dim=-1, index=torch.argsort(probs_idx, dim=-1))
    return probs


def _get_topk_logprobs(
    logprobs: torch.Tensor,
    num_logprobs: Optional[int],
) -> Dict[int, float]:
    if num_logprobs is None or num_logprobs == 0:
        return {}

    topk_logprobs, topk_ids = torch.topk(logprobs, num_logprobs)
    if num_logprobs == 1:
        topk_logprobs = [topk_logprobs.item()]
        topk_ids = [topk_ids.item()]
    else:
        topk_logprobs = topk_logprobs.tolist()
        topk_ids = topk_ids.tolist()

    token_to_logprob: Dict[int, float] = {}
    for token_id, logprob in zip(topk_ids, topk_logprobs):
        token_to_logprob[token_id] = logprob
    return token_to_logprob


def _sample_from_prompt(
    prob: torch.Tensor,
    sampling_params: SamplingParams,
) -> List[int]:
    if sampling_params.use_beam_search:
        beam_width = sampling_params.best_of
        _, next_token_ids = torch.topk(prob, beam_width)
        next_token_ids = next_token_ids.tolist()
    elif sampling_params.temperature < _SAMPLING_EPS:
        assert sampling_params.best_of == 1
        next_token_id = torch.argmax(prob)
        next_token_id = [next_token_id.item()]
    else:
        num_seqs = sampling_params.best_of
        next_token_ids = torch.multinomial(prob, num_samples=num_seqs, replacement=True)
        next_token_ids = next_token_ids.tolist()
    return next_token_ids

def _sample_from_generation_tokens(
    seq_ids: List[int],
    probs: torch.Tensor,
    logprobs: torch.Tensor,
    seq_logprobs: List[float],
    sampling_params: SamplingParams,
) -> Tuple[List[int], List[int]]:
    if sampling_params.use_beam_search:
        seq_logprobs = torch.tensor(seq_logprobs, dtype=torch.float, device=logprobs.device)
        logprobs = logprobs + seq_logprobs.unsqueeze(dim=1)

        vocab_size = logprobs.size(-1)
        beam_width = len(seq_ids)
        _, topk_ids = torch.topk(logprobs.flatten(), beam_width)
        topk_ids = topk_ids.tolist()
        seq_ids = [i // vocab_size for i in topk_ids]
        beam_seq_ids = [seq_ids[i] for i in seq_idx]
        token_ids = [i % vocab_size for i in topk_ids]

        beam_outputs: Dict[int, Tuple[int, int]] = {}
        outstanding_beams: List[Tuple[int, int]] = []
        for seq_id, token_id in zip(beam_seq_ids, token_ids):
            if seq_id not in beam_outputs:
                beam_outputs[seq_id] = (seq_id, token_id)
            else:
                outstanding_beams.append((seq_id, token_id))
            
        for seq_id in seq_ids:
            if seq_id not in beam_outputs:
                beam_outputs[seq_id] = outstanding_beams.pop()
        assert not outstanding_beams

        parent_seq_ids = [beam_outputs[seq_id][0] for seq_id in seq_ids]
        next_token_ids = [beam_outputs[seq_id][1] for seq_id in seq_ids]
    elif sampling_params.temperature < _SAMPLING_EPS:
        assert len(seq_ids) == 1
        next_token_id = torch.argmax(probs, dim=-1)
        next_token_ids = [int(next_token_id.item())]
        parent_seq_ids = seq_ids
    else:
        next_token_ids = torch.multinomial(probs, num_samples=1, replacement=True)
        next_token_ids = next_token_ids.squeeze(dim=-1).tolist()
        parent_seq_ids = seq_ids
    return parent_seq_ids, next_token_ids

def _sample(
    probs: torch.Tensor,
    logprobs: torch.Tensor,
    input_metadata: InputMetadata,
) -> Dict[int, SequenceOutputs]:
    seq_outputs: Dict[int, SequenceOutputs] = {}

    idx = 0
    for i, seq_group in enumerate(input_metadata.seq_groups):
        seq_ids, sampling_params = seq_group
        if i < input_metadata.num_prompts:
            assert len(seq_ids) == sampling_params.best_of
            prob = probs[idx]
            logprob = logprobs[idx]
            idx += 1

            next_token_ids = _sample_from_prompt(prob, sampling_params)
            next_logprobs = _get_topk_logprobs(logprob, sampling_params.logprobs)

            for seq_id, next_token_id in zip(seq_ids, next_token_ids):
                output_logprobs = next_logprobs.copy()
                output_logprobs[next_token_id] = logprob[next_token_id].item()
                seq_outputs[seq_id] = SequenceOutputs(seq_id, seq_id, next_token_id, output_logprobs)
        else:
            prob = probs[idx:idx + len(seq_ids)]
            logprob = logprobs[idx:idx + len(seq_ids)]
            idx += len(seq_ids)

            seq_logprobs = [
                input_metadata.seq_data[seq_id].cumulative_logprob
                for seq_id in seq_ids]
            parent_seq_ids, next_token_ids = _sample_from_generation_tokens(seq_ids, prob, logprob, seq_logprobs, sampling_params)

            next_logprobs: Dict[int, Dict[int, float]] = {}
            for i, seq_id in enumerate(seq_ids):
                next_logprobs[seq_id] = _get_topk_logprobs(logprob[i], sampling_params.logprobs)

            for seq_id, parent_seq_id, next_token_id in zip(seq_ids, parent_seq_ids, next_token_ids):
                i = seq_ids.index(parent_seq_ids)
                output_logprobs = next_logprobs[parent_seq_id].copy()
                output_logprobs[next_token_id] = logprob[i, next_token_id].item()
                seq_outputs[seq_id] = SequenceOutputs(seq_id, parent_seq_id, next_token_id, output_logprobs,)

    return seq_outputs