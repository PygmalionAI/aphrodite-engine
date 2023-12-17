from typing import Tuple, List

import torch
from torch import Tensor

from aphrodite.modeling.sampling_metadata import OutputMetadata, SamplingMetadata


def _fetch_args(
    metadata: SamplingMetadata
) -> Tuple[List[int], List[int], List[float], List[float], List[float]]:
    logit_indices: List[int] = []
    seqids: List[int] = []
    taus: List[float] = []
    etas: List[float] = []
    mus: List[float] = []

    index = 0
    for i, (seq_ids, params) in enumerate(metadata.seq_groups):
        # NOTE: If there are prompt logprobs here, SKIP THEM
        #       Miro persists data via seq_id, which these lack.
        #       In addition, mu is calculated using miro's chosen token,
        #       which prompt processing would ignore entirely.
        if (i < metadata.num_prompts and params.prompt_logprobs):
            index += metadata.prompt_lens[i] - 1

        if params.mirostat_mode == 2:
            logit_indices += [(index + i) for i in range(len(seq_ids))]
            seqids += seq_ids
            taus += [params.mirostat_tau] * len(seq_ids)
            etas += [params.mirostat_eta] * len(seq_ids)
            mus += [
                metadata.persistent_metadata.get(sid).get(
                    "miro_mu", params.mirostat_tau * 2) for sid in seq_ids
            ]
        index += len(seq_ids)
    return logit_indices, seqids, taus, etas, mus


def _store_args(seqids: List[int], mus: List[float],
                output_metadata: OutputMetadata) -> None:
    for sid, mu in zip(seqids, mus):
        output_metadata.add(sid, "miro_mu", mu)


def _apply_mirostat_v2(
        logits: Tensor,
        taus: List[float],  # AKA the targeted surprise
        etas: List[float],  # AKA the learning rate
        mus: List[
            float],  # AKA the accumulator that always tries to approach [tau]
) -> Tensor:
    ttaus = torch.tensor(taus, dtype=logits.dtype, device=logits.device)
    tetas = torch.tensor(etas, dtype=logits.dtype, device=logits.device)
    tmus = torch.tensor(mus, dtype=logits.dtype, device=logits.device)

    logit_surprise = torch.softmax(
        logits, dim=-1).log2_().neg_()  # Calculate surprise value per token
    # For compatibility with ooba/kobold, done in unit of bits(log base 2)
    # not nats(ln).
    # Ideally this would be a log_softmax, for numerical stability and
    # elegance purposes.
    # logit_surprise = torch.log_softmax(logits, dim=-1).neg_()

    miro_mask = logit_surprise > tmus.unsqueeze(
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
    next_token_ids = torch.multinomial(probs, num_samples=1, replacement=True)

    # Calculation new `mu` values
    # NOTE: If we can know the logit values of the PREVIOUS iteration,
    # it should be possible to update `mu` before applying mirostat each
    # iteration, thus letting us keep _sample as the last thing that happens.
    picked_surprises = torch.gather(logit_surprise,
                                    dim=-1,
                                    index=next_token_ids)
    eps = picked_surprises.squeeze() - ttaus
    tmus = tmus - tetas * eps
    mus[:] = tmus.tolist()

    logits.fill_(-float("inf"))
    # This value doesn't actually matter, so long as it's not -inf.
    # Vectors are now one-hot, after all.
    logits.scatter_(1, next_token_ids, 1.0)
    return logits


def is_applicable(sampling_metadata: SamplingMetadata) -> bool:
    return any((params.mirostat_mode == 2)
               for _, params in sampling_metadata.seq_groups)


def apply(logits: Tensor, sampling_metadata: SamplingMetadata,
          output_metadata: OutputMetadata) -> Tensor:
    logit_index, seqids, taus, etas, mus = _fetch_args(sampling_metadata)

    logits[logit_index] = _apply_mirostat_v2(
        logits[logit_index], taus, etas, mus)  # mus is an inout param, :vomit:
    _store_args(seqids, mus, output_metadata)
