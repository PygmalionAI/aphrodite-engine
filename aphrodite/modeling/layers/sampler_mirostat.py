from typing import Tuple, List

import torch
from torch import Tensor

from aphrodite.modeling.sampling_metadata import OutputMetadata, SamplingTensors
from aphrodite.modeling.layers.sampler import _multinomial


def _store_args(seqids: List[int], mus: List[float],
                output_metadata: OutputMetadata) -> None:
    for sid, mu in zip(seqids, mus.tolist()): # tolist might be premature optimization
        output_metadata.add(sid, "miro_mu", mu)


def _apply_mirostat_v2(
        logits: Tensor,
        taus: Tensor,  # AKA the targeted surprise
        etas: Tensor,  # AKA the learning rate
        mus: Tensor,  # AKA the accumulator that always tries to approach [tau]
) -> Tensor:

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


def apply(logits: Tensor, sampling_tensors: SamplingTensors,
          output_metadata: OutputMetadata) -> Tensor:
    idx = sampling_tensors.miro_indices
    seqids = sampling_tensors.miro_seqids
    taus = sampling_tensors.miro_taus
    etas = sampling_tensors.miro_etas
    mus = sampling_tensors.miro_mus

    logits[idx] = _apply_mirostat_v2(
        logits[idx], taus, etas, mus)  # mus is an inout param, :vomit:
    _store_args(seqids, mus, output_metadata)
    return logits
