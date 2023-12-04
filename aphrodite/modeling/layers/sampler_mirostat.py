import torch
from torch import Tensor

from aphrodite.modeling.metadata import OutputMetadata, InputMetadata


def _fetch_args(
    input_metadata: InputMetadata
) -> tuple[list[int], list[int], list[float], list[float], list[float]]:
    logit_indices: list[int] = []
    seqids: list[int] = []
    taus: list[float] = []
    etas: list[float] = []
    mus: list[float] = []

    index = 0
    for seq_ids, params in input_metadata.seq_groups:
        if params.mirostat_mode == 2:
            logit_indices += [(index + i) for i in range(len(seq_ids))]
            seqids += seq_ids
            taus += [params.mirostat_tau] * len(seq_ids)
            etas += [params.mirostat_eta] * len(seq_ids)
            mus += [
                input_metadata.persistent_data.get(sid).get(
                    "miro_mu", params.mirostat_tau * 2) for sid in seq_ids
            ]
        index += len(seq_ids)
    return logit_indices, seqids, taus, etas, mus


def _store_args(seqids: list[int], mus: list[float],
                output_metadata: OutputMetadata) -> None:
    for sid, mu in zip(seqids, mus):
        output_metadata.add(sid, "miro_mu", mu)


def _apply_mirostat_v2(
        logits: Tensor,
        taus: list[float],  # AKA the targeted surprise
        etas: list[float],  # AKA the learning rate
        mus: list[
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


def is_applicable(input_metadata: InputMetadata) -> bool:
    return any(
        (params.mirostat_mode == 2) for _, params in input_metadata.seq_groups)


def apply(logits: Tensor, input_metadata: InputMetadata,
          output_metadata: OutputMetadata) -> Tensor:
    logit_index, seqids, taus, etas, mus = _fetch_args(input_metadata)
    # print("logidx", logit_index)
    # print("seqids", seqids)
    # print("  taus", taus)

    logits[logit_index] = _apply_mirostat_v2(
        logits[logit_index], taus, etas, mus)  # mus is an inout param, :vomit:
    _store_args(seqids, mus, output_metadata)
