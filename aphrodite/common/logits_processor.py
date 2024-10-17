from abc import ABC, abstractmethod
from typing import Dict, List

import torch


class LogitsProcessor(ABC):

    @abstractmethod
    def __call__(self, output_tokens: List[int],
                 logits: torch.Tensor) -> torch.Tensor:
        """Logits are edited in-place"""
        pass

    @abstractmethod
    def batched(self, logits: torch.Tensor,
                output_tokens: List[List[int]]) -> None:
        """Logits are edited in-place"""
        pass


class BiasLogitsProcessor(LogitsProcessor):
    """Apply an additive bias to specific token logits.
    Args:
      biases: Dict of bias values. Each key corresponds to the the token id.
    """

    def __init__(self, biases: Dict[int, float]):
        assert biases
        self.biases = biases
        self.keys = torch.tensor(list(self.biases.keys()), dtype=torch.long)
        self.values = torch.tensor(list(self.biases.values()),
                                   dtype=torch.float)

    def __call__(self, output_tokens: List[int],
                 logits: torch.Tensor) -> torch.Tensor:
        values = self.values.to(logits.device)
        keys = self.keys.to(logits.device)
        logits[keys] += values
        return logits

    def batched(self, logits: torch.Tensor,
                output_tokens: List[List[int]]) -> None:
        values = self.values.to(logits.device)
        keys = self.keys.to(logits.device)
        logits[:, keys] += values


class BanEOSUntil(LogitsProcessor):
    """Bans the EOS token until a certain condition is met.
    In this case, 'number of output tokens'.

    With this condition, both 'min_tokens' and 'ignore_eos'
    parameters can be handled gracefully."""

    def __init__(self, min_tokens: int, eos_token_id: int):
        self._min_tokens = min_tokens
        self._eos_token_id = eos_token_id

    def __call__(self, output_tokens: List[int],
                 logits: torch.Tensor) -> torch.Tensor:
        if len(output_tokens) < self._min_tokens:
            logits[self._eos_token_id] = -float("inf")
        return logits

    def batched(self, logits: torch.Tensor,
                output_tokens: List[List[int]]) -> None:
        terminate_mask = torch.tensor(
            [len(toks) < self._min_tokens for toks in output_tokens],
            device=logits.device)
        logits[terminate_mask, self._eos_token_id] = -float("inf")
