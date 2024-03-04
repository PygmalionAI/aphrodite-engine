from abc import ABC, abstractmethod
import torch
from typing import Dict, List


class LogitsProcessor(ABC):

    @abstractmethod
    def __call__(self, logits: torch.Tensor,
                 output_tokens: List[List[int]]) -> None:
        """Logits are edited in-place"""
        pass


class BiasLogitsProcessor(LogitsProcessor):
    """This is to enable logit_bias in the OpenAI server,
    an additive bias on the original logit values.
    Args:
      biases: Dict of bias values. Each key corresponds to the the token id.
    """

    def __init__(self, biases: Dict[int, float]):
        super().__init__()
        self.biases = biases

        if not biases:
            return

        self.keys = torch.tensor(list(self.biases.keys()), dtype=torch.long)
        self.values = torch.tensor(list(self.biases.values()),
                                   dtype=torch.float)

    def __call__(self, logits: torch.Tensor,
                 output_tokens: List[List[int]]) -> None:
        if not self.biases:
            return

        values = self.values.to(logits.device)
        keys = self.keys.to(logits.device)
        logits[0, keys] += values


class BanEOSUntil(LogitsProcessor):
    """Bans the EOS token until a certain condition is met.
    In this case, 'number of output tokens'.

    With this condition, both 'min_tokens' and 'ignore_eos'
    parameters can be handled gracefully."""

    def __init__(self, min_tokens: int, eos_token_id: int):
        super().__init__()
        self._min_tokens = min_tokens
        self._eos_token_id = eos_token_id

    def __call__(self, logits: torch.Tensor,
                 output_tokens: List[List[int]]) -> None:
        for i in range(len(output_tokens)):
            if len(output_tokens[i]) < self._min_tokens:
                logits[i][self._eos_token_id] = -float("inf")
