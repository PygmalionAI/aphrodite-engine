from abc import ABC, abstractmethod
import torch
from typing import Dict


class LogitsProcessor(ABC):

    @abstractmethod
    def __call__(self, logits: torch.Tensor,
                 output_tokens: list[list[int]]) -> None:
        """Logits are edited in-place"""
        pass


class BiasLogitsProcessor(LogitsProcessor):
    """This is to enable logit_bias in the OpenAI server.
    biases is a dict where each value is -100 to 100
      according to the OpenAI API docs.
    Args:
      biases: Dict of values from -100 to 100 to scale the
        probability of a token being generated.
        Each key of the dict corresponds to the the token id.
    """

    def __init__(self, biases: Dict[int, float]):
        self.biases = biases

        if not biases:
            return

        self.keys = torch.tensor(list(self.biases.keys()), dtype=torch.long)
        self.values = torch.tensor(list(self.biases.values()),
                                   dtype=torch.long)

    def __call__(self, logits):
        if not self.biases:
            return

        values = self.values.to(logits.device)
        keys = self.keys.to(logits.device)

        update_factors = torch.where(values >= 0, 1 + (values / 100),
                                     1 / (1 - (values / 100)))
        logits[0, keys] *= update_factors


class BanEOSUntil(LogitsProcessor):
    """Bans the EOS token until a certain condition is met.
    In this case, 'number of output tokens'.

    With this condition, both 'min_tokens' and 'ignore_eos'
    parameters can be handled gracefully."""

    def __init__(self, min_tokens: int, eos_token_id: int):
        self._min_tokens = min_tokens
        self._eos_token_id = eos_token_id

    def __call__(self, logits, output_tokens):
        for i in range(len(output_tokens)):
            if len(output_tokens[i]) < self._min_tokens:
                logits[i][self._eos_token_id] = -float("inf")
