# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Offline contracts for the harness-to-Sonar integration; no model is loaded."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from lm_eval.api.instance import Instance
from lm_eval.models.openai_completions import LocalCompletionsAPI

from tests.quantization import lm_eval_utils


def test_thinking_generation_applies_task_stops_only_after_reasoning(monkeypatch):
    model = object.__new__(lm_eval_utils._SonarCompletionsAPI)
    model.think_end_token = "</think>"
    model._eos_string = "<eos>"
    requests = [
        Instance(
            request_type="generate_until", doc={}, arguments=("first", {"until": ["STOP"], "max_gen_toks": 10}), idx=0
        ),
        Instance(request_type="generate_until", doc={}, arguments=("second", {"until": ["END"]}), idx=1),
    ]
    outputs = ["reasoning STOP inside</think>AnswerSTOPdiscard", "reasoning</think>SecondENDdiscard"]

    def generate_with_stops(self, generation_requests, **kwargs):
        results = []
        for request, output in zip(generation_requests, outputs, strict=True):
            for stop in request.args[1].get("until", []):
                output = output.split(stop)[0]
            results.append(output)
        return results

    monkeypatch.setattr(LocalCompletionsAPI, "generate_until", generate_with_stops)

    assert model.generate_until(requests, disable_tqdm=True) == ["Answer", "Second"]


@pytest.mark.parametrize(
    ("second_top", "is_greedy"), [({"chosen": -0.2}, True), ({"chosen": -0.2, "other": -0.1}, False)]
)
def test_prompt_loglikelihood_scores_only_continuation(second_top, is_greedy):
    output = {
        "choices": [
            {
                "index": 0,
                "logprobs": {
                    "token_logprobs": [None, -0.1, -0.2, -9.0],
                    "top_logprobs": [None, {"first": -0.1}, second_top, {"generated": -9.0}],
                },
            }
        ]
    }

    scores = lm_eval_utils._SonarCompletionsAPI.parse_logprobs(output, ctxlens=[1])

    assert scores[0][0] == pytest.approx(-0.3)
    assert scores[0][1] is is_greedy


def test_rolling_loglikelihood_scores_each_token_once():
    model = object.__new__(lm_eval_utils._SonarCompletionsAPI)
    model.max_length = 5
    model.custom_prefix_token_id = 99
    model.tokenizer = SimpleNamespace(bos_token_id=99)
    model.tok_encode = lambda text: list(range(1, len(text) + 1))
    model.cache_hook = MagicMock()
    scored_tokens = []
    input_lengths = []
    contexts = []

    def score_windows(windows, **kwargs):
        scores = []
        for _, context, continuation in windows:
            contexts.append(context)
            input_lengths.append(len(context) + len(continuation))
            scored_tokens.extend(continuation)
            scores.append((-float(sum(continuation)), True))
        return scores

    model._loglikelihood_tokens = score_windows
    requests = [
        Instance(request_type="loglikelihood_rolling", doc={}, arguments=("abcdefg",), idx=0),
        Instance(request_type="loglikelihood_rolling", doc={}, arguments=("",), idx=1),
    ]

    assert model.loglikelihood_rolling(requests, disable_tqdm=True) == [-28.0, 0]
    assert scored_tokens == list(range(1, 8))
    assert all(length <= model.max_length for length in input_lengths)
    assert contexts[0] == [99]


def test_tokenization_avoids_duplicate_bos_in_mixed_requests():
    model = object.__new__(lm_eval_utils._SonarCompletionsAPI)
    model.add_bos_token = True
    model.tokenizer = MagicMock()
    model.tokenizer.bos_token_id = 1
    model.tokenizer.eos_token_id = 0
    model.tokenizer.decode.return_value = "<bos>"

    def encode(text, **kwargs):
        prefix = [1] if kwargs.get("add_special_tokens") else []
        literal_bos = [1] if text.startswith("<bos>") else []
        return SimpleNamespace(input_ids=prefix + literal_bos + [2, 3])

    model.tokenizer.side_effect = encode

    assert model.tok_encode(["<bos>first", "second"]) == [[1, 2, 3], [1, 2, 3]]
    assert model.tok_encode("second", add_special_tokens=False) == [2, 3]
    assert model.tok_encode(["<bos>first", "second"], left_truncate_len=2) == [[2, 3], [2, 3]]
    assert model.tok_encode("") == []
    assert model.tok_encode([]) == []
