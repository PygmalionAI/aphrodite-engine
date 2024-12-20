from typing import List, Optional, Tuple, Type

import librosa
import numpy as np
import pytest
from transformers import AutoModel, AutoTokenizer, BatchEncoding

from aphrodite.assets.audio import AudioAsset
from aphrodite.common.sequence import SampleLogprobs
from aphrodite.common.utils import STR_DTYPE_TO_TORCH_DTYPE

from ..conftest import HfRunner, AphroditeRunner
from .utils import check_logprobs_close

pytestmark = pytest.mark.vlm

MODEL_NAME = "fixie-ai/ultravox-v0_3"

AudioTuple = Tuple[np.ndarray, int]


@pytest.fixture(scope="session")
def audio_and_sample_rate():
    return AudioAsset("mary_had_lamb").audio_and_sample_rate


@pytest.fixture
def prompts_and_audios(audio_and_sample_rate):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    aphrodite_placeholder = "<|reserved_special_token_0|>"
    hf_placeholder = "<|audio|>"

    question = "What's in the audio?"
    aphrodite_prompt = tokenizer.apply_chat_template(
        [{
            'role': 'user',
            'content': f"{aphrodite_placeholder}\n{question}"
        }],
        tokenize=False,
        add_generation_prompt=True)
    hf_prompt = tokenizer.apply_chat_template(
        [{
            'role': 'user',
            'content': f"{hf_placeholder}\n{question}"
        }],
        tokenize=False,
        add_generation_prompt=True)

    return [(aphrodite_prompt, hf_prompt, audio_and_sample_rate)]


def aphrodite_to_hf_output(aphrodite_output: Tuple[List[int], str,
                                         Optional[SampleLogprobs]],
                      model: str):
    """Sanitize aphrodite output to be comparable with hf output."""
    output_ids, output_str, out_logprobs = aphrodite_output

    tokenizer = AutoTokenizer.from_pretrained(model)
    eos_token_id = tokenizer.eos_token_id

    hf_output_ids = output_ids[:]
    hf_output_str = output_str
    if hf_output_ids[-1] == eos_token_id:
        hf_output_str = hf_output_str + tokenizer.decode(eos_token_id)

    return hf_output_ids, hf_output_str, out_logprobs


def run_test(
    hf_runner: Type[HfRunner],
    aphrodite_runner: Type[AphroditeRunner],
    prompts_and_audios: List[Tuple[str, str, AudioTuple]],
    model: str,
    *,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    tensor_parallel_size: int,
    distributed_executor_backend: Optional[str] = None,
):
    """Inference result should be the same between hf and aphrodite."""
    torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[dtype]

    # NOTE: take care of the order. run Aphrodite first, and then run HF.
    # Aphrodite needs a fresh new process without cuda initialization.
    # if we run HF first, the cuda initialization will be done and it
    # will hurt multiprocessing backend with fork method (the default method).

    with aphrodite_runner(model,
                     dtype=dtype,
                     tensor_parallel_size=tensor_parallel_size,
                     distributed_executor_backend=distributed_executor_backend,
                     enforce_eager=True) as aphrodite_model:
        aphrodite_outputs_per_audio = [
            aphrodite_model.generate_greedy_logprobs([aphrodite_prompt],
                                                max_tokens,
                                                num_logprobs=num_logprobs,
                                                audios=[audio])
            for aphrodite_prompt, _, audio in prompts_and_audios
        ]

    def process(hf_inputs: BatchEncoding):
        hf_inputs["audio_values"] = hf_inputs["audio_values"] \
            .to(torch_dtype)  # type: ignore
        return hf_inputs

    with hf_runner(model,
                   dtype=dtype,
                   postprocess_inputs=process,
                   auto_cls=AutoModel) as hf_model:

        hf_outputs_per_audio = [
            hf_model.generate_greedy_logprobs_limit(
                [hf_prompt],
                max_tokens,
                num_logprobs=num_logprobs,
                audios=[(librosa.resample(audio[0],
                                          orig_sr=audio[1],
                                          target_sr=16000), 16000)])
            for _, hf_prompt, audio in prompts_and_audios
        ]

    for hf_outputs, aphrodite_outputs in zip(hf_outputs_per_audio,
                                        aphrodite_outputs_per_audio):
        check_logprobs_close(
            outputs_0_lst=hf_outputs,
            outputs_1_lst=[
                aphrodite_to_hf_output(aphrodite_output, model)
                for aphrodite_output in aphrodite_outputs
            ],
            name_0="hf",
            name_1="aphrodite",
        )


@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("num_logprobs", [5])
def test_models(hf_runner, aphrodite_runner, prompts_and_audios, dtype: str,
                max_tokens: int, num_logprobs: int) -> None:
    run_test(
        hf_runner,
        aphrodite_runner,
        prompts_and_audios,
        MODEL_NAME,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        tensor_parallel_size=1,
    )
