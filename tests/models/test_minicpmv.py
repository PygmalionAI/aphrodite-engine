from typing import List, Optional, Tuple, Type

import pytest
import torch
import torch.types
from transformers import BatchEncoding

from aphrodite.common.sequence import SampleLogprobs
from aphrodite.multimodal.utils import rescale_image_size

from ..conftest import IMAGE_ASSETS, AphroditeRunner, HfRunner, _ImageAssets
from .utils import check_logprobs_close

pytestmark = pytest.mark.vlm

# The image token is placed before "user" on purpose so that the test can pass
HF_IMAGE_PROMPTS = IMAGE_ASSETS.prompts({
    "stop_sign":
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" \
        "(<image>./</image>)\nWhat's the content of the image?<|eot_id|>" \
        "<|start_header_id|>assistant<|end_header_id|>\n\n",  # noqa: E501
    "cherry_blossom":
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" \
        "(<image>./</image>)\nWhat is the season?<|eot_id|>" \
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
})

models = ["openbmb/MiniCPM-Llama3-V-2_5"]


def _wrap_inputs(hf_inputs: BatchEncoding) -> BatchEncoding:
    return BatchEncoding({"model_inputs": hf_inputs})


def trunc_hf_output(hf_output: Tuple[List[int], str,
                                     Optional[SampleLogprobs]]):
    output_ids, output_str, out_logprobs = hf_output
    if output_str.endswith("<|eot_id|>"):
        output_str = output_str.split("<|eot_id|>")[0]
    return output_ids, output_str, out_logprobs


target_dtype = "half"


def run_test(
    hf_runner: Type[HfRunner],
    aphrodite_runner: Type[AphroditeRunner],
    image_assets: _ImageAssets,
    model: str,
    *,
    size_factors: List[float],
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    tensor_parallel_size: int,
    distributed_executor_backend: Optional[str] = None,
):
    """Inference result should be the same between hf and aphrodite.

    All the image fixtures for the test is under tests/images.
    For huggingface runner, we provide the PIL images as input.
    For aphrodite runner, we provide MultiModalDataDict objects 
    and corresponding MultiModalConfig as input.
    Note, the text input is also adjusted to abide by aphrodite contract.
    The text output is sanitized to be able to compare with hf.
    """
    images = [asset.pil_image for asset in image_assets]

    inputs_per_image = [(
        [prompt for _ in size_factors],
        [rescale_image_size(image, factor) for factor in size_factors],
    ) for image, prompt in zip(images, HF_IMAGE_PROMPTS)]

    # NOTE: take care of the order. run Aphrodite first, and then run HF.
    # Aphrodite needs a fresh new process without cuda initialization.
    # if we run HF first, the cuda initialization will be done and it
    # will hurt multiprocessing backend with fork method (the default method).

    # max_model_len should be greater than image_feature_size
    with aphrodite_runner(model,
                     max_model_len=4096,
                     max_num_seqs=1,
                     dtype=dtype,
                     tensor_parallel_size=tensor_parallel_size,
                     distributed_executor_backend=distributed_executor_backend,
                     enforce_eager=True) as aphrodite_model:
        tokenizer = aphrodite_model.model.get_tokenizer()
        stop_token_ids = [tokenizer.eos_id, tokenizer.eot_id]
        aphrodite_outputs_per_image = [
            aphrodite_model.generate_greedy_logprobs(prompts,
                                                max_tokens,
                                                num_logprobs=num_logprobs,
                                                images=images,
                                                stop_token_ids=stop_token_ids)
            for prompts, images in inputs_per_image
        ]

    hf_model = hf_runner(model, dtype=dtype, postprocess_inputs=_wrap_inputs)
    with hf_model, torch.no_grad():
        hf_outputs_per_image = [
            hf_model.generate_greedy_logprobs_limit(prompts,
                                                    max_tokens,
                                                    num_logprobs=num_logprobs,
                                                    images=images,
                                                    tokenizer=tokenizer)
            for prompts, images in inputs_per_image
        ]

    for hf_outputs, aphrodite_outputs in zip(hf_outputs_per_image,
                                        aphrodite_outputs_per_image):
        check_logprobs_close(
            outputs_0_lst=[
                trunc_hf_output(hf_output) for hf_output in hf_outputs
            ],
            outputs_1_lst=aphrodite_outputs,
            name_0="hf",
            name_1="aphrodite",
        )


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize(
    "size_factors",
    [
        # No image
        [],
        # Single-scale
        [1.0],
        # Single-scale, batched
        [1.0, 1.0, 1.0],
        # Multi-scale
        [0.25, 0.5, 1.0],
    ],
)
@pytest.mark.parametrize("dtype", [target_dtype])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("num_logprobs", [5])
def test_models(hf_runner, aphrodite_runner, image_assets, model, size_factors,
                dtype: str, max_tokens: int, num_logprobs: int) -> None:
    run_test(
        hf_runner,
        aphrodite_runner,
        image_assets,
        model,
        size_factors=size_factors,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        tensor_parallel_size=1,
    )


HF_MULTIIMAGE_IMAGE_PROMPT = \
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" \
    "(<image>./</image>)\n(<image>./</image>)\n" \
    "Describe these images.<|eot_id|>" \
    "<|start_header_id|>assistant<|end_header_id|>\n\n"


def run_multi_image_test(
    hf_runner: Type[HfRunner],
    aphrodite_runner: Type[AphroditeRunner],
    image_assets: _ImageAssets,
    model: str,
    *,
    size_factors: List[float],
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    tensor_parallel_size: int,
    distributed_executor_backend: Optional[str] = None,
):
    """Inference result should be the same between hf and aphrodite.

    All the image fixtures for the test is under tests/images.
    For huggingface runner, we provide the PIL images as input.
    For aphrodite runner, we provide MultiModalDataDict objects 
    and corresponding MultiModalConfig as input.
    Note, the text input is also adjusted to abide by aphrodite contract.
    The text output is sanitized to be able to compare with hf.
    """
    images = [asset.pil_image for asset in image_assets]

    inputs_per_case = [
        ([HF_MULTIIMAGE_IMAGE_PROMPT for _ in size_factors],
         [[rescale_image_size(image, factor) for image in images]
          for factor in size_factors])
    ]

    # NOTE: take care of the order. run Aphrodite first, and then run HF.
    # Aphrodite needs a fresh new process without cuda initialization.
    # if we run HF first, the cuda initialization will be done and it
    # will hurt multiprocessing backend with fork method (the default method).

    # max_model_len should be greater than image_feature_size
    with aphrodite_runner(model,
                     max_model_len=4096,
                     max_num_seqs=1,
                     limit_mm_per_prompt={"image": len(images)},
                     dtype=dtype,
                     tensor_parallel_size=tensor_parallel_size,
                     distributed_executor_backend=distributed_executor_backend,
                     enforce_eager=True) as aphrodite_model:
        tokenizer = aphrodite_model.model.get_tokenizer()
        stop_token_ids = [tokenizer.eos_id, tokenizer.eot_id]
        aphrodite_outputs_per_case = [
            aphrodite_model.generate_greedy_logprobs(prompts,
                                                max_tokens,
                                                num_logprobs=num_logprobs,
                                                images=images,
                                                stop_token_ids=stop_token_ids)
            for prompts, images in inputs_per_case
        ]

    hf_model = hf_runner(model, dtype=dtype, postprocess_inputs=_wrap_inputs)
    with hf_model, torch.no_grad():
        hf_outputs_per_case = [
            hf_model.generate_greedy_logprobs_limit(prompts,
                                                    max_tokens,
                                                    num_logprobs=num_logprobs,
                                                    images=images,
                                                    tokenizer=tokenizer)
            for prompts, images in inputs_per_case
        ]

    for hf_outputs, aphrodite_outputs in zip(hf_outputs_per_case,
                                        aphrodite_outputs_per_case):
        check_logprobs_close(
            outputs_0_lst=[
                trunc_hf_output(hf_output) for hf_output in hf_outputs
            ],
            outputs_1_lst=aphrodite_outputs,
            name_0="hf",
            name_1="aphrodite",
        )


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize(
    "size_factors",
    [
        # No image
        [],
        # Single-scale
        [1.0],
        # Single-scale, batched
        [1.0, 1.0, 1.0],
        # Multi-scale
        [0.25, 0.5, 1.0],
    ],
)
@pytest.mark.parametrize("dtype", [target_dtype])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("num_logprobs", [5])
def test_multi_images_models(hf_runner, aphrodite_runner, image_assets, model,
                             size_factors, dtype: str, max_tokens: int,
                             num_logprobs: int) -> None:
    run_multi_image_test(
        hf_runner,
        aphrodite_runner,
        image_assets,
        model,
        size_factors=size_factors,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        tensor_parallel_size=1,
    )
