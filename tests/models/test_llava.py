from typing import List, Optional, Tuple, Type

import pytest
from transformers import (AutoConfig, AutoModelForVision2Seq, AutoTokenizer,
                          BatchEncoding)

from aphrodite.common.sequence import SampleLogprobs
from aphrodite.common.utils import STR_DTYPE_TO_TORCH_DTYPE
from aphrodite.multimodal.utils import rescale_image_size

from ..conftest import IMAGE_ASSETS, AphroditeRunner, HfRunner, _ImageAssets
from .utils import check_logprobs_close

pytestmark = pytest.mark.vlm

HF_IMAGE_PROMPTS = IMAGE_ASSETS.prompts({
    "stop_sign":
    "USER: <image>\nWhat's the content of the image?\nASSISTANT:",
    "cherry_blossom":
    "USER: <image>\nWhat is the season?\nASSISTANT:",
})

models = [
    "llava-hf/llava-1.5-7b-hf",
    # TODO: Get this model to produce meaningful output in Aphrodite
    # "TIGER-Lab/Mantis-8B-siglip-llama3",
]


def aphrodite_to_hf_output(aphrodite_output: Tuple[List[int], str,
                                         Optional[SampleLogprobs]],
                      model: str):
    """Sanitize aphrodite output to be comparable with hf output."""
    output_ids, output_str, out_logprobs = aphrodite_output

    config = AutoConfig.from_pretrained(model)
    image_token_id = config.image_token_index

    tokenizer = AutoTokenizer.from_pretrained(model)
    eos_token_id = tokenizer.eos_token_id

    hf_output_ids = [
        token_id for idx, token_id in enumerate(output_ids)
        if token_id != image_token_id or output_ids[idx - 1] != image_token_id
    ]

    assert output_str[0] == " "
    hf_output_str = output_str[1:]
    if hf_output_ids[-1] == eos_token_id:
        hf_output_str = hf_output_str + tokenizer.decode(eos_token_id)

    return hf_output_ids, hf_output_str, out_logprobs


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
    # NOTE: For local use; this isn't tested in CI yet (see TODO above)
    if model.startswith("TIGER-Lab/Mantis"):
        from mantis.models.mllava import MLlavaProcessor

        torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[dtype]
        mantis_processor = MLlavaProcessor.from_pretrained(
            model, torch_dtype=torch_dtype)
        assert isinstance(mantis_processor, MLlavaProcessor)
    else:
        mantis_processor = None

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
                     dtype=dtype,
                     tensor_parallel_size=tensor_parallel_size,
                     distributed_executor_backend=distributed_executor_backend,
                     enforce_eager=True) as aphrodite_model:
        aphrodite_outputs_per_image = [
            aphrodite_model.generate_greedy_logprobs(prompts,
                                                max_tokens,
                                                num_logprobs=num_logprobs,
                                                images=images)
            for prompts, images in inputs_per_image
        ]

    if mantis_processor is not None:

        def process(hf_inputs: BatchEncoding):
            hf_inputs["pixel_values"] = hf_inputs["pixel_values"] \
                .to(torch_dtype)  # type: ignore
            return hf_inputs
    else:

        def process(hf_inputs: BatchEncoding):
            return hf_inputs

    with hf_runner(model,
                   dtype=dtype,
                   postprocess_inputs=process,
                   auto_cls=AutoModelForVision2Seq) as hf_model:
        hf_outputs_per_image = [
            hf_model.generate_greedy_logprobs_limit(prompts,
                                                    max_tokens,
                                                    num_logprobs=num_logprobs,
                                                    images=images)
            for prompts, images in inputs_per_image
        ]

    for hf_outputs, aphrodite_outputs in zip(hf_outputs_per_image,
                                        aphrodite_outputs_per_image):
        # TODO: Check whether using original CLIPVisionModel can improve
        # consistency against HF
        check_logprobs_close(
            outputs_0_lst=hf_outputs,
            outputs_1_lst=[
                aphrodite_to_hf_output(aphrodite_output, model)
                for aphrodite_output in aphrodite_outputs
            ],
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
@pytest.mark.parametrize("dtype", ["half"])
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


@pytest.mark.parametrize("model", models)
def test_context_length_too_short(aphrodite_runner, image_assets, model):
    images = [asset.pil_image for asset in image_assets]
    with pytest.raises(ValueError, match="too long to fit into the model"):
        aphrodite_model = aphrodite_runner(
            model,
            max_model_len=128,  # LLaVA has a feature size of 576
            enforce_eager=True,
        )
        with aphrodite_model:
            aphrodite_model.generate_greedy([HF_IMAGE_PROMPTS[0]],
                                       max_tokens=1,
                                       images=[images[0]])
