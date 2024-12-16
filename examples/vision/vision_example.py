"""
This example shows how to use Aphrodite for running offline inference 
with the correct prompt format on vision language models.
For most models, the prompt format should follow corresponding examples
on HuggingFace model repository.
"""
import os

from PIL import Image
from transformers import AutoTokenizer

from aphrodite import LLM, SamplingParams
from aphrodite.common.utils import FlexibleArgumentParser

# Input image and question
image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "burg.jpg")
image = Image.open(image_path).convert("RGB")
question = "What is the content of this image?"


# LLaVA-1.5
def run_llava(question):

    prompt = f"USER: <image>\n{question}\nASSISTANT:"

    llm = LLM(model="llava-hf/llava-1.5-7b-hf")

    return llm, prompt


# LLaVA-1.6/LLaVA-NeXT
def run_llava_next(question):

    prompt = f"[INST] <image>\n{question} [/INST]"
    llm = LLM(model="llava-hf/llava-v1.6-mistral-7b-hf")

    return llm, prompt


# Fuyu
def run_fuyu(question):

    prompt = f"{question}\n"
    llm = LLM(model="adept/fuyu-8b")

    return llm, prompt


# Phi-3-Vision
def run_phi3v(question):

    prompt = f"<|user|>\n<|image_1|>\n{question}<|end|>\n<|assistant|>\n"  # noqa: E501
    # Note: The default setting of max_num_seqs (256) and
    # max_model_len (128k) for this model may cause OOM.
    # You may lower either to run this example on lower-end GPUs.

    # In this example, we override max_num_seqs to 5 while
    # keeping the original context length of 128k.
    llm = LLM(
        model="microsoft/Phi-3.5-vision-instruct",
        trust_remote_code=True,
        max_num_seqs=5,
    )
    return llm, prompt


# PaliGemma
def run_paligemma(question):

    prompt = "caption en"
    llm = LLM(model="google/paligemma-3b-mix-224")

    return llm, prompt


# Chameleon
def run_chameleon(question):

    prompt = f"{question}<image>"
    llm = LLM(model="facebook/chameleon-7b")
    return llm, prompt


# MiniCPM-V
def run_minicpmv(question):

    # 2.0
    # The official repo doesn't work yet, so we need to use a fork for now
    # model_name = "HwwwH/MiniCPM-V-2"

    # 2.5
    model_name = "openbmb/MiniCPM-Llama3-V-2_5"
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
    )

    messages = [{
        'role': 'user',
        'content': f'(<image>./</image>)\n{question}'
    }]
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)
    return llm, prompt


# BLIP-2
def run_blip2(question):

    # BLIP-2 prompt format is inaccurate on HuggingFace model repository.
    # See https://huggingface.co/Salesforce/blip2-opt-2.7b/discussions/15#64ff02f3f8cf9e4f5b038262 #noqa
    prompt = f"Question: {question} Answer:"
    llm = LLM(model="Salesforce/blip2-opt-2.7b")
    return llm, prompt


# InternVL
def run_internvl(question):
    # Generally, InternVL can use chatml template for conversation
    TEMPLATE = "<|im_start|>User\n{prompt}<|im_end|>\n<|im_start|>Assistant\n"
    prompt = f"<image>\n{question}\n"
    prompt = TEMPLATE.format(prompt=prompt)
    llm = LLM(
        model="OpenGVLab/InternVL2-4B",
        trust_remote_code=True,
        max_num_seqs=28,
        tensor_parallel_size=2,
        max_model_len=8192,
    )
    return llm, prompt


model_example_map = {
    "llava": run_llava,
    "llava-next": run_llava_next,
    "fuyu": run_fuyu,
    "phi3_v": run_phi3v,
    "paligemma": run_paligemma,
    "chameleon": run_chameleon,
    "minicpmv": run_minicpmv,
    "blip-2": run_blip2,
    "internvl_chat": run_internvl,
}


def main(args):
    model = args.model_type
    if model not in model_example_map:
        raise ValueError(f"Model type {model} is not supported.")

    llm, prompt = model_example_map[model](question)

    # We set temperature to 0.2 so that outputs can be different
    # even when all prompts are identical when running batch inference.
    sampling_params = SamplingParams(temperature=0.2, max_tokens=128)

    assert args.num_prompts > 0
    if args.num_prompts == 1:
        # Single inference
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {
                "image": image
            },
        }

    else:
        # Batch inference
        inputs = [{
            "prompt": prompt,
            "multi_modal_data": {
                "image": image
            },
        } for _ in range(args.num_prompts)]

    outputs = llm.generate(inputs, sampling_params=sampling_params)

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description='Demo on using Aphrodite for offline inference with '
        'vision language models')
    parser.add_argument('--model-type',
                        '-m',
                        type=str,
                        default="llava",
                        choices=model_example_map.keys(),
                        help='Huggingface "model_type".')
    parser.add_argument('--num-prompts',
                        type=int,
                        default=1,
                        help='Number of prompts to run.')

    args = parser.parse_args()
    main(args)
