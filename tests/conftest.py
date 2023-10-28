from typing import List, Optional, Tuple

import pytest
import torch
from transformers import AutoModelForCausalLM

from aphrodite import LLM, SamplingParams
from aphrodite.transformers_utils.tokenizer import get_tokenizer

_TEST_PROMPTS = [
    # pylint: disable=line-too-long
    "Develop a detailed method for integrating a blockchain-based distributed ledger system into a pre-existing finance management application. The focus should be on ensuring security, transparency, and real-time updates of transactions.",
    "Design an AI-powered predictive analytics engine capable of identifying trends and patterns from unstructured data sets. The engine should be adaptable to different industry requirements such as healthcare, finance, and marketing.",
    "Construct a comprehensive model for a multi-cloud architecture that can smoothly transition between different cloud platforms (AWS, Google Cloud, Azure) without any interruption in service or loss of data.",
    "Propose a strategy for integrating Quantum Computing capabilities into existing high-performance computing (HPC) systems. The approach should consider potential challenges and solutions of Quantum-HPC integration.",
    "Create a robust cybersecurity framework for an Internet of Things (IoT) ecosystem. The framework should be capable of detecting, preventing, and mitigating potential security breaches.",
    "Develop a scalable high-frequency trading algorithm that uses machine learning to predict and respond to microtrends in financial markets. The algorithm should be capable of processing real-time data and executing trades within milliseconds.",
    "Translate the following English sentence into Japanese, French, and Swahili: 'The early bird catches the worm.'",
]


@pytest.fixture
def example_prompts() -> List[str]:
    return _TEST_PROMPTS


_STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.half,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
}


class HfRunner:

    def __init__(
        self,
        model_name: str,
        tokenizer_name: Optional[str] = None,
        dtype: str = "half",
    ) -> None:
        assert dtype in _STR_DTYPE_TO_TORCH_DTYPE
        torch_dtype = _STR_DTYPE_TO_TORCH_DTYPE[dtype]
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).cuda()
        if tokenizer_name is None:
            tokenizer_name = model_name
        self.tokenizer = get_tokenizer(tokenizer_name, trust_remote_code=True)

    def generate(
        self,
        prompts: List[str],
        **kwargs,
    ) -> List[Tuple[List[int], str]]:
        outputs: List[Tuple[List[int], str]] = []
        for prompt in prompts:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            output_ids = self.model.generate(
                input_ids.cuda(),
                use_cache=True,
                **kwargs,
            )
            output_str = self.tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            output_ids = output_ids.cpu().tolist()
            outputs.append((output_ids, output_str))
        return outputs

    def generate_greedy(
        self,
        prompts: List[str],
        max_tokens: int,
    ) -> List[Tuple[List[int], str]]:
        outputs = self.generate(prompts,
                                do_sample=False,
                                max_new_tokens=max_tokens)
        for i in range(len(outputs)):
            output_ids, output_str = outputs[i]
            outputs[i] = (output_ids[0], output_str[0])
        return outputs

    def generate_beam_search(
        self,
        prompts: List[str],
        beam_width: int,
        max_tokens: int,
    ) -> List[Tuple[List[int], str]]:
        outputs = self.generate(prompts,
                                do_sample=False,
                                max_new_tokens=max_tokens,
                                num_beams=beam_width,
                                num_return_sequences=beam_width)
        for i in range(len(outputs)):
            output_ids, output_str = outputs[i]
            for j in range(len(output_ids)):
                output_ids[j] = [
                    x for x in output_ids[j]
                    if x != self.tokenizer.pad_token_id
                ]
            outputs[i] = (output_ids, output_str)
        return outputs

    def generate_greedy_logprobs(
        self,
        prompts: List[str],
        max_tokens: int,
    ) -> List[List[torch.Tensor]]:
        all_logprobs = []
        for prompt in prompts:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            output = self.model.generate(
                input_ids.cuda(),
                use_cache=True,
                do_sample=False,
                max_new_tokens=max_tokens,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            seq_logprobs = []
            for hidden_states in output.hidden_states:
                last_hidden_states = hidden_states[-1][0]
                logits = torch.matmul(
                    last_hidden_states,
                    self.model.get_output_embeddings().weight.t(),
                )
                if self.model.get_output_embeddings().bias is not None:
                    logits += self.model.get_output_embeddings(
                    ).bias.unsqueeze(0)
                logprobs = torch.nn.functional.log_softmax(logits,
                                                           dim=-1,
                                                           dtype=torch.float32)
                seq_logprobs.append(logprobs)
            all_logprobs.append(seq_logprobs)
        return all_logprobs


@pytest.fixture
def hf_runner():
    return HfRunner


class AphroditeRunner:

    def __init__(
        self,
        model_name: str,
        tokenizer_name: Optional[str] = None,
        dtype: str = "half",
    ) -> None:
        self.model = LLM(
            model=model_name,
            tokenizer=tokenizer_name,
            trust_remote_code=True,
            dtype=dtype,
            swap_space=0,
        )

    def generate(
        self,
        prompts: List[str],
        sampling_params: SamplingParams,
    ) -> List[Tuple[List[int], str]]:
        req_outputs = self.model.generate(prompts,
                                          sampling_params=sampling_params)
        outputs = []
        for req_output in req_outputs:
            prompt_str = req_output.prompt
            prompt_ids = req_output.prompt_token_ids
            req_sample_output_ids = []
            req_sample_output_strs = []
            for sample in req_output.outputs:
                output_str = sample.text
                output_ids = sample.token_ids
                req_sample_output_ids.append(prompt_ids + output_ids)
                req_sample_output_strs.append(prompt_str + output_str)
            outputs.append((req_sample_output_ids, req_sample_output_strs))
        return outputs

    def generate_greedy(
        self,
        prompts: List[str],
        max_tokens: int,
    ) -> List[Tuple[List[int], str]]:
        greedy_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        outputs = self.generate(prompts, greedy_params)
        return [(output_ids[0], output_str[0])
                for output_ids, output_str in outputs]

    def generate_beam_search(
        self,
        prompts: List[str],
        beam_width: int,
        max_tokens: int,
    ) -> List[Tuple[List[int], str]]:
        beam_search_params = SamplingParams(n=beam_width,
                                            use_beam_search=True,
                                            temperature=0.0,
                                            max_tokens=max_tokens)
        outputs = self.generate(prompts, beam_search_params)
        return outputs


@pytest.fixture
def aphrodite_runner():
    return AphroditeRunner