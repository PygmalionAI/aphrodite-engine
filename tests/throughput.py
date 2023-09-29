import argparse
import json
import random
import time
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase
from tqdm import tqdm

from aphrodite import LLM, SamplingParams
from aphrodite.transformers_utils.tokenizer import get_tokenizer

def sample_requests(
        dataset_path: str,
        num_request: int,
        tokenizer: PreTrainedTokenizerBase,
    ) -> List[Tuple[str, int, int]]:
    with open(dataset_path) as f:
        dataset = json.load(f)

    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    dataset = [(data["conversations"][0]["value"], data["conversations"][1]["value"]) for data in dataset]

    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    sampled_requests = random.sample(filtered_dataset, num_request)
    return sampled_requests

def run_aphrodite(
        requests: List[Tuple[str, int, int]],
        model: str,
        tokenizer: str,
        quantization: Optional[str],
        tensor_parallel_size: int,
        seed: int,
        n: int,
        use_beam_search: bool,
        trust_remote_code: bool,
    ) -> float:
        llm = LLM(
                model=model,
                tokenizer=tokenizer,
                quantization=quantization,
                tensor_parallel_size=tensor_parallel_size,
                seed=seed,
                trust_remote_code=trust_remote_code,
        )

        for prompt, _, output_len in requests:
            sampling_params = SamplingParams(
                    n=n,
                    temperature=0.0 if use_beam_search else 1.0,
                    top_p=1.0,
                    use_beam_search=use_beam_search,
                    ignore_eos=True,
                    max_tokens=output_len,
                )
            llm._add_request(
                    prompt=prompt,
                    prompt_token_ids=None,
                    sampling_params=sampling_params,
                )

        start = time.time()
        llm._run_engine(use_tqdm=True)
        end = time.time()
        return end - start


def run_hf(
        requests: List[Tuple[str, int, int]],
        model: str,
        tokenizer: PreTrainedTokenizerBase,
        n: int,
        use_beam_search: bool,
        max_batch_size: int,
        trust_remote_code: bool,
    ) -> float:
        assert not use_beam_search
        llm = AutoModelForCausalLM.from_pretrained(
                model, torch_dtype=torch.float16, trust_remote_code=trust_remote_code)
        if llm.config.model_type == "llama":
            tokenizer.pad_token = tokenizer.eos_token
        llm = llm.cuda()

        pbar = tqdm(total=len(requests))
        start = time.time()
        batch: List[str] = []
        max_prompt_len = 0
        max_output_len = 0
        for i in range(len(requests)):
            prompt, prompt_len, output_len = requests[i]
            batch.append(prompt)
            max_prompt_len = max(max_prompt_len, prompt_len)
            max_output_len = max(max_output_len, output_len)
            if len(batch) < max_batch_size and i != len(requests) - 1:
                _, next_prompt_len, next_output_len = requests[i + 1]
                if max(max_prompt_len, next_prompt_len) + max(max_output_len, next_output_len) <= 2048:
                    continue

            input_ids = tokenizer(batch, return_tensors="pt", padding=True).input_ids
            llm_outputs = llm.generate(
                    input_ids=input_ids.cuda(),
                    do_sample= not use_beam_search,
                    num_return_sequences=n,
                    temperature=1.0,
                    top_p=1.0,
                    use_cache=True,
                    max_new_tokens=max_output_len,
                )
            tokenizer.batch_decode(llm_outputs, skip_special_tokens=True)
            pbar.update(len(batch))

            batch = []
            max_prompt_len = 0
            max_output_len = 0
        end = time.time()
        return end - start

def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)

    tokenizer = get_tokenizer(args.tokenizer, trust_remote_code=args.trust_remote_code)
    requests = sample_requests(args.dataset, args.num_prompts, tokenizer)

    if args.backend == "aphrodite":
        elapsed_time = run_aphrodite(requests, args.model, args.tokenizer,
                                    args.quantization, args.tensor_parallel_size,
                                    args.seed, args.n, args.use_beam_search,
                                    args.trust_remote_code)
    elif args.backend == "hf":
        assert args.tensor_parallel_size == 1
        elapsed_time = run_hf(requests, args.model, tokenizer, args.n,
                              args.use_beam_search, args.hf_max_batch_size,
                              args.trust_remote_code)
    else:
        raise ValueError(f"Unknown backend: {args.backend}")
    total_num_tokens = sum(prompt_len + output_len for _, prompt_len, output_len in requests)
    print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
            f"{total_num_tokens / elapsed_time:.2f} tokens/s")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--backend", type=str, choices=["aphrodite", "hf"],
                        default="aphrodite")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the dataset.")
    parser.add_argument("--quantization", "-q", choices=['awq', None], default=None)
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--n", type=int, default=1,
                        help="Number of generated sequences per prompt.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--num-prompts", type=int, default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hf-max-batch-size", type=int, default=None,
                        help="Maximum batch size for HF backend.")
    parser.add_argument("--trust-remote-code",
                        action='store_true',
                        help='trust remote code from HF.')
    args = parser.parse_args()

    if args.backend == "aphrodite":
        if args.hf_max_batch_size is not None:
            raise ValueError("HF max batch size is only for HF backend.")
    elif args.backend == "hf":
        if args.hf_max_batch_size is None:
            raise ValueError("HF max batch size is required for HF backend.")
        if args.quantization is not None:
            raise ValueError("Quantization isn't currently supported for HF.")
    if args.tokenizer is None:
        args.tokenizer = args.model

    main(args)
