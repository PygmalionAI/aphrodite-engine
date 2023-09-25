import argparse
import logging
import queue
import warnings
from typing import List, Optional

import pynvml
import torch
import torch.multiprocessing as mp
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from model import accel_model, init_model


def safe_numel(free_mem, model_size, max_intermediate):
    return int(free_mem - model_size) // max_intermediate

def avail_gpus(percentage=0.88):
    """Detect available GPUS.
    
    Args:
        percentage (float): The minimum percentage of free memory to be 
            considered available.
    
    Return:
        A list of GPU IDs.
        Average free memory on a single GPU.
    """
    gpus = []
    mems = []
    pynvml.nvmlInit()
    for i in range(torch.cuda.device_count()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(i))
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free, total = int(mem_info.free), int(mem_info.total)

        if free / total > percentage:
            gpus.append(i)
            mems.append(free)
    
    pynvml.nvmlShutdown()

    if len(gpus) == 0:
        raise RuntimeError('No GPUs available.')
    
    return gpus, sum(mems) / len(mems)


@torch.no_grad()
def decode_single(model: PreTrainedModel,
                  input_ids: torch.Tensor,
                  attention_mask: torch.Tensor = None,
                  return_logits=True):
    """Decode a single batch.

    Args:
        model (PreTrainedModel): Pretrained model.
        input_ids (torch.Tensor): A batch of input ids.
        attention_mask (torch.Tensor): A batch of attention mask.

    Returns:
        torch.Tensor: A batch of probabilities (on CPU).

    Note:
        This function assumes input_ids[i] = [bos, x1, x2, ..., xn]
        and returns prob = [p(x1|bos), p(x2|bos,x1), ..., p(xn|bos,...xn-1)]
        So probs are shorter than input_ids by 1.
    """
    
    # call causal LM forward
    outputs = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                    output_attentions=False,
                    use_cache=False,
                    return_dict=True)
    logits = outputs.logits

    if not return_logits:
        torch.softmax(logits, dim=-1, out=logits)

        shift_labels = input_ids[..., 1:].contiguous()
        shift_probs = logits[..., :-1, :].contiguous()
        logits = torch.gather(shift_probs, -1, shift_labels.unsqueeze(-1))

    if attention_mask is not None:
        logits *= attention_mask[..., None]

    logits = logits.cpu()

    return logits

def worker_fn(model_path: str,
              inq: mp.Queue,
              outq: mp.Queue,
              accel: Optional[str] = None,
              gpu_id=0):
    model, _ = init_model(model_path)
    model = model.eval()
    model = accel_model(model, accel, gpu_id=gpu_id)

    while True:
        try:
            idx, args = inq.get(timeout=1)
        except queue.Empty:
            continue

        if idx is None:
            print(f"Worker {gpu_id} received exit signal.")
            break

        input_ids, input_lens, *args = args

        input_ids = input_ids.cuda(gpu_id)
        max_len = max(input_lens)
        assert max_len == input_ids.size(-1), \
            f'input_ids.shape = {input_ids.shape}, max_len = {max_len}'
        
        input_lens = torch.tensor(input_lens, device=gpu_id)
        attention_mask = \
            torch.arrange(max_len, device=gpu_id)[None, :] < input_lens[:, None]
        
        assert attention_mask.shape == input_ids.shape, \
            f'attention_mask.shape = {attention_mask.shape}'
        
        try:
            probs = decoce_single(model, input_ids, attention_mask, *args)
        except torch.cuda.OutOfMemoryError:
            warnings.warn(
                f"OOM on GPU {gpu_id}, discard prompts at indices {idx}.")
            probs = torch.empty((input_ids.size(0), 0),
                                dtype=torch.float32,
                                device='cpu')
        
        outq.put((idx, probs))

    print(f"Exiting worker {gpu_id} ...")
    inq.close()
    outq.close()
    print(f"Worker {gpu_id} finished.")

class Engine:
    """Multi-GPU decoding engine.

    Args:
        model_path(str): Path to the pretrained model.
        tokenizer_path(str, optional): Path to the pretrained tokenizer.
            Defaults to None. Either tokenizer_path or tokenizer should
            be provided.
        tokenizer(PreTrainedTokenizerBase, optional): Pre-configured
            tokenizer. Defaults to None. Either tokenizer_path or
            tokenizer should be provided.
        accel(str, optional): Acceleration method. Defaults to None.
        gpu_mem_percentage(float, optional): GPU with memory larger than
            this value are considered available and ready for decoding use.
        model_size_byte(float, optional): (Approximate) memory cost per token
            in bytes. Defaults to 2e6 (2MB). `bytes_per_token` and `model_size_byte`
            are used to compute the maximum batch size for a given seq_length.
    """
    def __init__(self,
                 model_path: str,
                 tokenizer_path: Optional[str] = None,
                 tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 accel: Optional[str] = None,
                 gpu_mem_percentage: float = 0.96,
                 model_size_byte=14e9,
                 bytes_per_token=2e6):
        
        gpu_ids, mem = avail_gpus(gpu_mem_percentage)
        print(f"Available GPUs are: {gpu_ids}, ", end='')
        print(f"With {mem/2**30:.2f} GiB free.")

        ctx = mp.get_context('spawn')
        inq = ctx.Queue()
        outq = ctx.Queue()

        ps = []
        for id in gpu_ids:
            p = ctx.Process(target=worker_fn,
                            args=(model_path, inq, outq, accel, id))
            p.start()
            ps.append(p)
        
        if tokenizer is None:

            if tokenizer_path is None:
                tokenizer_path = model_path

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        self.gpu_ids = gpu_ids
        self.inq = inq
        self.outq = outq
        self.ps = ps
        self.tokenizer = tokenizer
        self.safe_numel = safe_numel(mem, model_size_byte, bytes_per_token)

    def clear_queue(self):
        for q in self.inq, self.outq:
            while not q.empty():
                q.get()
    
    def decode(self,
               token_ids: List[List[int]],
               sort=True,
               max_bs: int = 1024,
               pad=True,
               pad_token_id=2,
               return_logits=True):
        """Inference the model to compute probabilities.
        Args:
            token_ids (List[List[int]]): List of list of token ids.
            sort (bool, optional): Internally sort the prompts by length
                to achieve better efficiency. Defaults to True.
                Note: orders of returned probabilities are always the same
                as the input.
            max_bs (int, optional): Maximum batch size. Defaults to 1024.
            pad (bool, optional): Pad the prompts in every mini batch to the
                same length. Defaults to True. Set to False to save memory.
            return_logits (bool, optional): Return logits instead of probs.
            
        Returns:
            numpy.ndarray: Array of logits of shape [bsz, seqlen, vocab_size],
                with prob=0 padded, if pad is True
            List[numpy.ndarray]: List of logits without padding, if pad is False.
            
        Note:
            This function will accept input token_ids = [x0(=bos), x1, x2, ..., xn]
            and compute prob = [p(x1|x0), p(x2|x0,x1),...,p(xn|x0...xn-1)]
            So prob is shorter than input_ids by 1.
        """
        self.clear_queue()

        if sort:
            pids_and_indices = sorted(enumerate(token_ids),
                                      key=lambda i_and_x: len(i_and_x[1]))
        else:
            pids_and_indices = list(enumerate(token_ids))

        left = 0
        bs = max_bs

        while left < len(token_ids):
            if not sort:
                bs = max_bs

            right = min(left + bs, len(token_ids))

            sub_p_and_i = pids_and_indices[left:right]
            idx, sub_p = zip(*sub_p_and_i)

            input_ids = [torch.tensor(p) for p in sub_p]
            input_ids = pad_sequence(input_ids,
                                     batch_first=True,
                                     padding_value=pad_token_id)
            input_lens = [len(p) for p in sub_p]

            while input_ids.numel() > self.safe_numel:
                if bs == 1:
                    break
                bs = max(1, round(bs / 1.5))
                print(f"\nReduce bs to {bs} when seq len reaches "
                      f"{input_ids.shape[-1]}")
                idx = idx[:bs]
                input_lens = input_lens[:bs]
                input_ids = input_ids[:bs, :max(input_lens)]

            self.inq.put((idx, (input_ids, input_lens)))

            left += bs

            print(
                f"Distributing prompts {right}/{len(token_ids)},"
                f" {right/len(token_ids):.0%}",
                end='\r')
            
        print()

        all_probs = [None] * len(token_ids)
        count = 0

        while count < len(token_ids):
            idx, probs = self.outq.get()
            for i, p in zip(idx, probs):
                assert all_probs[i] is None
                all_probs[i] = p

            count += len(idx)
            print(
                f"Decoding and collecting ouputs "
                f"{count}/{len(token_ids)}, "
                f"{count/len(token_ids):.0%}",
                end='\r')
            
        print()

        if pad:
            all_probs = pad_sequence(all_probs, batch_first=True)
            all_probs = all_probs.cpu().numpy()
        else:
            all_probs = [p.cpu().numpy() for p in all_probs]
        
        return all_probs
    
    def __del__(self):
        print('Exiting engine ...')
        for _ in self.ps:
            self.inq.put((None, None))
        for p in self.ps:
            p.join(timeout=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path',
                        default='PygmalionAI/pygmalion-2-7b',
                        help='Path to HuggingFace model and tokenizer.')
    parser.add_argument('--test-path',
                        default='',
                        help='Path to text file, with each line containing a prompt.')
    parser.add_argument('-p', '--prompts',
                        nargs='*',
                        default=[
                            "What is a man? A miserable",
                            "A type is an assertion",
                            "A monad is just a monoid in the",
                            "Q: What is Jarilo-VI in Honkai Star Rail?\nA:"],
                            help="Prompt in commandline, please escape quotes with `\"\"`, lower in priority to --test-path")
    parser.add_argument('--save-to',
                        default='decode.out',
                        help='Save results to a file.')
    args = parser.parse_args()

    model_path = args.model_path
    test_path = args.test_path
    prompts = args.prompts

    logger = logging.getLogger(__name__)

    if test_path:
        with open(test_path, 'r') as f:
            prompts = f.readlines()

    prompts = [p.strip() for p in prompts]

    print(f"Model path: {model_path}")

    def _format(ts, start, end):
        if start < 0:
            start += len(ts)
        if end <= 0:
            end += len(ts)
        return '\n'.join(
            (f"{i}\t{t}" for i, t in zip(range(start, end), ts[start:end])))
    
    if len(prompts) > 10:
        print('Prompts:\n' + _format(prompts, 0, 5) + '\n.......\n' + _format(prompts, -5, 0))
    else:
        print('Prompts:\n' + _format(prompts, 0, 0))

    engine = Engine(model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'

    input_ids = tokenizer(prompts, padding=False) # type: ignore
    input_ids: List[List[int]] = input_ids.input_ids

    input_ids = [i for i in input_ids if len(i) >= args.min_len]
    if len(input_ids) < len(prompts):
        logger.warning(
            f"Filtered out {len(prompts) - len(input_ids)} prompts, "
            f"because they are shorter than {args.min_len}.")
        
    logits = engine.decode(input_ids)

    print(f"logits.shape = {logits.shape}")
    print(f"Dumping results to = {args.save_to}")

    torch.save(logits, args.save_to, pickle_protocol=4)

    del engine

    

