import itertools
import logging
from typing import Optional

import fire
import torch
from transformers import GenerationConfig, PreTrainedModel

from models import init_adapter
from dist import get_local_rank, get_rank, get_world_size
from model import accel_model, init_model
from session import BasicSessionManagerWithHistory
from utils import BasicStreamer, TerminalIO, control

logger = logging.getLogger(__name__)

def set_logging(log_file: str, debug: bool):
    torch.set_printoptions(linewidth=120)
    level = logging.DEBUG if debug else logging.INFO
    log_file = log_file or 'chat.log'
    if r := get_rank() != 0:
        log_file = log_file + f".{r}"
    logging.basicConfig(level=level,
                        format=('%(filename)s: '
                                '%(levelname)s: '
                                '%(funcName)s(): '
                                '%(lineno)d:\t'
                                '%(message)s'),
                        filename=log_file,
                        filemode='w')
    print(f"Worker {get_rank()} logging to {log_file}")

def main(
        model_path: str,
        tokenizer_path: Optional[str] = None,
        accel: Optional[str] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.95,
        seed: int = 0,
        use_fast_tokenizer: bool = True,
        max_alloc: int = 4096,
        max_session_len: int = None,
        log_file: Optional[str] = None,
        debug: bool = False,
        adapter: Optional[str] = None,
):
    set_logging(log_file, debug)

    torch.manual_seed(seed)

    local_rank = get_local_rank()
    world_size = get_world_size()

    if not tokenizer_path:
        tokenizer_path = model_path

    model, tokenizer = init_model(
        model_path,
        tokenizer_path,
        use_fast_tokenizer=use_fast_tokenizer,
    )

    model = init_adapter(model, tokenizer, adapter)

    model: PreTrainedModel = accel_model(model,
                                         accel,
                                         max_alloc=max_alloc,
                                         tp_size=world_size)
    
    warmup_config = GenerationConfig(
        max_new_tokens=1,
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=top_p,
    )
    model.generate(torch.tensor([[6]], device=get_local_rank()), warmup_config)

    get_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=top_p,
    )

    max_session_len = max_alloc if max_session_len is None else max_session_len
    sm = BasicSessionManagerWithHistory(max_session_len=max_session_len,
                                        start_ids=adapter.start_ids,
                                        sep_ids=adapter.sep_ids)
    io = TerminalIO()
    streamer = BasicStreamer(adapter.decode, io.output)

    for r in itertools.count(1):
        logger.info(f"Round {r}")

        prompt: str = io.input()
        logger.info(f"User input: {prompt}")

        if control(prompt, gen_config, sm):
            continue

        input_ids = adapter.encode_and_decorate(prompt)
        logger.info(f"Input IDs:\n{input_ids}")

        input_ids = sm.prepend_history(input_ids)
        logger.info(f"Input IDs with history:\n{input_ids}")

        input_ids = input_ids.cuda(local_rank)
        output = model.generate(input_ids,
                                gen_config,
                                streamer=streamer,
                                stopping_criteria=adapter.stopping_criteria)
        logger.info(f"Output:\n{output}")

        sm.add_to_history(output)

def cli():
    fire.Fire(main)

if __name__ == '__main__':
    cli()