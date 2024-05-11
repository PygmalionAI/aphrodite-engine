import torch
from loguru import logger
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from aphrodite.modeling.hf_downloader import convert_gguf_to_state_dict
from aphrodite.transformers_utils.config import extract_gguf_config
from aphrodite.transformers_utils.tokenizer import convert_gguf_to_tokenizer


def convert_save_model(checkpoint, unquantized_path, save_dir, max_shard_size):
    if unquantized_path is not None:
        config = AutoConfig.from_pretrained(unquantized_path)
    else:
        config = extract_gguf_config(checkpoint)

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    state_dict = convert_gguf_to_state_dict(checkpoint, config)
    logger.info(f"Saving model to {save_dir}...")
    model.save_pretrained(save_dir,
                          state_dict=state_dict,
                          max_shard_size=max_shard_size)


def convert_save_tokenizer(checkpoint, unquantized_path, save_dir):
    if unquantized_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(unquantized_path)
    else:
        tokenizer = convert_gguf_to_tokenizer(checkpoint)
    tokenizer.save_pretrained(save_dir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Convert GGUF checkpoints to torch')

    parser.add_argument('--input', type=str, help='The path to GGUF file')
    parser.add_argument('--output',
                        type=str,
                        help='The path to output directory')
    parser.add_argument(
        '--unquantized-path',
        default=None,
        type=str,
        help='The path to the unquantized model to copy config and tokenizer')
    parser.add_argument('--no-tokenizer',
                        action='store_true',
                        help='Do not try to copy or extract the tokenizer')
    parser.add_argument(
        '--max-shard-size',
        default="5GB",
        type=str,
        help='Shard the model in specified shard size, e.g. 5GB')
    args = parser.parse_args()
    convert_save_model(args.input, args.unquantized_path, args.output,
                       args.max_shard_size)
    if not args.no_tokenizer:
        convert_save_tokenizer(args.input, args.unquantized_path, args.output)
