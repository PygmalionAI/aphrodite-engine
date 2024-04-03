from accelerate import init_empty_weights
from loguru import logger
from transformers import AutoConfig, AutoModelForCausalLM, LlamaForCausalLM

from aphrodite.modeling.hf_downloader import convert_gguf_to_state_dict
from aphrodite.transformers_utils.config import extract_gguf_config
from aphrodite.transformers_utils.tokenizer import convert_gguf_to_tokenizer


def convert_save_model(checkpoint, save_dir, max_shard_size):
    try:
        config = AutoConfig.from_pretrained(save_dir)
    except Exception:
        logger.warning(
            f"Unable to load config from {save_dir}, trying to extract from GGUF"
        )
        config = extract_gguf_config(checkpoint)

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    state_dict = convert_gguf_to_state_dict(checkpoint, config)
    logger.info(f"Saving model to {save_dir}...")
    model.save_pretrained(save_dir,
                          state_dict=state_dict,
                          max_shard_size=max_shard_size)


def convert_save_tokenizer(checkpoint, save_dir):
    logger.info("Converting tokenizer...")
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
        '--tokenizer',
        action='store_true',
        help='Extract the tokenizer from GGUF file. Only llama is supported')
    parser.add_argument(
        '--max-shard-size',
        default="5GB",
        type=str,
        help='Shard the model in specified shard size, e.g. 5GB')
    args = parser.parse_args()
    convert_save_model(args.input, args.output, args.max_shard_size)

    if args.tokenizer:
        convert_save_tokenizer(args.input, args.output)
