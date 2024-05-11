import argparse

from aphrodite.endpoints.openai.api_server import run_server
from aphrodite.endpoints.openai.args import make_arg_parser


def main():
    parser = argparse.ArgumentParser(description="Aphrodite CLI")
    subparsers = parser.add_subparsers()

    serve_parser = subparsers.add_parser(
        "run",
        help="Start the Aphrodite OpenAI Compatible API server",
        usage="aphrodite run <model_tag> [options]")
    make_arg_parser(serve_parser)
    # Override the `--model` optional argument, make it positional.
    serve_parser.add_argument("model",
                              type=str,
                              help="The model tag or path to"
                              " run.")
    serve_parser.set_defaults(func=run_server)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
