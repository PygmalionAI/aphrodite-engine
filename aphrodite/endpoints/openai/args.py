"""
This file contains the command line arguments for Aphrodite's
OpenAI-compatible server. It is kept in a separate file for documentation
purposes.
"""

import argparse
import json

from aphrodite.engine.args_tools import AsyncEngineArgs
from aphrodite.endpoints.openai.serving_engine import LoRA


class LoRAParserAction(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        lora_list = []
        for item in values:
            name, path = item.split('=')
            lora_list.append(LoRA(name, path))
        setattr(namespace, self.dest, lora_list)


def make_arg_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(
            description="Aphrodite OpenAI-Compatible RESTful API server.")
    parser.add_argument("--host", type=str, default=None, help="host name")
    parser.add_argument("--port", type=int, default=2242, help="port number")
    parser.add_argument("--allow-credentials",
                        action="store_true",
                        help="allow credentials")
    parser.add_argument("--allowed-origins",
                        type=json.loads,
                        default=["*"],
                        help="allowed origins")
    parser.add_argument("--allowed-methods",
                        type=json.loads,
                        default=["*"],
                        help="allowed methods")
    parser.add_argument("--allowed-headers",
                        type=json.loads,
                        default=["*"],
                        help="allowed headers")
    parser.add_argument(
        "--api-keys",
        type=str,
        default=None,
        help=
        "If provided, the server will require this key to be presented in the "
        "header.")
    parser.add_argument(
        "--admin-key",
        type=str,
        default=None,
        help=
        "If provided, the server will require this key to be presented in the "
        "header for admin operations.")
    parser.add_argument(
        "--launch-kobold-api",
        action="store_true",
        help=
        "Launch the Kobold API server in addition to the OpenAI API server.")
    parser.add_argument("--max-length",
                        type=int,
                        default=256,
                        help="The maximum length of the generated response. "
                        "For use with Kobold Horde.")
    parser.add_argument("--served-model-name",
                        type=str,
                        default=None,
                        help="The model name used in the API. If not "
                        "specified, the model name will be the same as "
                        "the huggingface name.")
    parser.add_argument(
        "--lora-modules",
        type=str,
        default=None,
        nargs='+',
        action=LoRAParserAction,
        help=
        "LoRA module configurations in the format name=path. Multiple modules "
        "can be specified.")
    parser.add_argument("--chat-template",
                        type=str,
                        default=None,
                        help="The file path to the chat template, "
                        "or the template in single-line form "
                        "for the specified model")
    parser.add_argument("--response-role",
                        type=str,
                        default="assistant",
                        help="The role name to return if "
                        "`request.add_generation_prompt=true`.")
    parser.add_argument("--ssl-keyfile",
                        type=str,
                        default=None,
                        help="The file path to the SSL key file")
    parser.add_argument("--ssl-certfile",
                        type=str,
                        default=None,
                        help="The file path to the SSL cert file")
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy")
    parser.add_argument(
        "--middleware",
        type=str,
        action="append",
        default=[],
        help="Additional ASGI middleware to apply to the app. "
        "We accept multiple --middleware arguments. "
        "The value should be an import path. "
        "If a function is provided, Aphrodite will add it to the server using "
        "@app.middleware('http'). "
        "If a class is provided, Aphrodite will add it to the server using "
        "app.add_middleware(). ")

    parser = AsyncEngineArgs.add_cli_args(parser)
    return parser
