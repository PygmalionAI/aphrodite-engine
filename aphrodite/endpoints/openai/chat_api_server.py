import argparse
from http import HTTPStatus
import json
import time
from typing import AsyncGenerator, Dict, List, Optional, Union, Any

import fastapi
from fastapi import BackgroundTasks, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

from aphrodite.engine.args_tools import AsyncEngineArgs
from aphrodite.engine.async_aphrodite import AsyncAphrodite
from aphrodite.endpoints.openai.protocol import (
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice, ChatCompletionStreamResponse,
    ChatMessage, DeltaMessage, ErrorResponse, LogProbs,
    ModelCard, ModelList, ModelPermission, UsageInfo)
from fastchat.conversation import Conversation, SeparatorStyle, get_conv_template
from aphrodite.common.logger import init_logger
from aphrodite.common.outputs import RequestOutput
from aphrodite.common.sampling_params import SamplingParams
from aphrodite.transformers_utils.tokenizer import get_tokenizer
from aphrodite.common.utils import random_uuid

TIMEOUT_KEEP_ALIVE = 5 #seconds