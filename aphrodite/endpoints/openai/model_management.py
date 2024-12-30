from dataclasses import fields
from typing import Any, ClassVar, Dict

from pydantic import ConfigDict, Field, model_validator

from aphrodite.common.utils import FlexibleArgumentParser
from aphrodite.endpoints.openai.args import make_arg_parser
from aphrodite.endpoints.openai.protocol import OpenAIBaseModel
from aphrodite.engine.args_tools import EngineArgs


class ModelLoadRequest(OpenAIBaseModel):
    """Request to load a new model with optional configuration."""
    model_config = ConfigDict(extra='allow')

    model: str = Field(..., description="The model to load")

    _available_args: ClassVar[Dict[str, Any]] = {}

    @classmethod
    def get_available_args(cls) -> Dict[str, Any]:
        """Get all available arguments and their default values."""
        if not cls._available_args:
            engine_args = {
                field.name: field.default
                for field in fields(EngineArgs)
                if field.name != 'model'
            }

            parser = FlexibleArgumentParser()
            parser = make_arg_parser(parser)
            server_args = vars(parser.parse_args([]))

            cls._available_args = {**engine_args, **server_args}

        return cls._available_args

    @model_validator(mode='before')
    @classmethod
    def validate_args(cls, values):
        """Validate that all provided args are valid."""
        available_args = cls.get_available_args()
        for key in values:
            if key != "model" and key not in available_args:
                raise ValueError(f"Unknown argument: {key}")
        return values
