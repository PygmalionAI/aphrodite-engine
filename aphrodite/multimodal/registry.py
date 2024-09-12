import functools
from collections import UserDict
from typing import Dict, Mapping, Optional, Sequence

from loguru import logger

from aphrodite.common.config import ModelConfig, MultiModalConfig

from .audio import AudioPlugin
from .base import (MultiModalDataDict, MultiModalInputMapper, MultiModalInputs,
                   MultiModalPlugin, MultiModalTokensCalc, NestedTensors)
from .image import ImagePlugin


class _MultiModalLimits(UserDict):
    """
    Wraps `_limits_by_model` for a more informative error message
    when attempting to access a model that does not exist.
    """

    def __getitem__(self, key: ModelConfig) -> Dict[str, int]:
        try:
            return super().__getitem__(key)
        except KeyError as exc:
            msg = (f"Cannot find `mm_limits` for model={key.model}. Did you "
                   "forget to call `init_mm_limits_per_prompt`?")
            raise KeyError(msg) from exc

class MultiModalRegistry:
    """
    A registry to dispatch data processing
    according to its modality and the target model.

    The registry handles both external and internal data input.
    """

    DEFAULT_PLUGINS = (ImagePlugin(), AudioPlugin())

    def __init__(
            self,
            *,
            plugins: Sequence[MultiModalPlugin] = DEFAULT_PLUGINS) -> None:
        self._plugins = {p.get_data_key(): p for p in plugins}

        # This is used for non-multimodal models
        self._disabled_limits_per_plugin = {k: 0 for k in self._plugins}

        self._limits_by_model = _MultiModalLimits()

    def register_plugin(self, plugin: MultiModalPlugin) -> None:
        data_type_key = plugin.get_data_key()

        if data_type_key in self._plugins:
            logger.warning(
                "A plugin is already registered for data type "
                f"{data_type_key}, "
                f"and will be overwritten by the new plugin {plugin}.")

        self._plugins[data_type_key] = plugin

    def _get_plugin(self, data_type_key: str):
        plugin = self._plugins.get(data_type_key)
        if plugin is not None:
            return plugin

        msg = f"Unknown multi-modal data type: {data_type_key}"
        raise NotImplementedError(msg)

    def register_input_mapper(
        self,
        data_type_key: str,
        mapper: Optional[MultiModalInputMapper] = None,
    ):
        """
        Register an input mapper for a specific modality to a model class.
        See :meth:`MultiModalPlugin.register_input_mapper` for more details.
        """
        return self._get_plugin(data_type_key).register_input_mapper(mapper)

    def register_image_input_mapper(
        self,
        mapper: Optional[MultiModalInputMapper] = None,
    ):
        """
        Register an input mapper for image data to a model class.
        See :meth:`MultiModalPlugin.register_input_mapper` for more details.
        """
        return self.register_input_mapper("image", mapper)

    def map_input(self, model_config: ModelConfig,
                  data: MultiModalDataDict) -> MultiModalInputs:
        """
        Apply an input mapper to the data passed to the model.
        The data belonging to each modality is passed to the corresponding
        plugin which in turn converts the data into into keyword arguments
        via the input mapper registered for that model.
        See :meth:`MultiModalPlugin.map_input` for more details.
        Note:
            This should be called after :meth:`init_mm_limits_per_prompt`.
        """
        merged_dict: Dict[str, NestedTensors] = {}

        for data_key, data_value in data.items():
            plugin = self._get_plugin(data_key)

            num_items = len(data_value) if isinstance(data_value, list) else 1
            max_items = self._limits_by_model[model_config][data_key]
            if num_items > max_items:
                raise ValueError(
                    f"You set {data_key}={max_items} (or defaulted to 1) in "
                    f"`--limit-mm-per-prompt`, but found {num_items} items "
                    "in the same prompt.")

            input_dict = plugin.map_input(model_config, data_value)
            for input_key, input_tensor in input_dict.items():
                if input_key in merged_dict:
                    raise ValueError(f"The input mappers (keys={set(data)}) "
                                     f"resulted in a conflicting keyword "
                                     f"argument to `forward()`: {input_key}")
                merged_dict[input_key] = input_tensor
        return MultiModalInputs(merged_dict)

    def create_input_mapper(self, model_config: ModelConfig):
        """
        Create an input mapper (see :meth:`map_input`) for a specific model.
        """
        return functools.partial(self.map_input, model_config)

    def register_max_multimodal_tokens(
        self,
        data_type_key: str,
        max_mm_tokens: Optional[MultiModalTokensCalc] = None,
    ):
        """
        Register the maximum number of tokens, corresponding to a single
        instance of multimodal data belonging to a specific modality, that are
        passed to the language model for a model class.
        """
        return self._get_plugin(data_type_key) \
            .register_max_multimodal_tokens(max_mm_tokens)

    def register_max_image_tokens(
        self,
        max_mm_tokens: Optional[MultiModalTokensCalc] = None,
    ):
        """
        Register the maximum number of image tokens, corresponding to a single
        image, that are passed to the language model for a model class.
        """
        return self.register_max_multimodal_tokens("image", max_mm_tokens)

    def get_max_multimodal_tokens(self, model_config: ModelConfig) -> int:
        """
        Get the maximum number of multi-modal tokens
        for profiling the memory usage of a model.
        See :meth:`MultiModalPlugin.get_max_multimodal_tokens` for more details.
        Note:
            This should be called after :meth:`init_mm_limits_per_prompt`.
        """
        limits_per_plugin = self._limits_by_model[model_config]

        return sum((limits_per_plugin[key] *
                    plugin.get_max_multimodal_tokens(model_config))
                   for key, plugin in self._plugins.items())

    def init_mm_limits_per_prompt(
        self,
        model_config: ModelConfig,
        multimodal_config: Optional[MultiModalConfig],
    ) -> None:
        """
        Initialize the maximum number of multi-modal input instances for each
        modality that are allowed per prompt for a model class.
        """
        if model_config in self._limits_by_model:
            logger.warning(
                f"`mm_limits` has already been set for model="
                f"{model_config.model}, and will be overwritten by the "
                "new values.")

        if multimodal_config is None:
            limits_per_plugin = self._disabled_limits_per_plugin
        else:
            config_limits_per_plugin = multimodal_config.limit_per_prompt

            extra_keys = config_limits_per_plugin.keys() - self._plugins.keys()
            if extra_keys:
                logger.warning(
                    "Detected extra keys in `--limit-mm-per-prompt` which "
                    f"are not registered as multi-modal plugins: {extra_keys}."
                    " They will be ignored.")

            # NOTE: Currently the default is set to 1 for each plugin
            # TODO: Automatically determine the limits based on budget
            # once more models support multi-image inputs
            limits_per_plugin = {
                key: config_limits_per_plugin.get(key, 1)
                for key in self._plugins
            }

        self._limits_by_model[model_config] = limits_per_plugin

    def get_mm_limits_per_prompt(
        self,
        model_config: ModelConfig,
    ) -> Mapping[str, int]:
        """
        Get the maximum number of multi-modal input instances for each modality
        that are allowed per prompt for a model class.
        Note:
            This should be called after :meth:`init_mm_limits_per_prompt`.
        """
        return self._limits_by_model[model_config]
