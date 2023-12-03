import logging
from abc import ABC, abstractmethod, abstractproperty
from typing import Any, List, Optional, Set, Type, Union

import torch

from aphrodite.slora.models import (TARGET_MODULES_QKV, LoRAModel,
                                    LoRAModelManager, LRUCacheLoRAModelManager)
from aphrodite.slora.request import LoRARequest
from aphrodite.slora.layers import LoRAMapping
from aphrodite.common.config import LoRAConfig

logger = logging.getLogger(__name__)


class AbstractWorkerLoRAManager(ABC):
    """Abstract class for managing LoRA models on the worker side."""

    def __init__(self, max_num_seqs: int, max_num_batched_tokens: int,
                 vocab_size: int, lora_config: LoRAConfig,
                 device: torch.device):
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.vocab_size = vocab_size
        self.device = device
        self.lora_config = lora_config

    @abstractproperty
    def is_enabled(self) -> bool:
        ...
    
    @abstractmethod
    def create_lora_adapter(
        self,
        model: torch.nn.Module,
        target_modules: Union[str, List[str]] = TARGET_MODULES_QKV,
    ) -> Any:
        ...
    
    @abstractmethod
    def apply_loras(self, lora_requests: List[LoRARequest],
                    lora_mapping: LoRAMapping) -> None:
        ...

    @abstractmethod
    def add_lora(self, lora_request: LoRARequest) -> bool:
        ...

    @abstractmethod
    def add_dummy_lora(self, lora_request: LoRARequest, rank: int) -> bool:
        ...
    
    @abstractmethod
    def remove_lora(self, lora_id: int) -> bool:
        ...

    @abstractmethod
    def remove_all_loras(self) -> bool:
        ...

    @abstractmethod
    def list_loras(self) -> Set[int]:
        ...


class DisabledWorkerLoRAManager(AbstractWorkerLoRAManager):
    """WorkerLoRAManager that does nothing."""

    @property
    def is_enabled(self) -> bool:
        return False

    def create_lora_adapter(
        self,
        model: torch.nn.Module,
        target_modules: Union[str, List[str]] = TARGET_MODULES_QKV,
    ) -> Any:
        return model

    def apply_loras(self, lora_requests: List[LoRARequest],
                    lora_mapping: LoRAMapping) -> None:
        return
    
    def add_lora(self, lora_request: LoRARequest) -> bool:
        return False

    def add_dummy_lora(self, lora_request: LoRARequest, rank: int) -> bool:
        return False

    def remove_lora(self, lora_id: int) -> bool:
        return False

    def remove_all_loras(self) -> bool:
        return
    
    def list_loras(self) -> Set[int]:
        return set()


class WorkerLoRAManager(AbstractWorkerLoRAManager):
    """WorkerLoRAManager that manages LoRA models on the worker side.
    
    Every request, the requested LoRAs will be loaded (unless they're
    already loaded), and every other LoRA will be unloaded."""

    _lora_manager_cls: Type[LoRAModelManager] = LoRAModelManager

    def __init__(
            self,
            max_num_seqs: int,
            max_num_batched_tokens: int,
            vocab_size: int,
            lora_config: LoRAConfig,
            device: torch.device,
            lora_model_cls: Type[LoRAModel] = LoRAModel,
    ):
        self._lora_manager: Optional[LoRAModelManager] = None
        self._lora_model_cls = lora_model_cls
        super().__init__(max_num_seqs, max_num_batched_tokens, vocab_size,
                         lora_config, device)
        
    @property
    def is_enabled(self) -> bool:
        return True

    def create_lora_adapter(
            self,
            model: torch.nn.Module,
            target_modules: Union[str, List[str]] = TARGET_MODULES_QKV,
    ) -> Any:
        lora_manager = create_lora_adapter(
            model,
            max_num_seqs=self.max_num_seqs,
            max_num_batched_tokens=self.max_num_batched_tokens,
            target_modules=target_modules,
            vocab_size=self.vocab_size,
            lora_config=self.lora_config,
            lora_manager_cls=self._lora_manager_cls,
        )
        self._lora_manager: LoRAModelManager = lora_manager
        return lora_manager.model

    def apply_loras(self, lora_requests: List[LoRARequest],
                    lora_mapping: LoRAMapping) -> None:
        self._apply_loras(lora_requests)
        self._lora_manager.set_row_lora_mapping(lora_mapping)

    def _apply_loras(self, lora_requests: List[LoRARequest]) -> None:
        loras_that_exist = self.list_loras()
        loras_map = {
            lora_request.lora_int_id: lora_request
            for lora_request in lora_requests if lora_request
        }
        if len(loras_map) > self._lora_manager.lora_slots:
            raise RuntimeError(
                f"number of requested LoRAs ({len(loras_map)}) is greater "
                "the number of GPU LoRA slots "
                f"({self._lora_manager.lora_slots})")

        new_loras = set(loras_map)
        loras_to_add = new_loras - loras_that_exist
        loras_to_remove = loras_that_exist - new_loras

        for lora_id in loras_to_remove:
            self.remove_lora(lora_id)

        for lora_id in loras_to_add:
            self.add_lora(loras_map[lora_id])

    def _load_lora(self, lora_request: LoRARequest) -> LoRAModel:
        try:
            lora = self._lora_model_clas.from_local_checkpoint(
                lora_request.lora_local_path,
                lora_model_id=lora_request.lora_int_id,
                device="cpu",
                dtype=self.lora_config.lora_dtype,
                target_embedding_padding=self.vocab_size +
                self.lora_config.lora_extra_vocab_size,
            )
        except Exception as e:
            raise RuntimeError(
                f"Loading LoRA {lora_request.lora_id} failed") from e
        if lora.rank > self.lora_config.max_lora_rank:
            raise ValueError(
                f"LoRA rank {lora.rank} is greather than max_lora_rank "
                f"{self.lora_config.max_lora_rank}.")
        return lora

    def add_dummy_lora(self, lora_request: LoRARequest, rank: int) -> bool:
        if lora_request.lora_int_id in self.list_loras():
            return False
        return self._lora_manager.add_lora(
            self._lora_manager.create_dummy_lora(lora_request.lora_int_id,
                                                 rank))
        
    def add_lora(self, lora_request: LoRARequest) -> bool:
        if lora_request.lora_int_id in self.list_loras():
            return False
        lora = self._load_lora(lora_request)
        loaded = self._lora_manager.add_lora(lora)
        self._lora_manager.activate_lora(lora.id)
        return loaded
    
    def remove_lora(self, lora_id: int) -> bool:
        return self._lora_manager.remove_lora(lora_id)

    def remove_all_loras(self) -> bool:
        self._lora_manager.remove_all_loras()

    def list_loras(self) -> Set[int]:
        return self._lora_manager.list_loras()


class LRUCacheWorkerLoRAManager(WorkerLoRAManager):
    """WorkerLoRAManager that manages LoRA models on the worker side.
    
    Uses an LRU cache. Every request, the requested LoRAs will be loaded
    (unless they're already loaded), and least recently used LoRAs will be
    unloaded if the cache is above capacity."""

    _lora_manager_cls: Type[LoRAModelManager] = LRUCacheLoRAModelManager


    def create_lora_adapter(
        self,
        model: torch.nn.Module,
        target_modules: Union[str, List[str]] = TARGET_MODULES_QKV,
    ) -> Any:
        lora_manager = create_lora_adapter(
            model,
            target_modules=target_modules,
            lora_manager_cls=self._lora_manager_cls,
            max_num_seqs=self.max_num_seqs,
            vocab_size=self.vocab_size,
            lora_config=self.lora_config,
            max_num_batched_tokens=self.max_num_batched_tokens,
        )
        self._lora_manager: LRUCacheLoRAModelManager = lora_manager
        return lora_manager.model

    def _apply_loras(self, lora_requests: List[LoRARequest]) -> None:
        loras_map = {
            lora_request.lora_int_id: lora_request
            for lora_request in lora_requests if lora_request
        }
        if len(loras_map) > self._lora_manager.lora_slots:
            raise RuntimeError(
                f"number of requested LoRAs ({len(loras_map)}) is greater "
                "the number of GPU LoRA slots "
                f"({self._lora_manager.lora_slots})")
        for lora in loras_map.values():
            self.add_lora(lora)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        if lora_request.lora_int_id not in self.list_loras():
            # remove before we load the new lora to save memory
            if len(self._lora_manager) + 1 > self._lora_manager.capacity:
                self._lora_manager.remove_oldest_lora()
            lora = self._load_lora(lora_request)
            loaded = self._lora_manager.add_lora(lora)
        else:
            # lora is already loaded, so we touch it to update its position
            # in the cache
            loaded = self._lora_manager.get_lora(self_request.lora_int_id)
            self._lora_manager.activate_lora(lora_request.lora_int_id)
        return loaded
            

