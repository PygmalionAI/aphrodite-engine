"""
These types are defined in this file to avoid importing
aphrodite.engine.metrics and therefore importing prometheus_client.
This is required due to usage of Prometheus multiprocess mode to enable 
metrics after splitting out the uvicorn process from the engine process.
Prometheus multiprocess mode requires setting PROMETHEUS_MULTIPROC_DIR
before prometheus_client is imported. Typically, this is done by setting
the env variable before launch, but since we are a library, we need to
do this in Python code and lazily import prometheus_client.
"""
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from aphrodite.spec_decode.metrics import SpecDecodeWorkerMetrics


@dataclass
class Stats:
    """Created by AphroditeEngine for use by StatLogger."""
    now: float
    # System stats (should have _sys suffix)
    #   Scheduler State
    num_running_sys: int
    num_waiting_sys: int
    num_swapped_sys: int
    #   KV Cache Usage in %
    gpu_cache_usage_sys: float
    cpu_cache_usage_sys: float
    # Iteration stats (should have _iter suffix)
    num_prompt_tokens_iter: int
    num_generation_tokens_iter: int
    time_to_first_tokens_iter: List[float]
    time_per_output_tokens_iter: List[float]
    num_preemption_iter: int
    # Request stats (should have _requests suffix)
    #   Latency
    time_e2e_requests: List[float]
    #   Metadata
    num_prompt_tokens_requests: List[int]
    num_generation_tokens_requests: List[int]
    best_of_requests: List[int]
    n_requests: List[int]
    finished_reason_requests: List[str]
    spec_decode_metrics: Optional["SpecDecodeWorkerMetrics"] = None
class SupportsMetricsInfo(Protocol):
    def metrics_info(self) -> Dict[str, str]:
        ...
class StatLoggerBase(ABC):
    """Base class for StatLogger."""
    def __init__(self, local_interval: float,
                 per_request_logging: bool = False) -> None:
        # Tracked stats over current local logging interval.
        self.num_prompt_tokens: List[int] = []
        self.num_generation_tokens: List[int] = []
        self.last_local_log = time.time()
        self.local_interval = local_interval
        self.per_request_logging = per_request_logging
        self.current_request_stats: Optional[Dict[str, Any]] = None
        self.spec_decode_metrics: Optional["SpecDecodeWorkerMetrics"] = None

    @abstractmethod
    def log(self, stats: Stats) -> None:
        raise NotImplementedError

    @abstractmethod
    def log_per_request(self, request_id: str,
                        stats: Dict[str, Any]) -> None:
        raise NotImplementedError
    
    def start_request(self, request_id: str) -> None:
        if self.per_request_logging:
            self.current_request_stats = {
                "request_id": request_id,
                "start_time": time.time(),
                "prompt_tokens": 0.0,
                "generation_tokens": 0.0,
                "gpu_cache_usage": 0.0,
                "cpu_cache_usage": 0.0
            }
    
    def end_request(self) -> None:
        if self.per_request_logging and self.current_request_stats:
            end_time = time.time()
            duration = end_time - self.current_request_stats["start_time"]
            prompt_throughput = self.current_request_stats["prompt_tokens"]
            generation_throughput = self.current_request_stats["generation_tokens"]

            self.log_per_request(self.current_request_stats["request_id"], {
                "duration": duration,
                "prompt_throughput": prompt_throughput,
                "generation_throughput": generation_throughput,
                "gpu_cache_usage": self.current_request_stats["gpu_cache_usage"],
                "cpu_cache_usage": self.current_request_stats["cpu_cache_usage"]
            })

            self.current_request_stats = None

    @abstractmethod
    def info(self, type: str, obj: SupportsMetricsInfo) -> None:
        raise NotImplementedError
    def maybe_update_spec_decode_metrics(self, stats: Stats):
        """Save spec decode metrics (since they are unlikely
        to be emitted at same time as log interval)."""
        if stats.spec_decode_metrics is not None:
            self.spec_decode_metrics = stats.spec_decode_metrics
