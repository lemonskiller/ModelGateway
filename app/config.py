from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


VALID_MODES = {"hot", "warm", "cold"}
VALID_BACKENDS = {"managed_vllm", "external_openai", "ollama"}


@dataclass(frozen=True)
class ModelSpec:
    id: str
    path: str = ""
    served_model_name: str = ""
    gpu_group: tuple[int, ...] = ()
    port: int = 0
    mode: str = "cold"
    backend: str = "managed_vllm"
    base_url: str | None = None
    priority: int = 0
    idle_ttl: int = 1800
    quantization: str | None = None
    max_model_len: int | None = None
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    gpu_memory_utilization: float = 0.82
    tensor_parallel_size: int | None = None
    extra_args: tuple[str, ...] = field(default_factory=tuple)
    worker_api_key: str = ""
    enabled: bool = True

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "ModelSpec":
        required = ("id", "served_model_name")
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"model config is missing required fields: {', '.join(missing)}")

        backend = str(value.get("backend", "managed_vllm")).lower()
        if backend not in VALID_BACKENDS:
            raise ValueError(f"unsupported model backend: {backend}")

        mode = str(value.get("mode", "cold")).lower()
        if mode not in VALID_MODES:
            raise ValueError(f"unsupported model mode: {mode}")

        if backend == "managed_vllm":
            managed_required = ("path", "gpu_group", "port")
            missing = [key for key in managed_required if key not in value]
            if missing:
                raise ValueError(
                    f"managed vLLM model {value['id']} is missing required fields: "
                    + ", ".join(missing)
                )
        else:
            if not value.get("base_url"):
                raise ValueError(f"external model {value['id']} must define base_url")

        gpu_group = tuple(int(gpu) for gpu in value.get("gpu_group", []))
        if backend == "managed_vllm" and not gpu_group:
            raise ValueError(f"model {value['id']} must reserve at least one GPU")

        return cls(
            id=str(value["id"]),
            path=str(value.get("path", "")),
            served_model_name=str(value["served_model_name"]),
            gpu_group=gpu_group,
            port=int(value.get("port", 0)),
            mode=mode,
            backend=backend,
            base_url=_normalize_base_url(value.get("base_url")),
            priority=int(value.get("priority", 0)),
            idle_ttl=int(value.get("idle_ttl", 1800)),
            quantization=value.get("quantization"),
            max_model_len=_optional_int(value.get("max_model_len")),
            max_num_seqs=_optional_int(value.get("max_num_seqs")),
            max_num_batched_tokens=_optional_int(value.get("max_num_batched_tokens")),
            gpu_memory_utilization=float(value.get("gpu_memory_utilization", 0.82)),
            tensor_parallel_size=_optional_int(value.get("tensor_parallel_size")),
            extra_args=tuple(str(arg) for arg in value.get("extra_args", [])),
            worker_api_key=str(value.get("worker_api_key", "")),
            enabled=bool(value.get("enabled", True)),
        )

    def vllm_command(self, binary: str = "vllm") -> list[str]:
        command = [binary, "serve", self.path, "--served-model-name", self.served_model_name]
        if self.quantization:
            command.extend(["--quantization", self.quantization])
        if self.max_model_len is not None:
            command.extend(["--max-model-len", str(self.max_model_len)])
        if self.max_num_seqs is not None:
            command.extend(["--max-num-seqs", str(self.max_num_seqs)])
        if self.max_num_batched_tokens is not None:
            command.extend(["--max-num-batched-tokens", str(self.max_num_batched_tokens)])
        if self.gpu_memory_utilization:
            command.extend(["--gpu-memory-utilization", str(self.gpu_memory_utilization)])
        if self.tensor_parallel_size is not None:
            command.extend(["--tensor-parallel-size", str(self.tensor_parallel_size)])
        if self.mode == "warm":
            command.append("--enable-sleep-mode")
        command.extend(["--port", str(self.port)])
        command.extend(self.extra_args)
        return command

    @property
    def worker_root_url(self) -> str:
        if self.base_url:
            return self.base_url.removesuffix("/v1").rstrip("/")
        return f"http://127.0.0.1:{self.port}"

    @property
    def worker_api_url(self) -> str:
        if self.base_url:
            return self.base_url
        return f"{self.worker_root_url}/v1"


@dataclass(frozen=True)
class GatewayConfig:
    models: dict[str, ModelSpec]
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str = ""
    admin_key: str = ""
    vllm_binary: str = "vllm"
    startup_timeout: float = 600.0
    request_timeout: float = 300.0
    drain_timeout: float = 120.0
    poll_interval: float = 1.0
    reaper_interval: float = 15.0
    max_queue_size_per_model: int = 32

    @classmethod
    def from_file(cls, path: str | Path) -> "GatewayConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}

        gateway = raw.get("gateway", {})
        specs: dict[str, ModelSpec] = {}
        for item in raw.get("models", []):
            spec = ModelSpec.from_mapping(item)
            if spec.id in specs:
                raise ValueError(f"duplicate model id: {spec.id}")
            specs[spec.id] = spec

        if not specs:
            raise ValueError("at least one model must be configured")

        return cls(
            models=specs,
            host=str(gateway.get("host", "0.0.0.0")),
            port=int(gateway.get("port", 8000)),
            api_key=os.getenv("MODEL_GATEWAY_API_KEY", str(gateway.get("api_key", ""))),
            admin_key=os.getenv("MODEL_GATEWAY_ADMIN_KEY", str(gateway.get("admin_key", ""))),
            vllm_binary=os.getenv("VLLM_BIN", str(gateway.get("vllm_binary", "vllm"))),
            startup_timeout=float(gateway.get("startup_timeout", 600)),
            request_timeout=float(gateway.get("request_timeout", 300)),
            drain_timeout=float(gateway.get("drain_timeout", 120)),
            poll_interval=float(gateway.get("poll_interval", 1)),
            reaper_interval=float(gateway.get("reaper_interval", 15)),
            max_queue_size_per_model=int(gateway.get("max_queue_size_per_model", 32)),
        )


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _normalize_base_url(value: Any) -> str | None:
    if value is None:
        return None
    base_url = str(value).rstrip("/")
    if not base_url:
        return None
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url
