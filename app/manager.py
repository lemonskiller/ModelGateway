from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

import httpx

from .config import GatewayConfig, ModelSpec

logger = logging.getLogger(__name__)


class GatewayError(Exception):
    status_code = 500


class ModelNotFound(GatewayError):
    status_code = 404


class ModelUnavailable(GatewayError):
    status_code = 503


class RequestQueueFull(GatewayError):
    status_code = 429


class Unauthorized(GatewayError):
    status_code = 401


@dataclass
class WorkerState:
    spec: ModelSpec
    status: str = "STOPPED"
    process: asyncio.subprocess.Process | None = None
    active_requests: int = 0
    pending_requests: int = 0
    last_used: float = 0.0
    log_task: asyncio.Task[None] | None = None
    requests_total: int = 0
    request_errors_total: int = 0
    request_duration_total: float = 0.0
    request_stream_total: int = 0
    request_input_tokens_total: int = 0
    request_output_tokens_total: int = 0
    request_tokens_total: int = 0


class ModelManager:
    def __init__(self, config: GatewayConfig, client: httpx.AsyncClient | None = None) -> None:
        self.config = config
        # Worker traffic is private/local and must not inherit shell HTTP(S)_PROXY settings.
        self.client = client or httpx.AsyncClient(trust_env=False)
        self.states = {model_id: WorkerState(spec) for model_id, spec in config.models.items()}
        self._gpu_locks: dict[int, asyncio.Lock] = {
            gpu: asyncio.Lock()
            for state in self.states.values()
            for gpu in state.spec.gpu_group
        }
        self._reaper_task: asyncio.Task[None] | None = None
        self._preload_task: asyncio.Task[None] | None = None
        self._closed = False

    async def start(self) -> None:
        self._reaper_task = asyncio.create_task(self._reaper(), name="model-gateway-reaper")
        self._preload_task = asyncio.create_task(
            self._preload_models(),
            name="model-gateway-preloader",
        )

    async def _preload_models(self) -> None:
        preload_models = [
            state.spec.id
            for state in self.states.values()
            if state.spec.enabled and (state.spec.mode == "hot" or state.spec.preload)
        ]
        if not preload_models:
            return
        results = await asyncio.gather(
            *(self._preload_model(model_id) for model_id in preload_models),
            return_exceptions=True,
        )
        for model_id, result in zip(preload_models, results):
            if isinstance(result, Exception):
                logger.error("preloaded model %s failed to start: %s", model_id, result)

    async def _preload_model(self, model_id: str) -> None:
        await self.ensure_ready(model_id)
        spec = self.states[model_id].spec
        if spec.backend == "managed_vllm" and spec.mode == "warm":
            await self.sleep(model_id)

    async def close(self) -> None:
        self._closed = True
        if self._preload_task:
            self._preload_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._preload_task
        if self._reaper_task:
            self._reaper_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reaper_task
        for state in self.states.values():
            await self._stop_worker(state)
        await self.client.aclose()

    def get_spec(self, model_id: str) -> ModelSpec:
        state = self.states.get(model_id)
        if not state or not state.spec.enabled:
            raise ModelNotFound(f"unknown or disabled model: {model_id}")
        return state.spec

    def snapshot(self, model_id: str) -> dict[str, object]:
        state = self.states.get(model_id)
        if not state:
            raise ModelNotFound(f"unknown model: {model_id}")
        return {
            "id": state.spec.id,
            "served_model_name": state.spec.served_model_name,
            "backend": state.spec.backend,
            "base_url": state.spec.base_url,
            "gpu_group": list(state.spec.gpu_group),
            "mode": state.spec.mode,
            "preload": state.spec.preload,
            "priority": state.spec.priority,
            "status": state.status,
            "active_requests": state.active_requests,
            "pending_requests": state.pending_requests,
            "last_used": state.last_used,
            "requests_total": state.requests_total,
            "request_errors_total": state.request_errors_total,
            "request_duration_total": state.request_duration_total,
            "request_stream_total": state.request_stream_total,
            "request_input_tokens_total": state.request_input_tokens_total,
            "request_output_tokens_total": state.request_output_tokens_total,
            "request_tokens_total": state.request_tokens_total,
        }

    def snapshots(self) -> list[dict[str, object]]:
        return [self.snapshot(model_id) for model_id in self.states]

    @asynccontextmanager
    async def lease(self, model_id: str) -> AsyncIterator[ModelSpec]:
        state = self.states.get(model_id)
        if not state or not state.spec.enabled:
            raise ModelNotFound(f"unknown or disabled model: {model_id}")
        if state.status != "READY":
            if state.pending_requests >= self.config.max_queue_size_per_model:
                raise RequestQueueFull(f"model queue is full: {model_id}")
            state.pending_requests += 1
            try:
                await self.ensure_ready(model_id)
            finally:
                state.pending_requests = max(0, state.pending_requests - 1)
        state.active_requests += 1
        state.last_used = time.monotonic()
        try:
            yield state.spec
        finally:
            state.active_requests = max(0, state.active_requests - 1)
            state.last_used = time.monotonic()

    async def ensure_ready(self, model_id: str) -> None:
        spec = self.get_spec(model_id)
        state = self.states[model_id]
        if state.status == "READY":
            state.last_used = time.monotonic()
            return

        async with self._reserve_gpus(spec.gpu_group):
            if state.status == "READY":
                state.last_used = time.monotonic()
                return
            if state.status == "SLEEPING":
                await self._evict_conflicts(spec)
                await self._wake_worker(state)
                return
            if spec.backend != "managed_vllm":
                await self._check_external_worker(state)
                return
            await self._evict_conflicts(spec)
            await self._start_worker(state)

    async def unload(self, model_id: str) -> None:
        spec = self.get_spec(model_id)
        async with self._reserve_gpus(spec.gpu_group):
            state = self.states[model_id]
            await self._drain(state)
            if state.spec.backend != "managed_vllm":
                state.status = "STOPPED"
                return
            await self._stop_worker(state)

    async def sleep(self, model_id: str) -> None:
        spec = self.get_spec(model_id)
        async with self._reserve_gpus(spec.gpu_group):
            state = self.states[model_id]
            await self._drain(state)
            if state.spec.backend != "managed_vllm":
                state.status = "STOPPED"
                return
            if state.spec.mode != "warm":
                raise ModelUnavailable(f"model {model_id} is not configured for warm sleep")
            if state.status == "READY":
                await self._sleep_worker(state)

    async def wake(self, model_id: str) -> None:
        await self.ensure_ready(model_id)

    def record_request(
        self,
        model_id: str,
        *,
        duration_seconds: float,
        success: bool,
        stream: bool,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
    ) -> None:
        state = self.states.get(model_id)
        if not state:
            raise ModelNotFound(f"unknown model: {model_id}")
        state.requests_total += 1
        if not success:
            state.request_errors_total += 1
        state.request_duration_total += max(0.0, float(duration_seconds))
        if stream:
            state.request_stream_total += 1
        if input_tokens is not None:
            state.request_input_tokens_total += max(0, int(input_tokens))
        if output_tokens is not None:
            state.request_output_tokens_total += max(0, int(output_tokens))
        if total_tokens is not None:
            state.request_tokens_total += max(0, int(total_tokens))

    async def _evict_conflicts(self, target: ModelSpec) -> None:
        target_gpus = set(target.gpu_group)
        conflicts = [
            state
            for state in self.states.values()
            if state.spec.id != target.id
            and state.status == "READY"
            and target_gpus.intersection(state.spec.gpu_group)
        ]
        conflicts.sort(key=lambda item: (item.spec.mode == "hot", item.spec.priority, item.last_used))
        for state in conflicts:
            if state.spec.mode == "hot" and target.priority <= state.spec.priority:
                raise ModelUnavailable(
                    f"GPU group {target.gpu_group} is occupied by hot model {state.spec.id}"
                )
            await self._drain(state)
            if state.spec.mode == "warm":
                try:
                    await self._sleep_worker(state)
                except Exception:
                    logger.exception("failed to sleep conflicting model %s; stopping it", state.spec.id)
                    await self._stop_worker(state)
            else:
                await self._stop_worker(state)

    async def _start_worker(self, state: WorkerState) -> None:
        state.status = "STARTING"
        environment = os.environ.copy()
        environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in state.spec.gpu_group)
        environment["CUDA_HOME"] = environment.get("VLLM_CUDA_HOME", "/usr/local/cuda")
        environment.setdefault("VLLM_USE_V1", "0")
        vllm_binary = state.spec.vllm_binary or self.config.vllm_binary
        cuda_library_paths = [
            "/usr/local/nvidia/lib64",
            "/usr/local/cuda/lib64",
            "/lib/x86_64-linux-gnu",
            "/usr/lib/x86_64-linux-gnu",
        ]
        cuda_library_paths.extend(_python_env_library_paths(vllm_binary))
        existing_library_path = environment.get("LD_LIBRARY_PATH")
        if existing_library_path:
            cuda_library_paths.append(existing_library_path)
        environment["LD_LIBRARY_PATH"] = ":".join(cuda_library_paths)
        if state.spec.mode == "warm":
            environment["VLLM_SERVER_DEV_MODE"] = "1"

        try:
            state.process = await asyncio.create_subprocess_exec(
                *state.spec.vllm_command(vllm_binary),
                env=environment,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            state.log_task = asyncio.create_task(self._drain_logs(state), name=f"logs-{state.spec.id}")
            await self._wait_ready(state)
            state.status = "READY"
            state.last_used = time.monotonic()
        except Exception as error:
            await self._stop_worker(state)
            state.status = "FAILED"
            raise ModelUnavailable(f"failed to start model {state.spec.id}: {error}") from error

    async def _check_external_worker(self, state: WorkerState) -> None:
        state.status = "CHECKING"
        try:
            response = await self.client.get(
                f"{state.spec.worker_api_url}/models",
                headers=self._worker_headers(state.spec),
                timeout=5,
            )
            response.raise_for_status()
            state.status = "READY"
            state.last_used = time.monotonic()
        except Exception as error:
            state.status = "FAILED"
            raise ModelUnavailable(f"external model {state.spec.id} is unavailable: {error}") from error

    async def _wait_ready(self, state: WorkerState) -> None:
        deadline = time.monotonic() + self.config.startup_timeout
        headers = self._worker_headers(state.spec)
        while time.monotonic() < deadline:
            if state.process and state.process.returncode is not None:
                raise RuntimeError(f"vLLM exited with code {state.process.returncode}")
            try:
                response = await self.client.get(
                    f"{state.spec.worker_api_url}/models",
                    headers=headers,
                    timeout=2,
                )
                if response.is_success:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(self.config.poll_interval)
        raise TimeoutError(f"model {state.spec.id} did not become ready in time")

    async def _wake_worker(self, state: WorkerState) -> None:
        state.status = "WAKING"
        try:
            response = await self.client.post(
                f"{state.spec.worker_root_url}/wake_up",
                headers=self._worker_headers(state.spec),
                timeout=self.config.startup_timeout,
            )
            response.raise_for_status()
            await self._wait_ready(state)
            state.status = "READY"
            state.last_used = time.monotonic()
        except Exception as error:
            await self._stop_worker(state)
            state.status = "FAILED"
            raise ModelUnavailable(f"failed to wake model {state.spec.id}: {error}") from error

    async def _sleep_worker(self, state: WorkerState) -> None:
        response = await self.client.post(
            f"{state.spec.worker_root_url}/sleep?level=1",
            headers=self._worker_headers(state.spec),
            timeout=self.config.request_timeout,
        )
        response.raise_for_status()
        state.status = "SLEEPING"

    async def _drain(self, state: WorkerState) -> None:
        if not state.active_requests:
            return
        state.status = "DRAINING"
        deadline = time.monotonic() + self.config.drain_timeout
        while state.active_requests and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        if state.active_requests:
            raise ModelUnavailable(f"model {state.spec.id} still has active requests")

    async def _stop_worker(self, state: WorkerState) -> None:
        process = state.process
        state.process = None
        if state.log_task:
            state.log_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await state.log_task
            state.log_task = None
        if process:
            if process.returncode is None:
                process.terminate()
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(process.wait(), timeout=10)
            if process.returncode is None:
                process.kill()
                await process.wait()
        state.status = "STOPPED"

    async def _drain_logs(self, state: WorkerState) -> None:
        if not state.process or not state.process.stdout:
            return
        try:
            async for line in state.process.stdout:
                logger.info("[%s] %s", state.spec.id, line.decode(errors="replace").rstrip())
        except asyncio.CancelledError:
            raise

    async def _reaper(self) -> None:
        while not self._closed:
            await asyncio.sleep(self.config.reaper_interval)
            now = time.monotonic()
            for state in self.states.values():
                if state.status != "READY" or state.active_requests or state.spec.mode == "hot":
                    continue
                if now - state.last_used < state.spec.idle_ttl:
                    continue
                try:
                    if state.spec.mode == "warm":
                        await self.sleep(state.spec.id)
                    else:
                        await self.unload(state.spec.id)
                except Exception:
                    logger.exception("failed to reap model %s", state.spec.id)

    @asynccontextmanager
    async def _reserve_gpus(self, gpu_group: tuple[int, ...]) -> AsyncIterator[None]:
        locks = [self._gpu_locks.setdefault(gpu, asyncio.Lock()) for gpu in sorted(set(gpu_group))]
        acquired: list[asyncio.Lock] = []
        try:
            for lock in locks:
                await lock.acquire()
                acquired.append(lock)
            yield
        finally:
            for lock in reversed(acquired):
                lock.release()

    def gpu_group_locked(self, gpu_group: tuple[int, ...]) -> bool:
        return any(self._gpu_locks[gpu].locked() for gpu in gpu_group if gpu in self._gpu_locks)

    @staticmethod
    def _worker_headers(spec: ModelSpec) -> dict[str, str]:
        if spec.worker_api_key:
            return {"Authorization": f"Bearer {spec.worker_api_key}"}
        return {}


def _python_env_library_paths(binary: str) -> list[str]:
    env_root = Path(binary).resolve().parent.parent
    paths: list[str] = []
    for candidate in (env_root / "lib").glob("python*/site-packages/nvidia/**/lib"):
        if candidate.is_dir():
            paths.append(str(candidate))
    return paths
