from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
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


class ModelManager:
    def __init__(self, config: GatewayConfig, client: httpx.AsyncClient | None = None) -> None:
        self.config = config
        # Worker traffic is private/local and must not inherit shell HTTP(S)_PROXY settings.
        self.client = client or httpx.AsyncClient(trust_env=False)
        self.states = {model_id: WorkerState(spec) for model_id, spec in config.models.items()}
        self._group_locks: dict[tuple[int, ...], asyncio.Lock] = {}
        self._reaper_task: asyncio.Task[None] | None = None
        self._closed = False

    async def start(self) -> None:
        self._reaper_task = asyncio.create_task(self._reaper(), name="model-gateway-reaper")
        hot_models = [
            state.spec.id
            for state in self.states.values()
            if state.spec.enabled and state.spec.mode == "hot"
        ]
        if hot_models:
            results = await asyncio.gather(
                *(self.ensure_ready(model_id) for model_id in hot_models),
                return_exceptions=True,
            )
            for model_id, result in zip(hot_models, results):
                if isinstance(result, Exception):
                    logger.error("hot model %s failed to start: %s", model_id, result)

    async def close(self) -> None:
        self._closed = True
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
            "gpu_group": list(state.spec.gpu_group),
            "mode": state.spec.mode,
            "priority": state.spec.priority,
            "status": state.status,
            "active_requests": state.active_requests,
            "pending_requests": state.pending_requests,
            "last_used": state.last_used,
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

        lock = self._group_locks.setdefault(spec.gpu_group, asyncio.Lock())
        async with lock:
            if state.status == "READY":
                state.last_used = time.monotonic()
                return
            if state.status == "SLEEPING":
                await self._wake_worker(state)
                return
            await self._evict_conflicts(spec)
            await self._start_worker(state)

    async def unload(self, model_id: str) -> None:
        spec = self.get_spec(model_id)
        lock = self._group_locks.setdefault(spec.gpu_group, asyncio.Lock())
        async with lock:
            state = self.states[model_id]
            await self._drain(state)
            await self._stop_worker(state)

    async def sleep(self, model_id: str) -> None:
        spec = self.get_spec(model_id)
        lock = self._group_locks.setdefault(spec.gpu_group, asyncio.Lock())
        async with lock:
            state = self.states[model_id]
            await self._drain(state)
            if state.status == "READY":
                await self._sleep_worker(state)

    async def wake(self, model_id: str) -> None:
        await self.ensure_ready(model_id)

    async def _evict_conflicts(self, target: ModelSpec) -> None:
        target_gpus = set(target.gpu_group)
        conflicts = [
            state
            for state in self.states.values()
            if state.spec.id != target.id
            and state.status in {"READY", "SLEEPING"}
            and target_gpus.intersection(state.spec.gpu_group)
        ]
        conflicts.sort(key=lambda item: (item.spec.mode == "hot", item.spec.priority, item.last_used))
        for state in conflicts:
            if state.spec.mode == "hot" and target.priority <= state.spec.priority:
                raise ModelUnavailable(
                    f"GPU group {target.gpu_group} is occupied by hot model {state.spec.id}"
                )
            await self._drain(state)
            await self._stop_worker(state)

    async def _start_worker(self, state: WorkerState) -> None:
        state.status = "STARTING"
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in state.spec.gpu_group)
        if state.spec.mode == "warm":
            environment["VLLM_SERVER_DEV_MODE"] = "1"

        try:
            state.process = await asyncio.create_subprocess_exec(
                *state.spec.vllm_command(self.config.vllm_binary),
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

    @staticmethod
    def _worker_headers(spec: ModelSpec) -> dict[str, str]:
        if spec.worker_api_key:
            return {"Authorization": f"Bearer {spec.worker_api_key}"}
        return {}
