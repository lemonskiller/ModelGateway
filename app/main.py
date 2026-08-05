from __future__ import annotations

import asyncio
import contextlib
import hmac
import os
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .config import GatewayConfig
from .manager import GatewayError, ModelManager, Unauthorized


def _config_path() -> str:
    return os.getenv("MODEL_GATEWAY_CONFIG", "config/models.yaml")


def _token_from_request(request: Request) -> str:
    value = request.headers.get("authorization", "")
    return value.removeprefix("Bearer ").strip()


def _authorized(request: Request, expected: str) -> bool:
    return not expected or hmac.compare_digest(_token_from_request(request), expected)


def create_app(config: GatewayConfig | None = None) -> FastAPI:
    loaded_config = config or GatewayConfig.from_file(_config_path())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        manager = ModelManager(loaded_config)
        app.state.manager = manager
        await manager.start()
        try:
            yield
        finally:
            await manager.close()

    app = FastAPI(title="Model Gateway", version="0.1.0", lifespan=lifespan)

    def manager(request: Request) -> ModelManager:
        return request.app.state.manager

    @app.exception_handler(GatewayError)
    async def gateway_error_handler(_: Request, error: GatewayError):
        return JSONResponse(status_code=error.status_code, content={"error": {"message": str(error)}})

    @app.get("/healthz")
    async def healthz(request: Request):
        service = manager(request)
        return {"status": "ok", "models": service.snapshots()}

    @app.get("/readyz")
    async def readyz(request: Request):
        service = manager(request)
        snapshots = service.snapshots()
        hot_failed = [
            item["id"]
            for item in snapshots
            if item["mode"] == "hot" and item["status"] != "READY"
        ]
        if hot_failed:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "hot_models": hot_failed},
            )
        return {"status": "ready"}

    @app.get("/metrics")
    async def metrics():
        return Response(
            content="# Model Gateway metrics are intentionally minimal in v0.1.0.\n",
            media_type="text/plain; version=0.0.4",
        )

    @app.get("/v1/models")
    async def list_models(request: Request):
        if not _authorized(request, loaded_config.api_key):
            return JSONResponse(status_code=401, content={"error": {"message": "invalid API key"}})
        service = manager(request)
        return {
            "object": "list",
            "data": [
                {
                    "id": spec.id,
                    "object": "model",
                    "owned_by": "model-gateway",
                    "root": spec.path,
                    "status": service.states[spec.id].status,
                }
                for spec in loaded_config.models.values()
                if spec.enabled
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        if not _authorized(request, loaded_config.api_key):
            return JSONResponse(status_code=401, content={"error": {"message": "invalid API key"}})
        payload: dict[str, Any] = await request.json()
        model_id = str(payload.get("model", "")).strip()
        if not model_id:
            return JSONResponse(status_code=400, content={"error": {"message": "model is required"}})
        stream = bool(payload.get("stream", False))
        service = manager(request)
        payload["model"] = service.get_spec(model_id).served_model_name
        if stream:
            return await _stream_completion(request, service, model_id, payload)
        async with service.lease(model_id) as spec:
            response = await service.client.post(
                f"{spec.worker_api_url}/chat/completions",
                headers={**service._worker_headers(spec), "content-type": "application/json"},
                json=payload,
                timeout=loaded_config.request_timeout,
            )
            return Response(
                content=response.content,
                status_code=response.status_code,
                media_type=response.headers.get("content-type", "application/json"),
            )

    async def _stream_completion(request: Request, service: ModelManager, model_id: str, payload: dict[str, Any]):
        lease = service.lease(model_id)
        spec = await lease.__aenter__()
        context = service.client.stream(
            "POST",
            f"{spec.worker_api_url}/chat/completions",
            headers={**service._worker_headers(spec), "content-type": "application/json"},
            json=payload,
            timeout=loaded_config.request_timeout,
        )
        try:
            response = await context.__aenter__()
            if response.is_error:
                content = await response.aread()
                await context.__aexit__(None, None, None)
                await lease.__aexit__(None, None, None)
                return Response(
                    content=content,
                    status_code=response.status_code,
                    media_type=response.headers.get("content-type", "application/json"),
                )
        except Exception:
            with contextlib.suppress(Exception):
                await context.__aexit__(None, None, None)
            await lease.__aexit__(None, None, None)
            raise

        async def body():
            try:
                async for chunk in response.aiter_raw():
                    if await request.is_disconnected():
                        break
                    yield chunk
            finally:
                await context.__aexit__(None, None, None)
                await lease.__aexit__(None, None, None)

        return StreamingResponse(body(), status_code=response.status_code, media_type="text/event-stream")

    def require_admin(request: Request) -> None:
        if not _authorized(request, loaded_config.admin_key or loaded_config.api_key):
            raise Unauthorized("invalid admin API key")

    @app.get("/admin/models")
    async def admin_models(request: Request):
        require_admin(request)
        return {"data": manager(request).snapshots()}

    @app.get("/admin/models/{model_id}/status")
    async def admin_model_status(model_id: str, request: Request):
        require_admin(request)
        return manager(request).snapshot(model_id)

    @app.post("/admin/models/{model_id}/load")
    async def admin_load(model_id: str, request: Request):
        require_admin(request)
        await manager(request).ensure_ready(model_id)
        return manager(request).snapshot(model_id)

    @app.post("/admin/models/{model_id}/unload")
    async def admin_unload(model_id: str, request: Request):
        require_admin(request)
        await manager(request).unload(model_id)
        return manager(request).snapshot(model_id)

    @app.post("/admin/models/{model_id}/sleep")
    async def admin_sleep(model_id: str, request: Request):
        require_admin(request)
        await manager(request).sleep(model_id)
        return manager(request).snapshot(model_id)

    @app.post("/admin/models/{model_id}/wake")
    async def admin_wake(model_id: str, request: Request):
        require_admin(request)
        await manager(request).wake(model_id)
        return manager(request).snapshot(model_id)

    return app


app = create_app() if os.path.exists(_config_path()) else FastAPI(title="Model Gateway")
