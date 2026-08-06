from __future__ import annotations

import asyncio
import contextlib
import hmac
import os
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from .config import GatewayConfig
from .manager import GatewayError, ModelManager, Unauthorized


def _config_path() -> str:
    return os.getenv("MODEL_GATEWAY_CONFIG", "config/models.yaml")


def _token_from_request(request: Request) -> str:
    value = request.headers.get("authorization", "")
    return value.removeprefix("Bearer ").strip()


def _authorized(request: Request, expected: str) -> bool:
    return bool(expected) and hmac.compare_digest(_token_from_request(request), expected)


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

    @app.get("/model-gateway", response_class=HTMLResponse)
    async def model_gateway_ui_redirect():
        return Response(status_code=307, headers={"location": "/model-gateway/"})

    @app.get("/model-gateway/", response_class=HTMLResponse)
    async def model_gateway_ui():
        return HTMLResponse(_MODEL_GATEWAY_UI)

    @app.get("/model-gateway/api/models")
    async def ui_models(request: Request):
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

    @app.post("/model-gateway/api/chat")
    async def ui_chat(request: Request):
        payload: dict[str, Any] = await request.json()
        return await _chat_completion_response(request, payload, require_client_key=False)

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
    async def metrics(request: Request):
        service = manager(request)
        return Response(
            content=_render_metrics(service),
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
        payload: dict[str, Any] = await request.json()
        return await _chat_completion_response(request, payload, require_client_key=True)

    async def _chat_completion_response(request: Request, payload: dict[str, Any], require_client_key: bool):
        if require_client_key and not _authorized(request, loaded_config.api_key):
            return JSONResponse(status_code=401, content={"error": {"message": "invalid API key"}})
        model_id = str(payload.get("model", "")).strip()
        if not model_id:
            return JSONResponse(status_code=400, content={"error": {"message": "model is required"}})
        stream = bool(payload.get("stream", False))
        service = manager(request)
        payload["model"] = service.get_spec(model_id).served_model_name
        if stream:
            return await _stream_completion(request, service, model_id, payload)
        async with service.lease(model_id) as spec:
            try:
                response = await service.client.post(
                    f"{spec.worker_api_url}/chat/completions",
                    headers={**service._worker_headers(spec), "content-type": "application/json"},
                    json=payload,
                    timeout=loaded_config.request_timeout,
                )
            except httpx.HTTPError as error:
                return JSONResponse(
                    status_code=502,
                    content={"error": {"message": f"upstream model request failed: {error}"}},
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


_MODEL_GATEWAY_UI = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ModelGateway vLLM Test</title>
  <style>
    :root { color-scheme: light dark; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; background: #f6f7f9; color: #171b21; }
    main { max-width: 980px; margin: 0 auto; padding: 28px 20px; }
    h1 { font-size: 24px; margin: 0 0 16px; }
    .toolbar { display: grid; grid-template-columns: minmax(220px, 1fr) auto auto; gap: 10px; align-items: center; margin-bottom: 14px; }
    select, textarea, button { font: inherit; border: 1px solid #c9ced6; border-radius: 6px; background: white; color: #171b21; }
    select, button { height: 38px; padding: 0 12px; }
    button { cursor: pointer; background: #1f6feb; color: white; border-color: #1f6feb; font-weight: 600; }
    button.secondary { background: white; color: #171b21; border-color: #c9ced6; }
    button:disabled { opacity: .6; cursor: wait; }
    textarea { width: 100%; min-height: 110px; padding: 12px; resize: vertical; box-sizing: border-box; }
    .chat { border: 1px solid #d7dce3; border-radius: 8px; background: white; min-height: 360px; padding: 14px; margin: 14px 0; overflow: auto; }
    .msg { white-space: pre-wrap; line-height: 1.55; margin: 0 0 14px; padding: 10px 12px; border-radius: 7px; }
    .user { background: #eef4ff; }
    .assistant { background: #f4f5f7; }
    .meta { color: #667085; font-size: 13px; margin-bottom: 6px; }
    .status { color: #667085; font-size: 13px; min-height: 20px; }
    @media (prefers-color-scheme: dark) {
      body { background: #111418; color: #e8eaed; }
      select, textarea, button.secondary, .chat { background: #181c22; color: #e8eaed; border-color: #353b45; }
      .user { background: #18263f; }
      .assistant { background: #20242b; }
      .status, .meta { color: #a7adb7; }
    }
  </style>
</head>
<body>
  <main>
    <h1>ModelGateway vLLM Test</h1>
    <div class="toolbar">
      <select id="model"></select>
      <button class="secondary" id="refresh">刷新模型</button>
      <button id="send">发送</button>
    </div>
    <div class="status" id="status"></div>
    <div class="chat" id="chat"></div>
    <textarea id="prompt" placeholder="输入要测试的问题。Ctrl/Cmd + Enter 发送。"></textarea>
  </main>
  <script>
    const modelSelect = document.querySelector('#model');
    const refreshBtn = document.querySelector('#refresh');
    const sendBtn = document.querySelector('#send');
    const promptBox = document.querySelector('#prompt');
    const chat = document.querySelector('#chat');
    const statusEl = document.querySelector('#status');
    const messages = [];
    const setStatus = (text) => { statusEl.textContent = text || ''; };
    const addMessage = (role, content) => {
      const item = document.createElement('div');
      item.className = `msg ${role}`;
      item.innerHTML = `<div class="meta">${role}</div><div></div>`;
      item.lastElementChild.textContent = content;
      chat.appendChild(item);
      chat.scrollTop = chat.scrollHeight;
      return item.lastElementChild;
    };
    async function loadModels() {
      setStatus('加载模型列表...');
      const res = await fetch('api/models');
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || '加载模型失败');
      modelSelect.innerHTML = '';
      for (const model of data.data || []) {
        const opt = document.createElement('option');
        opt.value = model.id;
        opt.textContent = `${model.id} (${model.status})`;
        modelSelect.appendChild(opt);
      }
      setStatus(`已加载 ${modelSelect.options.length} 个 ModelGateway 模型`);
    }
    async function send() {
      const text = promptBox.value.trim();
      if (!text) return;
      const model = modelSelect.value;
      promptBox.value = '';
      messages.push({ role: 'user', content: text });
      addMessage('user', text);
      const target = addMessage('assistant', '');
      sendBtn.disabled = true;
      setStatus(`请求 ${model}；cold 模型首次加载可能需要几分钟...`);
      try {
        const res = await fetch('api/chat', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ model, messages })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error?.message || data.error || JSON.stringify(data));
        const content = data.choices?.[0]?.message?.content || '';
        target.textContent = content || '(空响应)';
        messages.push({ role: 'assistant', content });
        setStatus('完成');
      } catch (err) {
        target.textContent = `错误：${err.message}`;
        setStatus('请求失败');
      } finally {
        sendBtn.disabled = false;
      }
    }
    refreshBtn.addEventListener('click', () => loadModels().catch(err => setStatus(err.message)));
    sendBtn.addEventListener('click', send);
    promptBox.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) send();
    });
    loadModels().catch(err => setStatus(err.message));
  </script>
</body>
</html>'''


def _metric_label(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _labels(values: dict[str, object]) -> str:
    return ",".join(f'{name}="{_metric_label(value)}"' for name, value in values.items())


def _render_metrics(service: ModelManager) -> str:
    statuses = ("STOPPED", "STARTING", "CHECKING", "READY", "SLEEPING", "WAKING", "DRAINING", "FAILED")
    lines = [
        "# HELP model_gateway_up Whether the Model Gateway process is serving metrics.",
        "# TYPE model_gateway_up gauge",
        "model_gateway_up 1",
        "# HELP model_gateway_model_info Static Model Gateway model registration metadata.",
        "# TYPE model_gateway_model_info gauge",
    ]
    for state in service.states.values():
        spec = state.spec
        base_labels = {
            "model": spec.id,
            "served_model_name": spec.served_model_name,
            "backend": spec.backend,
            "mode": spec.mode,
            "gpu_group": ",".join(str(gpu) for gpu in spec.gpu_group),
            "port": spec.port,
            "enabled": int(spec.enabled),
        }
        lines.append(f"model_gateway_model_info{{{_labels(base_labels)}}} 1")

    lines.extend([
        "# HELP model_gateway_model_status Model state by deployment backend and serving mode.",
        "# TYPE model_gateway_model_status gauge",
    ])
    for state in service.states.values():
        spec = state.spec
        for status in statuses:
            lines.append(
                "model_gateway_model_status{"
                + _labels({
                    "model": spec.id,
                    "served_model_name": spec.served_model_name,
                    "backend": spec.backend,
                    "mode": spec.mode,
                    "status": status,
                    "gpu_group": ",".join(str(gpu) for gpu in spec.gpu_group),
                })
                + f"}} {1 if state.status == status else 0}"
            )

    lines.extend([
        "# HELP model_gateway_model_active_requests Active requests currently leased to a model.",
        "# TYPE model_gateway_model_active_requests gauge",
        "# HELP model_gateway_model_pending_requests Requests waiting for a model to become ready.",
        "# TYPE model_gateway_model_pending_requests gauge",
        "# HELP model_gateway_model_last_used_seconds Monotonic timestamp of last model use.",
        "# TYPE model_gateway_model_last_used_seconds gauge",
    ])
    for state in service.states.values():
        spec = state.spec
        dynamic_labels = _labels({
            "model": spec.id,
            "served_model_name": spec.served_model_name,
            "backend": spec.backend,
            "mode": spec.mode,
        })
        lines.append(f"model_gateway_model_active_requests{{{dynamic_labels}}} {state.active_requests}")
        lines.append(f"model_gateway_model_pending_requests{{{dynamic_labels}}} {state.pending_requests}")
        lines.append(f"model_gateway_model_last_used_seconds{{{dynamic_labels}}} {state.last_used}")

    lines.extend([
        "# HELP model_manager_model_state Model manager state by model and deployment backend.",
        "# TYPE model_manager_model_state gauge",
    ])
    manager_states = {
        "STOPPED": "STOPPED",
        "STARTING": "STARTING",
        "CHECKING": "STARTING",
        "READY": "READY",
        "SLEEPING": "SLEEPING",
        "WAKING": "WAKING",
        "DRAINING": "DRAINING",
        "FAILED": "FAILED",
    }
    for state in service.states.values():
        spec = state.spec
        active_state = manager_states.get(state.status, state.status)
        for status in sorted(set(manager_states.values())):
            lines.append(
                "model_manager_model_state{"
                + _labels({
                    "model": spec.id,
                    "state": status,
                    "backend": spec.backend,
                    "mode": spec.mode,
                    "gpu_group": ",".join(str(gpu) for gpu in spec.gpu_group),
                    "worker_id": str(spec.port or spec.base_url or ""),
                    "priority": spec.priority,
                    "alert_email_group": "model-gateway-default",
                })
                + f"}} {1 if active_state == status else 0}"
            )

    lines.extend([
        "# HELP model_manager_queue_size Requests waiting for a model manager lease.",
        "# TYPE model_manager_queue_size gauge",
        "# HELP model_manager_gpu_lock_held Whether a GPU group lock is currently held.",
        "# TYPE model_manager_gpu_lock_held gauge",
        "# HELP model_manager_model_last_used_timestamp_seconds Last model use timestamp as reported by the manager.",
        "# TYPE model_manager_model_last_used_timestamp_seconds gauge",
    ])
    locked_gpu_groups = {
        ",".join(str(gpu) for gpu in gpu_group)
        for gpu_group, lock in service._group_locks.items()
        if lock.locked()
    }
    for state in service.states.values():
        spec = state.spec
        gpu_group = ",".join(str(gpu) for gpu in spec.gpu_group)
        manager_labels = _labels({
            "model": spec.id,
            "backend": spec.backend,
            "mode": spec.mode,
            "gpu_group": gpu_group,
            "alert_email_group": "model-gateway-default",
        })
        lines.append(f"model_manager_queue_size{{{manager_labels}}} {state.pending_requests}")
        lines.append(f"model_manager_gpu_lock_held{{{manager_labels}}} {1 if gpu_group in locked_gpu_groups else 0}")
        lines.append(f"model_manager_model_last_used_timestamp_seconds{{{manager_labels}}} {state.last_used}")

    return "\n".join(lines) + "\n"
