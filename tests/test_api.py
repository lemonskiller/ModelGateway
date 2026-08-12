import asyncio

from fastapi.testclient import TestClient

from app.config import GatewayConfig, ModelSpec
from app.main import create_app
from app.manager import ModelManager


def make_config() -> GatewayConfig:
    return GatewayConfig(
        api_key="client-key",
        admin_key="admin-key",
        models={
            "qwen3-8b": ModelSpec(
                "qwen3-8b",
                "/models/qwen3",
                "qwen3-8b",
                (0,),
                9101,
                mode="cold",
            )
        },
    )


def test_models_endpoint_requires_client_key():
    with TestClient(create_app(make_config())) as client:
        assert client.get("/v1/models").status_code == 401
        response = client.get(
            "/v1/models",
            headers={"Authorization": "Bearer client-key"},
        )

    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "qwen3-8b"


def test_admin_endpoint_requires_admin_key():
    with TestClient(create_app(make_config())) as client:
        response = client.get(
            "/admin/models",
            headers={"Authorization": "Bearer wrong-key"},
        )

    assert response.status_code == 401


def test_readiness_reports_no_hot_model_for_cold_only_config():
    with TestClient(create_app(make_config())) as client:
        response = client.get("/readyz")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_readiness_waits_for_preloaded_warm_model(monkeypatch):
    async def wait_for_cancellation(_manager):
        await asyncio.Future()

    monkeypatch.setattr(ModelManager, "_preload_models", wait_for_cancellation)
    spec = ModelSpec(
        "warm",
        "/models/warm",
        "warm",
        (0,),
        9101,
        mode="warm",
        preload=True,
    )
    app = create_app(GatewayConfig(models={"warm": spec}))

    with TestClient(app) as client:
        response = client.get("/readyz")
        assert response.status_code == 503
        assert response.json() == {
            "status": "not_ready",
            "preload_models": ["warm"],
        }

        app.state.manager.states["warm"].status = "SLEEPING"
        response = client.get("/readyz")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_model_gateway_ui_serves_relative_api_client():
    with TestClient(create_app(make_config())) as client:
        response = client.get("/model-gateway/")

    assert response.status_code == 200
    assert "ModelGateway vLLM Test" in response.text
    assert "const apiBase =" in response.text
    assert "fetch(`${apiBase}api/models`)" in response.text
    assert "fetch(`${apiBase}api/chat`" in response.text
    assert "readResponseBody(res)" in response.text
    assert "formatError(res, data)" in response.text


def test_model_gateway_ui_models_endpoint_does_not_require_client_key():
    with TestClient(create_app(make_config())) as client:
        response = client.get("/model-gateway/api/models")

    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "qwen3-8b"


def test_empty_client_key_does_not_allow_models_endpoint():
    config = GatewayConfig(
        api_key="",
        admin_key="admin-key",
        models={
            "qwen3-8b": ModelSpec(
                "qwen3-8b",
                "/models/qwen3",
                "qwen3-8b",
                (0,),
                9101,
                mode="cold",
            )
        },
    )

    with TestClient(create_app(config)) as client:
        response = client.get("/v1/models")

    assert response.status_code == 401


def test_metrics_expose_model_status_by_backend_and_mode():
    with TestClient(create_app(make_config())) as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    body = response.text
    assert "model_gateway_up 1" in body
    assert 'model_gateway_model_info{model="qwen3-8b"' in body
    assert 'backend="managed_vllm"' in body
    assert 'mode="cold"' in body
    assert 'status="STOPPED"' in body
    assert "model_gateway_model_active_requests" in body
    assert "model_gateway_model_pending_requests" in body
    assert "model_gateway_model_requests_total" in body
    assert "model_gateway_model_request_errors_total" in body
    assert "model_gateway_model_request_duration_seconds_total" in body
    assert "model_gateway_model_stream_requests_total" in body
    assert "model_gateway_model_input_tokens_total" in body
    assert "model_gateway_model_output_tokens_total" in body
    assert "model_gateway_model_tokens_total" in body
    assert 'model_manager_model_state{model="qwen3-8b",state="STOPPED",backend="managed_vllm"' in body
    assert "model_manager_queue_size" in body
    assert "model_manager_gpu_lock_held" in body


def test_chat_completion_updates_request_metrics_and_logs(caplog):
    config = make_config()
    app = create_app(config)
    with TestClient(app) as client:
        app.state.manager.states["qwen3-8b"].status = "READY"
        with caplog.at_level("INFO", logger="model_gateway"):
            async def fake_post(*_args, **_kwargs):
                class FakeResponse:
                    status_code = 200
                    content = (
                        b'{"choices":[{"message":{"content":"ok"}}],'
                        b'"usage":{"prompt_tokens":5,"completion_tokens":7,"total_tokens":12}}'
                    )
                    headers = {"content-type": "application/json"}

                    def json(self):
                        return {
                            "choices": [{"message": {"content": "ok"}}],
                            "usage": {
                                "prompt_tokens": 5,
                                "completion_tokens": 7,
                                "total_tokens": 12,
                            },
                        }

                return FakeResponse()

            app.state.manager.client.post = fake_post
            response = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer client-key"},
                json={"model": "qwen3-8b", "messages": []},
            )

    assert response.status_code == 200
    body = app.state.manager.snapshot("qwen3-8b")
    assert body["requests_total"] == 1
    assert body["request_errors_total"] == 0
    assert body["request_duration_total"] >= 0
    assert body["request_stream_total"] == 0
    assert body["request_input_tokens_total"] == 5
    assert body["request_output_tokens_total"] == 7
    assert body["request_tokens_total"] == 12
    assert any('"event": "chat_completion"' in record.message for record in caplog.records)
    assert any('"model_id": "qwen3-8b"' in record.message for record in caplog.records)


def test_stream_chat_completion_updates_request_metrics_and_logs(caplog):
    config = make_config()
    app = create_app(config)

    with TestClient(app) as client:
        app.state.manager.states["qwen3-8b"].status = "READY"

        class FakeStreamResponse:
            status_code = 200
            headers = {"content-type": "text/event-stream"}
            is_error = False

            def __init__(self):
                self._chunks = [
                    b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n',
                    b'data: {"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}}\n\n',
                    b"data: [DONE]\n\n",
                ]

            async def aiter_raw(self):
                for chunk in self._chunks:
                    yield chunk

            async def aread(self):
                return b""

        class FakeStreamContext:
            async def __aenter__(self):
                return FakeStreamResponse()

            async def __aexit__(self, *_args):
                return None

        def fake_stream(*_args, **_kwargs):
            return FakeStreamContext()

        with caplog.at_level("INFO", logger="model_gateway"):
            app.state.manager.client.stream = fake_stream
            response = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer client-key"},
                json={"model": "qwen3-8b", "messages": [], "stream": True},
            )

    assert response.status_code == 200
    body = app.state.manager.snapshot("qwen3-8b")
    assert body["requests_total"] == 1
    assert body["request_errors_total"] == 0
    assert body["request_stream_total"] == 1
    assert body["request_input_tokens_total"] == 3
    assert body["request_output_tokens_total"] == 4
    assert body["request_tokens_total"] == 7
    assert any('"event": "chat_completion"' in record.message for record in caplog.records)
    assert any('"stream": true' in record.message for record in caplog.records)
