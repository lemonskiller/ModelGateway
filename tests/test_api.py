from fastapi.testclient import TestClient

from app.config import GatewayConfig, ModelSpec
from app.main import create_app


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
