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
