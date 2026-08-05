import asyncio

import pytest

from app.config import GatewayConfig, ModelSpec
from app.manager import ModelManager, ModelUnavailable, RequestQueueFull


def make_config() -> GatewayConfig:
    return GatewayConfig(
        startup_timeout=1,
        poll_interval=0.01,
        models={
            "hot": ModelSpec("hot", "/models/hot", "hot", (0,), 9101, mode="hot", priority=100),
            "cold": ModelSpec("cold", "/models/cold", "cold", (0,), 9102, mode="cold", priority=50),
        },
    )


@pytest.mark.asyncio
async def test_lower_priority_model_cannot_evict_hot_model():
    manager = ModelManager(make_config())
    manager.states["hot"].status = "READY"

    with pytest.raises(ModelUnavailable):
        await manager.ensure_ready("cold")

    await manager.client.aclose()


@pytest.mark.asyncio
async def test_conflicting_model_is_stopped_before_target_starts(monkeypatch):
    manager = ModelManager(make_config())
    manager.states["hot"].spec = ModelSpec(
        "hot", "/models/hot", "hot", (0,), 9101, mode="warm", priority=100
    )
    manager.states["hot"].status = "READY"
    started: list[str] = []

    async def fake_start(state):
        started.append(state.spec.id)
        state.status = "READY"

    async def fake_stop(state):
        state.status = "STOPPED"

    monkeypatch.setattr(manager, "_start_worker", fake_start)
    monkeypatch.setattr(manager, "_stop_worker", fake_stop)

    await manager.ensure_ready("cold")

    assert started == ["cold"]
    assert manager.states["hot"].status == "STOPPED"
    await manager.client.aclose()


@pytest.mark.asyncio
async def test_model_queue_rejects_when_full():
    config = make_config()
    config = config.__class__(
        models=config.models,
        max_queue_size_per_model=0,
    )
    manager = ModelManager(config)

    with pytest.raises(RequestQueueFull):
        async with manager.lease("cold"):
            pass

    await manager.client.aclose()
