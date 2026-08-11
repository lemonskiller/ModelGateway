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
async def test_conflicting_warm_model_is_slept_before_target_starts(monkeypatch):
    manager = ModelManager(make_config())
    manager.states["hot"].spec = ModelSpec(
        "hot", "/models/hot", "hot", (0,), 9101, mode="warm", priority=100
    )
    manager.states["hot"].status = "READY"
    events: list[str] = []

    async def fake_start(state):
        events.append(f"start:{state.spec.id}")
        state.status = "READY"

    async def fake_sleep(state):
        events.append(f"sleep:{state.spec.id}")
        state.status = "SLEEPING"

    monkeypatch.setattr(manager, "_start_worker", fake_start)
    monkeypatch.setattr(manager, "_sleep_worker", fake_sleep)

    await manager.ensure_ready("cold")

    assert events == ["sleep:hot", "start:cold"]
    assert manager.states["hot"].status == "SLEEPING"
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


@pytest.mark.asyncio
async def test_start_does_not_wait_for_hot_model_load(monkeypatch):
    manager = ModelManager(make_config())
    started = asyncio.Event()
    release = asyncio.Event()

    async def fake_ensure_ready(model_id):
        started.set()
        await release.wait()

    monkeypatch.setattr(manager, "ensure_ready", fake_ensure_ready)

    await asyncio.wait_for(manager.start(), timeout=0.1)
    await asyncio.wait_for(started.wait(), timeout=0.1)

    release.set()
    await manager.close()


@pytest.mark.asyncio
async def test_preloaded_warm_model_is_loaded_then_slept(monkeypatch):
    config = GatewayConfig(
        models={
            "warm": ModelSpec(
                "warm",
                "/models/warm",
                "warm",
                (0,),
                9101,
                mode="warm",
                preload=True,
            )
        }
    )
    manager = ModelManager(config)
    events: list[str] = []

    async def fake_ensure_ready(model_id):
        events.append(f"load:{model_id}")
        manager.states[model_id].status = "READY"

    async def fake_sleep(model_id):
        events.append(f"sleep:{model_id}")
        manager.states[model_id].status = "SLEEPING"

    monkeypatch.setattr(manager, "ensure_ready", fake_ensure_ready)
    monkeypatch.setattr(manager, "sleep", fake_sleep)

    await manager._preload_models()

    assert events == ["load:warm", "sleep:warm"]
    await manager.client.aclose()


@pytest.mark.asyncio
async def test_wake_sleeps_conflicting_warm_model_first(monkeypatch):
    config = GatewayConfig(
        models={
            "one": ModelSpec("one", "/models/one", "one", (0,), 9101, mode="warm"),
            "two": ModelSpec("two", "/models/two", "two", (0,), 9102, mode="warm"),
        }
    )
    manager = ModelManager(config)
    manager.states["one"].status = "READY"
    manager.states["two"].status = "SLEEPING"
    events: list[str] = []

    async def fake_sleep(state):
        events.append(f"sleep:{state.spec.id}")
        state.status = "SLEEPING"

    async def fake_wake(state):
        assert manager.states["one"].status == "SLEEPING"
        events.append(f"wake:{state.spec.id}")
        state.status = "READY"

    monkeypatch.setattr(manager, "_sleep_worker", fake_sleep)
    monkeypatch.setattr(manager, "_wake_worker", fake_wake)

    await manager.ensure_ready("two")

    assert events == ["sleep:one", "wake:two"]
    await manager.client.aclose()


@pytest.mark.asyncio
async def test_sleeping_model_remains_sleeping_when_sleep_is_repeated():
    config = GatewayConfig(
        models={
            "warm": ModelSpec("warm", "/models/warm", "warm", (0,), 9101, mode="warm")
        }
    )
    manager = ModelManager(config)
    manager.states["warm"].status = "SLEEPING"

    await manager.sleep("warm")

    assert manager.states["warm"].status == "SLEEPING"
    await manager.client.aclose()


@pytest.mark.asyncio
async def test_overlapping_gpu_groups_are_serialized(monkeypatch):
    config = GatewayConfig(
        models={
            "one": ModelSpec("one", "/models/one", "one", (0,), 9101),
            "two": ModelSpec("two", "/models/two", "two", (0, 1), 9102),
        }
    )
    manager = ModelManager(config)
    active_starts = 0
    max_active_starts = 0

    async def fake_evict(_spec):
        return None

    async def fake_start(state):
        nonlocal active_starts, max_active_starts
        active_starts += 1
        max_active_starts = max(max_active_starts, active_starts)
        await asyncio.sleep(0.01)
        state.status = "READY"
        active_starts -= 1

    monkeypatch.setattr(manager, "_evict_conflicts", fake_evict)
    monkeypatch.setattr(manager, "_start_worker", fake_start)

    await asyncio.gather(manager.ensure_ready("one"), manager.ensure_ready("two"))

    assert max_active_starts == 1
    await manager.client.aclose()


@pytest.mark.asyncio
async def test_managed_vllm_start_disables_v1_engine(monkeypatch):
    config = GatewayConfig(
        models={
            "warm": ModelSpec("warm", "/models/warm", "warm", (0,), 9101)
        }
    )
    manager = ModelManager(config)
    captured_env: dict[str, str] = {}

    async def fake_create_subprocess_exec(*_args, **kwargs):
        captured_env.update(kwargs["env"])

        class FakeProcess:
            returncode = 0
            stdout = None

            def terminate(self):
                return None

            async def wait(self):
                return 0

            def kill(self):
                return None

        return FakeProcess()

    async def fake_wait_ready(state):
        state.status = "READY"

    monkeypatch.setattr("app.manager.asyncio.create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(manager, "_wait_ready", fake_wait_ready)

    await manager._start_worker(manager.states["warm"])

    assert captured_env["VLLM_USE_V1"] == "0"
    await manager.close()
