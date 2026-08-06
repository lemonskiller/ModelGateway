from pathlib import Path

from app.config import GatewayConfig, ModelSpec


def test_model_spec_builds_vllm_command():
    spec = ModelSpec.from_mapping(
        {
            "id": "qwen3-8b",
            "path": "/models/qwen3",
            "served_model_name": "qwen3-8b",
            "gpu_group": [0, 1],
            "port": 9101,
            "quantization": "awq",
            "max_model_len": 8192,
            "tensor_parallel_size": 2,
            "extra_args": ["--enable-prefix-caching"],
        }
    )

    assert spec.vllm_command("vllm") == [
        "vllm",
        "serve",
        "/models/qwen3",
        "--served-model-name",
        "qwen3-8b",
        "--quantization",
        "awq",
        "--max-model-len",
        "8192",
        "--gpu-memory-utilization",
        "0.82",
        "--tensor-parallel-size",
        "2",
        "--port",
        "9101",
        "--enable-prefix-caching",
    ]


def test_gateway_config_loads_models(tmp_path: Path):
    path = tmp_path / "models.yaml"
    path.write_text(
        """
gateway:
  port: 8123
models:
  - id: one
    path: /models/one
    served_model_name: one
    gpu_group: [0]
    port: 9101
""",
        encoding="utf-8",
    )

    config = GatewayConfig.from_file(path)

    assert config.port == 8123
    assert config.models["one"].gpu_group == (0,)


def test_external_openai_model_uses_base_url():
    spec = ModelSpec.from_mapping(
        {
            "id": "remote",
            "backend": "external_openai",
            "base_url": "http://127.0.0.1:18000",
            "served_model_name": "remote-vllm",
        }
    )

    assert spec.worker_api_url == "http://127.0.0.1:18000/v1"
    assert spec.worker_root_url == "http://127.0.0.1:18000"


def test_warm_model_enables_sleep_mode():
    spec = ModelSpec.from_mapping(
        {
            "id": "warm",
            "path": "/models/warm",
            "served_model_name": "warm",
            "gpu_group": [0],
            "port": 9102,
            "mode": "warm",
        }
    )

    assert "--enable-sleep-mode" in spec.vllm_command()
