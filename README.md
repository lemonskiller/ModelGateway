# Model Gateway

独立的内部模型网关，为多个业务服务提供统一的 OpenAI-compatible API，并按需管理 vLLM Worker。

## 架构

```text
业务服务 -> Model Gateway -> vLLM Worker
                         -> 模型加载/休眠/回收
```

业务服务只访问 Gateway，不直接访问 vLLM Worker。Gateway 通过 `model` 逻辑别名路由模型，并使用 GPU 组锁避免并发加载导致 OOM。

## 本地运行

要求：Python 3.12+、uv，以及宿主机上可执行的 `vllm` 命令。

```bash
cd ~/Documents/ModelGateway
cp config/models.yaml.example config/models.yaml
uv sync --dev
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Docker 部署（Gateway 和由它启动的 vLLM Worker 在同一容器）：

```bash
cp config/models.yaml.example config/models.yaml
export MODEL_STORAGE_PATH=/path/to/local/models
export MODEL_GATEWAY_API_KEY="internal-service-key"
export MODEL_GATEWAY_ADMIN_KEY="admin-key"
docker compose up -d --build
```

健康检查：

```text
/healthz  进程存活检查
/readyz   Hot/预热模型就绪检查；未完成预热时返回 503
```

如果使用环境变量配置密钥：

```bash
export MODEL_GATEWAY_API_KEY="internal-service-key"
export MODEL_GATEWAY_ADMIN_KEY="admin-key"
export MODEL_GATEWAY_CONFIG="$PWD/config/models.yaml"
```

## 调用示例

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Authorization: Bearer internal-service-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-8b",
    "messages": [{"role": "user", "content": "你好"}],
    "stream": true
  }'
```

管理接口使用 `MODEL_GATEWAY_ADMIN_KEY`：

```bash
curl -X POST http://127.0.0.1:8000/admin/models/qwen3-8b/load \
  -H 'Authorization: Bearer admin-key'
```

## 加速首次加载和模型切换

对需要频繁切换的模型使用以下配置：

```yaml
mode: warm
preload: true
safetensors_load_strategy: eager
```

- `preload: true`：Gateway 启动后在后台完成首次加载；`/readyz` 会在预热完成前返回 503。
- `mode: warm`：模型被切换出去时使用 vLLM Sleep Mode level 1，将权重保留在 CPU 内存；再次使用时从 CPU 恢复到 GPU，避免重新启动进程和读取权重文件。
- `safetensors_load_strategy: eager`：适合当前 `/nfs` 模型目录，避免网络文件系统上的大量随机读取。它会在加载时增加 CPU 内存峰值。

共享 GPU 的 warm 模型会依次预热，使用不同 GPU 的模型可以并行预热。每个 warm 模型需要足够的 CPU 内存保存一份休眠权重；如果内存不足，只给常用模型配置 `preload: true`，其余模型保留 `mode: cold`。

## 接入 NextOffer

```bash
VLLM_MODEL_BASE_URL="http://model-gateway:8000/v1"
VLLM_MODEL_API_KEY="internal-service-key"
VLLM_MODEL_NAME="qwen3-8b"
```

Docker Compose 或同一私网中的其他服务使用 `model-gateway` 作为 DNS 服务名；宿主机直接运行时使用 `127.0.0.1`、私网 IP 或内部 DNS。

如果 NextOffer 和 Model Gateway 使用不同的 Compose 文件，两个服务需要加入同一个 Docker 网络：

```yaml
networks:
  model-network:
    external: true
    name: model-network
```

## 当前实现范围

- OpenAI-compatible `/v1/chat/completions`
- `/v1/models`、`/healthz`、`/metrics`
- 模型别名注册
- 按 GPU 组串行加载
- Worker 启动、健康检查、停止
- Hot/Warm/Cold 模式基础支持
- Sleep Mode 唤醒和回收
- 请求队列由 Gateway 的模型就绪流程统一控制

下一步可增加 Prometheus 指标、服务级限流、Redis/数据库注册表和多节点调度。
