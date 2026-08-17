# 第 25 章：API 与 SDK

本章实操 LangGraph Server 的 REST API 与 Python/JS SDK，串起"创建线程 → 发起 run → 流式接收 → HITL resume"的完整生产闭环。

## 25.1 SDK 安装与连接

```bash
pip install langgraph-sdk
```

```python
from langgraph_sdk import get_client

client = get_client(url="http://127.0.0.1:2024")            # 本地 langgraph dev
# 云端：
# from langgraph_sdk import get_cloud_client
# client = get_cloud_client(api_key=os.environ["LANGSMITH_API_KEY"])
```

## 25.2 Threads：会话管理

```python
# 创建
thread = await client.threads.create(
    metadata={"user_id": "u-42", "ticket": "T-9527"},   # 业务元数据（可检索）
)
thread_id = thread["thread_id"]

# 查看/搜索
thread = await client.threads.get(thread_id)
threads = await client.threads.search(
    metadata={"user_id": "u-42"}, limit=20,
)

# 状态操作（等价本地 get_state / update_state）
state = await client.threads.get_state(thread_id)
await client.threads.update_state(thread_id, {"messages": [...]})

# 历史（时间旅行的远程版）
history = await client.threads.get_history(thread_id, limit=10)

# 删除（合规）
await client.threads.delete(thread_id)
```

## 25.3 Runs：发起执行（三种模式）

### 后台 run（长任务标准姿势）

```python
run = await client.runs.create(
    thread_id,
    assistant_id="agent",                # langgraph.json 里的图名，或 assistant 配置
    input={"messages": [{"role": "user", "content": "帮我深度研究……"}]},
    config={"configurable": {"user_id": "u-42"}},
    metadata={"source": "web"},
    multitask_strategy="reject",         # 同 thread 已有在跑 run 时的策略
)
run_id = run["run_id"]

# 随时查状态
run = await client.runs.get(thread_id, run_id)     # status: pending/running/interrupted/success...
```

### 流式 run（聊天 UI 姿势）

```python
async for chunk in client.runs.stream(
    thread_id, "agent",
    input={"messages": [("user", "上海天气？")]},
    stream_mode=["messages-tuple", "updates"],     # 与本地 stream_mode 对齐
):
    print(chunk.event, chunk.data)

# 也可以 join 一个已在跑的 run 的流（断线重连/多端同步）
async for chunk in client.runs.join_stream(thread_id, run_id, stream_mode="messages-tuple"):
    ...
```

### 无状态 run（函数式调用，不落 thread）

```python
result = await client.runs.wait(
    None,                               # thread_id=None → stateless
    "agent",
    input={"messages": [("user", "把这句话翻译成英文：你好")]},
)
```

适合翻译/抽取类高频接口：无持久化开销。

## 25.4 完整闭环：流式 + HITL（生产主流程）

```python
# ── 第一跳：发起并收流，中途遇到 interrupt ──
async for chunk in client.runs.stream(
    thread_id, "agent",
    input={"messages": [("user", "我要退款 3000 元")]},
    stream_mode=["messages-tuple", "updates"],
):
    if chunk.event == "metadata":
        continue
    mode, data = chunk.data if isinstance(chunk.data, list) else (chunk.event, chunk.data)
    if mode == "messages-tuple":
        msg_chunk = data[0]
        print(getattr(msg_chunk, "content", ""), end="")
    elif mode == "updates" and "__interrupt__" in str(data):
        print("\n[需要人工审批]", data)

# ── 查确认暂停状态 ──
state = await client.threads.get_state(thread_id)
# state["next"] 非空 + tasks 里有 interrupts

# ── 第二跳：审批后恢复（新 run 接力）──
async for chunk in client.runs.stream(
    thread_id, "agent",
    command={"resume": {"action": "approve"}},   # ← resume 走 command 参数
    stream_mode="messages-tuple",
):
    ...
```

第 14-15 章的 HITL 时序，在 Server 上就是"两次 stream 调用"。

## 25.5 Assistants 与 Cron

```python
# 创建配置版本（灰度/AB）
assistant = await client.assistants.create(
    graph_id="agent",
    name="agent-v2-试用",
    config={"configurable": {"model": "gpt-4.1", "tone": "casual"}},
)

# 定时任务
cron = await client.crons.create(
    assistant_id="agent",
    schedule="0 8 * * *",
    input={"messages": [("user", "生成昨日日报")]},
)
```

## 25.6 REST API 直调（任意语言）

SDK 只是薄封装，REST 端点稳定可直调（`http://127.0.0.1:2024/docs` 有完整 OpenAPI）：

```bash
# 创建 thread
curl -X POST http://127.0.0.1:2024/threads

# 流式 run（SSE）
curl -N -X POST http://127.0.0.1:2024/threads/{tid}/runs/stream \
  -H "Content-Type: application/json" \
  -d '{"assistant_id":"agent",
       "input":{"messages":[{"role":"user","content":"hi"}]},
       "stream_mode":"messages-tuple"}'

# 恢复 interrupt 的 thread
curl -X POST http://127.0.0.1:2024/threads/{tid}/runs/wait \
  -H "Content-Type: application/json" \
  -d '{"assistant_id":"agent","command":{"resume":{"action":"approve"}}}'
```

JS/TS 侧用 `@langchain/langgraph-sdk`，API 命名与 Python 版一致。

## 25.7 Store 的远程访问

```python
# 长期记忆也走 API（Server 配置的 store，如 Postgres）
await client.store.put_item(("users", "u-42", "preferences"), "food", {"text": "不吃辣"})
items = await client.store.search_items(("users", "u-42", "preferences"), query="饮食")
```

业务系统（不跑图的服务）也能读写同一份记忆——**记忆成为全公司共享资产**。

## 25.8 常用运维端点速查

| 操作 | 端点 |
|---|---|
| 健康检查 | `GET /ok` |
| 列出 runs | `GET /threads/{tid}/runs` |
| 取消 run | `POST /threads/{tid}/runs/{rid}/cancel` |
| run 输出（后台任务） | `GET /threads/{tid}/runs/{rid}` → `output` |
| 图结构 | `GET /info`（assistants 与图元信息） |
| 文档 | `GET /docs`（Swagger UI） |

## 本章小结

- SDK 对象与第 24 章概念一一对应：threads / runs / assistants / crons / store
- Run 三模式：后台（create+join）、流式（stream / join_stream）、无状态（wait + thread=None）
- HITL 闭环 = 两次 stream 调用：input 先行、`command={"resume":...}` 接力
- REST 完整稳定，任意语言可接；Store 远程读写让记忆跨系统共享

> 下一章：真正部署上线——Docker、K8s、认证与自定义端点。
