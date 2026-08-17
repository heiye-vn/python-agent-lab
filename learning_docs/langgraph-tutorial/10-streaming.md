# 第 10 章：流式输出 Streaming

流式是把 Agent 接到用户界面的关键能力。LangGraph 提供 6 种 stream mode，可以组合使用——本章逐一拆解，并给出前后端对接方案。

## 10.1 一图看懂六种 stream mode

| mode | 粒度 | 输出内容 | 典型用途 |
|---|---|---|---|
| `values` | 粗 | 每个 superstep 后的**完整状态** | 调试、简单前端 |
| `updates` | 中 | 每个 superstep **哪个节点更新了什么** | 展示"正在做什么" |
| `messages` | 细 | **LLM token 级增量** | 打字机效果（最常用） |
| `custom` | 自定义 | 业务代码主动发的任意事件 | 进度条、中间产物 |
| `events` | 细 | 底层回调事件（链式透传 `astream_events`） | 精细监听（含子链内部） |
| `debug` | 最全 | 任务/状态/事件全量 | 深度调试 |

## 10.2 values：完整状态快照

```python
for chunk in graph.stream(inputs, config, stream_mode="values"):
    print(chunk["messages"][-1])      # 每个 superstep 后的完整状态
# 输出次数 = superstep 数 + 1（含初始输入）
```

## 10.3 updates：节点级增量（推荐默认）

```python
for chunk in graph.stream(inputs, config, stream_mode="updates"):
    for node_name, update in chunk.items():
        print(f"节点 [{node_name}] 完成，更新字段: {list(update.keys())}")
# {'chatbot': {'messages': [AIMessage(...)]}}
# {'tools': {'messages': [ToolMessage(...)]}}
```

注意：返回 None 的节点 `update` 为 None；带 `Command(goto=...)` 的节点会附带 `__interrupt__` 等元信息键。

## 10.4 messages：token 级流式（前端打字机）

```python
for msg, metadata in graph.stream(inputs, config, stream_mode="messages"):
    # msg: AIMessageChunk（含 content 增量、tool_call_chunks）
    # metadata: {"langgraph_node": "agent", "langgraph_step": 3, ...}
    print(msg.content, end="", flush=True)
```

要点：
- `msg.content` 是**增量**，前端直接 append
- 工具调用参数也是流式的（`msg.tool_call_chunks`），可以渲染"正在构造工具调用..."
- `metadata["langgraph_node"]` 告诉你 token 来自哪个节点——多 Agent 场景可显示"XX 助手正在回答"
- token 事件**只在调用方进程产生**；跨进程要用 LangGraph Server 的流式（第 25 章）

## 10.5 custom：业务自定义事件

在任意深处用 `get_stream_writer` 发事件：

```python
from langgraph.config import get_stream_writer

def research_node(state):
    writer = get_stream_writer()
    writer({"type": "progress", "step": 1, "total": 4, "msg": "正在检索资料"})
    docs = search(state["query"])
    writer({"type": "found_docs", "count": len(docs)})
    return {"docs": docs}

# 消费端
for chunk in graph.stream(inputs, config, stream_mode="custom"):
    print(chunk)   # {'type': 'progress', 'step': 1, ...}
```

**技巧：多模式组合**——token 流与自定义进度同时拿：

```python
for chunk in graph.stream(
    inputs, config, stream_mode=["messages", "custom"]
):
    # 组合模式下 chunk 是 (mode_name, payload) 元组
    mode, payload = chunk
    if mode == "messages":
        print(payload[0].content, end="")
    else:
        print("\n[进度]", payload)
```

## 10.6 events / debug

```python
# events：LangChain 的 astream_events 协议，能听到子链内部（如 LCEL 内部）的事件
async for event in graph.astream(inputs, config, stream_mode="events"):
    if event["event"] == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="")

# debug：everything。适合排查"图到底怎么走的"
for chunk in graph.stream(inputs, config, stream_mode="debug"):
    print(chunk)
```

（1.2 版本引入了 stream_events v3 的新实现，API 兼容，性能更好。）

## 10.7 子图流式：subgraphs=True

默认子图输出被折叠。展开：

```python
for chunk in graph.stream(inputs, config, stream_mode="updates", subgraphs=True):
    # chunk = ((namespace, ), {node: update})
    (ns,) = chunk[0] if isinstance(chunk[0], tuple) else (chunk[0],)
    print(f"[{ns or 'parent'}] {chunk[1]}")
```

namespace 标识子图路径（如 `research_agent:subgraph`），多 Agent 应用里用它区分"谁在说话"。

## 10.8 异步流式与超时控制

```python
async for chunk in graph.astream(inputs, config, stream_mode="messages"):
    print(chunk[0].content, end="")

# 带超时保护，防止某个节点卡死拖垮整个连接
import asyncio
async with asyncio.timeout(120):
    async for chunk in graph.astream(...):
        ...
```

## 10.9 对接 Web 前端：SSE 方案

FastAPI + SSE 的最小实现：

```python
# pip install fastapi uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json

app = FastAPI()

@app.get("/chat")
def chat(q: str, thread_id: str):
    def event_stream():
        for chunk in graph.stream(
            {"messages": [("user", q)]},
            {"configurable": {"thread_id": thread_id}},
            stream_mode="messages",
        ):
            msg, meta = chunk
            if msg.content:  # 过滤空 chunk（工具调用块）
                yield f"data: {json.dumps({'text': msg.content, 'node': meta.get('langgraph_node')})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

前端（浏览器）：

```javascript
const es = new EventSource("/chat?q=你好&thread_id=t-1");
es.onmessage = (e) => {
    if (e.data === "[DONE]") return es.close();
    const { text, node } = JSON.parse(e.data);
    appendToChat(text);          // 打字机效果
};
```

> 生产环境更推荐直接用 LangGraph Server 的流式 API（第 25 章），它自带 thread 管理、断线重连（`join`）与鉴权。

## 本章小结

- 六种 mode：values（全量）/ updates（节点级）/ messages（token 级）/ custom（自定义）/ events / debug
- 组合模式返回 `(mode, payload)` 元组；`subgraphs=True` 展开子图
- `get_stream_writer()` 在任意代码深处发自定义事件
- token 流只在调用方进程产生；跨进程流式靠 LangGraph Server

> 至此"进阶控制流"部分完成。接下来是 LangGraph 的企业核心卖点：持久化与记忆。
