# 第 9 章：Functional API（@entrypoint / @task）

LangGraph 有两套等价的编程接口：**Graph API**（StateGraph，显式画图）和 **Functional API**（`@entrypoint` + `@task`，写普通函数）。Functional API 在 1.x 逐渐成熟，特别适合**流程动态、不想被图结构束缚**的场景。

## 9.1 动机：图不是万能的表达方式

Graph API 的前提是"流程结构**编译期可知**"。但有些工作流：

- 步骤由数据决定（处理文件树、遍历未知深度的依赖）
- 有复杂的业务 if/else，画成图是一团面条
- 想直接复用现有 Python 函数，不想改成节点协议

Functional API 让你**写普通 Python，自动获得持久化、恢复、HITL**。

## 9.2 基本用法

```python
from langgraph.func import entrypoint, task


@task
def fetch_data(url: str) -> dict:
    """一个可持久化的工作单元"""
    return requests.get(url).json()


@task
def summarize(data: dict) -> str:
    return llm.invoke(f"总结：{data}").content


@entrypoint(checkpointer=InMemorySaver())   # 同样可以配 checkpointer / store
def workflow(inputs: dict) -> dict:
    data = fetch_data(inputs["url"]).result()   # .result() 取 task 结果
    summary = summarize({"text": data}).result()
    return {"data": data, "summary": summary}


workflow.invoke({"url": "https://example.com/api"},
                config={"configurable": {"thread_id": "w-1"}})
```

核心概念：

| 概念 | 说明 |
|---|---|
| `@task` | 工作单元。结果被持久化；重跑时若已完成则直接返回缓存结果（不重复执行） |
| `@entrypoint` | 入口。相当于图的 START+END + 顶层编排逻辑 |
| `.result()` / `await task()` | 获取 task 结果（同步/异步） |
| checkpointer | 挂在 entrypoint 上，按 thread 持久化 |

**关键语义**：task 完成即存档。如果进程在第 3 个 task 后崩溃，恢复执行时前 2 个 task **直接读存档，不会重跑**——这就是 durable execution。副作用（写库、发邮件）因此不会被重复执行。

## 9.3 entrypoint 的 previous 状态

`@entrypoint` 支持跨 invoke 保存"入口级状态"（类似图的 State）：

```python
from langgraph.func import entrypoint, get_entrypoint_state  # 或用 previous 参数

@entrypoint(checkpointer=InMemorySaver())
def chat(inputs: list, *, previous: list | None = None) -> list:
    # previous = 上一次同一 thread 调用结束时的返回值
    messages = previous or []
    messages = messages + inputs

    @task
    def call_model(msgs):
        return llm.invoke(msgs)

    response = call_model(messages).result()
    messages = messages + [response]
    return messages                     # 下次成为 previous
```

适合"消息列表就是全部状态"的简单对话场景，几行代码获得记忆。

## 9.4 task 之间的并发

```python
@entrypoint(checkpointer=InMemorySaver())
def parallel_workflow(inputs):
    # 同时发起多个 task
    t1 = fetch_data("https://a.com")
    t2 = fetch_data("https://b.com")
    # 再统一取结果 —— 两个请求已并发执行
    return {"a": t1.result(), "b": t2.result()}
```

异步版用 `asyncio.gather` 风格：

```python
@entrypoint(checkpointer=InMemorySaver())
async def workflow(inputs):
    r1, r2 = await asyncio.gather(fetch_data("a"), fetch_data("b"))
    return [r1, r2]
```

## 9.5 HITL 与流式：Functional API 全支持

```python
from langgraph.types import interrupt, Command
from langgraph.config import get_stream_writer

@entrypoint(checkpointer=InMemorySaver())
def approval_flow(inputs):
    draft = summarize(inputs).result()

    writer = get_stream_writer()
    writer({"draft": draft})                    # 自定义流式事件

    decision = interrupt({"question": "发布吗？", "draft": draft})  # 暂停等人
    if decision == "approve":
        publish(draft).result()
        return "published"
    return "rejected"

# 恢复（与图完全一致的用法）
approval_flow.invoke(Command(resume="approve"),
                     config={"configurable": {"thread_id": "a-1"}})
```

## 9.6 Graph 与 Functional 混用

二者可以互相嵌套：

```python
# 图作为 task
@task
def run_rag(question: str) -> str:
    return rag_graph.invoke({"question": question})["answer"]

# Functional 作为图的节点
builder.add_node("workflow_step", my_entrypoint_workflow)
```

实践中常见架构：**主流程用 Graph（结构清晰可可视化），局部动态逻辑用 Functional**。

## 9.7 选型指南：Graph vs Functional

| 维度 | Graph API | Functional API |
|---|---|---|
| 流程结构 | 编译期固定，可视化 | 运行时任意 |
| 可视化 | ✅ mermaid / Studio | ❌ 无图可画 |
| 团队协作/评审 | 结构即文档 | 读代码 |
| 循环控制、Send 并行 | 原生支持 | 普通 Python 循环/gather |
| 动态子流程、数据驱动 | 别扭 | 丝滑 |
| LangSmith 调试 | span 即节点 | span 即 task |

**建议**：Agent / 多 Agent / 需要给业务方看流程图 → Graph；数据处理 pipeline、动态工作流、快速改造存量 Python 代码 → Functional。

## 本章小结

- Functional API = 普通 Python 函数 + 持久化运行时（`@task` 存档、崩溃后跳过已完成的）
- `@entrypoint` 的 `previous` 参数实现入口级记忆
- 并发、interrupt、流式与 Graph API 能力对等
- 两套 API 可互相嵌套；按"结构是否需要可视化/固定"选型

> 下一章：流式输出——把 LangGraph 接到用户界面的关键一课。
