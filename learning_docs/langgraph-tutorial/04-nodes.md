# 第 4 章：节点 Node

节点是图中"真正干活"的地方：调 LLM、执行工具、跑业务逻辑。本章覆盖从基础写法到企业级节点的全部知识。

## 4.1 节点的四种形态

```python
from langchain_core.runnables import Runnable

# 1. 普通函数（最常用）
def node_fn(state: State) -> dict: ...

# 2. 异步函数（生产推荐：IO 密集型节点不阻塞事件循环）
async def node_async(state: State) -> dict:
    result = await llm.ainvoke(state["messages"])
    return {"messages": [result]}

# 3. Runnable（LCEL 表达式可直接当节点）
llm_node = llm | (lambda msg: {"messages": [msg]})
builder.add_node("llm", llm_node)  # 输出会被转成 {"messages": [...]}

# 4. 类实例（需要持有配置/依赖注入时）
class MyNode:
    def __init__(self, db_pool):
        self.db = db_pool
    def __call__(self, state: State) -> dict:
        ...
builder.add_node("persist", MyNode(pool))
```

## 4.2 输入输出约定

节点接收 state（完整或按注解裁剪，见 3.4），**返回 dict，只写要改的字段**：

```python
def node(state: State) -> dict:
    return {"answer": "42"}      # 只更新 answer，其他字段原样

def no_change(state):            # 返回 None / 空 dict = 不更新任何字段
    return None
```

注意：**不要原地修改 state 再返回它**（如 `state["x"] = 1; return state`）。虽然多数情况能跑，但会绕过 reducer、破坏持久化快照的一致性。始终返回新 dict。

## 4.3 第二个参数：RunnableConfig

节点可以声明第二个参数 `config`，拿到运行时元信息：

```python
from langchain_core.runnables import RunnableConfig

def node(state: State, config: RunnableConfig) -> dict:
    configurable = config["configurable"]
    thread_id = configurable["thread_id"]       # 当前会话 ID
    user_id = configurable.get("user_id")       # 你自定义透传的参数
    run_id = config.get("run_id")               # 本次 run 的 UUID
    callbacks = config.get("callbacks")         # 回调（tracing 用）
    ...
```

自定义参数在 invoke 时传入：

```python
graph.invoke(
    {"question": "..."},
    config={"configurable": {"thread_id": "t-1", "user_id": "u-42"}},
)
```

这是企业应用的标准做法：把"谁在用、什么租户、什么环境"通过 configurable 传进图里，而不是硬编码。

## 4.4 获取图上下文的三个辅助函数

LangGraph 1.x 提供 `langgraph.config` 命名空间，在**任意深处**的代码（不只是节点函数签名里）都能取到上下文：

```python
from langgraph.config import get_config, get_store, get_stream_writer

def node(state):
    # 当前 run 的完整 config（含 configurable）
    config = get_config()
    thread_id = config["configurable"]["thread_id"]

    # 长期记忆 store（compile 时传入了 store 才有）
    store = get_store()

    # 自定义流式事件发送器（第 10 章）
    writer = get_stream_writer()
    writer({"progress": "50%"})
```

典型用途：在**工具函数内部**（无法从节点传 config 下去的场景）拿 thread_id / store / writer。

## 4.5 重试策略 RetryPolicy

对 LLM 调用等不稳定节点配置指数退避重试：

```python
from langgraph.pregel import RetryPolicy

builder.add_node(
    "llm_call",
    llm_node,
    retry=RetryPolicy(
        initial_interval=1.0,     # 首次重试等 1s
        max_attempts=3,           # 最多试 3 次
        exponential_base=2.0,     # 退避倍数：1s, 2s, 4s...
        jitter=True,
        retry_on=ValueError,      # 哪些异常才重试（默认宽泛）
    ),
)
```

默认会对多数瞬时异常（超时、限流类）重试；对确定性错误（如代码 bug 的 KeyError）应排除以免浪费调用。

## 4.6 节点缓存 CachePolicy

幂等节点（同样输入必同样输出）可缓存结果，省钱省时：

```python
from langgraph.pregel import CachePolicy

builder.add_node(
    "summarize",
    summarize_fn,
    cache_policy=CachePolicy(ttl=3600),  # 缓存 1 小时，键默认由节点输入计算
)
```

注意：只有真正幂等的节点才能开（例如纯函数式的 LLM 摘要）。带副作用（写库、发消息）的节点禁用。

## 4.7 节点返回 Command：把"干活"和"路由"合为一体

节点可以不返回 state dict，而是返回 `Command`，同时完成状态更新和下一步跳转（第 5 章详解）：

```python
from langgraph.types import Command

def router(state) -> Command:
    if state["needs_review"]:
        return Command(goto="human_review", update={"flagged": True})
    return Command(goto="finalize", update={"flagged": False})
```

## 4.8 superstep 语义：并行的规则复习

同一轮里被同时触发的节点**并发执行**，全部完成后状态合并，进入下一轮。推论：

1. 并行节点之间**不要有隐式的执行顺序假设**
2. 并行写同一字段必须配 Reducer（第 3 章）
3. 一个节点"依赖另一个节点的输出"= 必须用边串行化（A → B），而不是指望同轮顺序

```
     ┌──> node_b ──┐
START ──> node_a ──┼──> join_node     # a、b 同轮并行；join_node 在下一轮执行，
     └──> node_c ──┘                  # 此时已能看到 a、b、c 三者的写入
```

## 4.9 企业级节点编写清单

一个健壮的生产节点长这样：

```python
import logging
from tenacity import stop_after_attempt  # 或用内置 RetryPolicy

logger = logging.getLogger(__name__)

async def retrieve_node(state: AgentState, config: RunnableConfig) -> dict:
    # 1. 结构化日志：带上下文，方便 LangSmith 关联
    logger.info("retrieve start", extra={
        "thread_id": config["configurable"].get("thread_id"),
        "query_len": len(state["query"]),
    })
    try:
        # 2. 显式超时，防止拖死整张图
        docs = await asyncio.wait_for(search(state["query"]), timeout=10)
    except TimeoutError:
        # 3. 降级而非崩溃：给下游一个可识别的信号
        return {"retrieved_docs": [], "retrieval_failed": True}

    # 4. 只返回增量，绝不动整份 state
    return {"retrieved_docs": docs}
```

要点：日志带 trace 上下文、显式超时、降级路径、partial update。

## 本章小结

- 节点 = 函数/async 函数/Runnable/可调用类，返回 partial update
- `RunnableConfig` + `langgraph.config.get_*` 是获取运行时上下文的标准途径
- `add_node` 可挂 `retry`（RetryPolicy）与 `cache_policy`（CachePolicy）
- superstep 内并行、轮间串行——设计依赖时先想清楚谁和谁在同一轮

> 下一章：边与 Command——LangGraph 的两种控制流范式。
