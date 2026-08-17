# 第 7 章：循环、分支与并行（Send API）

分支与基础循环在第 5 章已讲。本章聚焦**并行**：静态并行（fan-out/fan-in）与动态并行（Send API / map-reduce）——这是 LangGraph 区别于普通状态机框架的招牌能力。

## 7.1 静态并行：fan-out / fan-in

同时加多条边即并行：

```python
builder.add_edge("retrieve", "web_search")
builder.add_edge("retrieve", "db_search")
builder.add_edge("web_search", "merge")
builder.add_edge("db_search", "merge")
```

执行时序：

```
superstep 1: retrieve
superstep 2: web_search ∥ db_search     ← 同轮并发
superstep 3: merge（此时两路结果都已在状态里）
```

**merge 节点汇聚的前提**：两路写的是带 reducer 的不同/相同字段。最典型：

```python
class State(TypedDict):
    query: str
    results: Annotated[list, operator.add]   # 两路检索都往里追加

def web_search(state): return {"results": ["来自网络的结果"]}
def db_search(state):  return {"results": ["来自数据库的结果"]}
def merge(state):      # results == [网络, 数据库]（顺序不定！）
```

牢记：**并行写入的顺序不确定**，merge 节点不要依赖列表顺序；需要顺序就按内容打标再排序。

## 7.2 动态并行：Send API（map-reduce 标准解法）

静态并行的问题是**分支数量在画图时就固定了**。要"对 N 个输入各起一个处理单元"（N 运行时才知道），用 `Send`：

```python
from langgraph.types import Send
import operator

class State(TypedDict):
    topics: list                 # 待处理主题（运行时决定数量）
    summaries: Annotated[list, operator.add]   # 汇聚结果


def fan_out(state: State) -> list[Send]:
    # 为每个 topic 动态生成一个 "worker 实例"，各自携带自己的输入状态
    return [Send("worker", {"topic": t}) for t in state["topics"]]


def worker(state: dict) -> dict:      # 注意：worker 收到的是 Send 携带的独立输入
    summary = llm.invoke(f"总结这个主题：{state['topic']}").content
    return {"summaries": [summary]}


def final_report(state: State) -> dict:
    return {"report": "\n".join(state["summaries"])}


builder = StateGraph(State)
builder.add_node("worker", worker)
builder.add_node("final_report", final_report)
builder.add_conditional_edges("__start__", fan_out, ["worker"])  # 入口即分发
builder.add_edge("worker", "final_report")
builder.add_edge("final_report", END)
graph = builder.compile()

graph.invoke({"topics": ["AI", "量子计算", "航天"]})
# 3 个 worker 并发执行，结果自动按 operator.add 汇聚
```

**Send 的三个关键点**：

1. `Send("worker", input)` 中 `input` 是**该实例独享的输入状态**（不必是全图 State 类型）
2. 所有被 Send 触发的 worker 在**同一个 superstep 并发**，全部完成后才进入下一步
3. worker 返回的更新走**全图的 reducer**（如 `summaries` 的 `operator.add`），自动汇聚

用 Command 也能发 Send（等价写法，节点内）：

```python
def start(state) -> Command:
    return Command(goto=[Send("worker", {"topic": t}) for t in state["topics"]])
```

### 经典应用：对长文档分块并行摘要

```python
chunks = split(text, size=2000)
# fan_out: [Send("summarize_chunk", {"chunk": c}) for c in chunks]
# reduce: operator.add 汇总各段摘要 → final_report 生成总摘要
```

比串行处理快 N 倍，且逻辑集中、可视化清晰。

## 7.3 两种并行对比

| | 静态 fan-out | Send 动态并行 |
|---|---|---|
| 分支数 | 画图时固定 | 运行时任意 |
| 各分支输入 | 共享全图 State | 每实例独立输入 |
| 典型场景 | 固定多路检索（web+db） | map-reduce、批量任务 |
| 写法 | 多条 add_edge | 路由函数返回 Send 列表 |

## 7.4 循环的工程控制

Agent 循环（模型⇄工具）必须有刹车，三层保险：

```python
# 1. 逻辑层：条件里显式判断步数/轮数
def should_continue(state):
    if state["rounds"] >= 5:
        return "force_final"
    ...

# 2. 引擎层：recursion_limit 兜底（默认 25 个 superstep）
graph.invoke(inputs, {"recursion_limit": 30})

# 3. 代码层：捕获 GraphRecursionError 优雅降级
from langgraph.errors import GraphRecursionError
try:
    result = graph.invoke(inputs)
except GraphRecursionError:
    result = {"answer": "抱歉，任务过于复杂，已达到最大步数"}
```

`recursion_limit` 数的是 **superstep 数**不是 token 数；LLM 一次回复里多个工具调用仍算一个 superstep。

## 7.5 条件分支的健壮性写法

路由函数返回未知节点名是运行时错误。健壮写法：

```python
VALID_NEXT = {"refund_agent", "faq_agent", "tech_support"}

def route(state) -> str:
    next_node = classify(state)          # LLM 或规则判断
    if next_node not in VALID_NEXT:
        return "faq_agent"               # 白名单校验 + 默认兜底
    return next_node
```

如果路由本身由 LLM 决定，务必用结构化输出（见第 19 章）拿到可枚举字段，而不是解析自由文本。

## 本章小结

- 静态并行：多条出边同轮并发，汇聚靠 reducer，顺序不可依赖
- `Send` = 运行时动态 map-reduce：每实例独立输入，同轮并发，reducer 自动汇聚
- 循环三层刹车：逻辑判断、recursion_limit、异常捕获降级
- 路由函数返回值做白名单校验 + 默认路由

> 下一章：子图——把图当节点用，构建可复用组件与多 Agent 系统。
