# 第 6 章：图的构建、编译与可视化

本章收尾核心概念：`compile()` 的完整参数、图校验规则、可视化方法，以及把前面四章串起来的完整示例。

## 6.1 compile() 完整签名

```python
graph = builder.compile(
    checkpointer=None,        # 持久化器（第 11 章）
    store=None,               # 长期记忆存储（第 12 章）
    interrupt_before=None,    # 在指定节点执行前暂停（第 14 章）
    interrupt_after=None,     # 在指定节点执行后暂停
    config_schema=None,       # 声明本图支持的 configurable 字段的 schema
)
```

- `checkpointer`：不传 = 无记忆、不可恢复（纯函数式调用）
- `interrupt_before=["human_review"]`：老式全局打断；新代码推荐节点内 `interrupt()`（第 14 章对比）
- `config_schema`：让图的调用方（如 LangGraph Server）知道有哪些可配置项：

```python
class GraphConfig(TypedDict):
    model_name: str
    user_id: str

graph = builder.compile(config_schema=GraphConfig)
```

## 6.2 构建规则与校验

compile 时会做静态检查，常见的坑：

| 错误 | 原因 |
|---|---|
| `ValueError: Node must be reached from START` | 有节点既没从 START 可达，也没别的边指向它 |
| `ValueError: found unreachble node` 类告警 | 悬空节点（不影响运行但要清理） |
| `InvalidUpdateError`（运行时） | 并行写无 reducer 字段（第 3 章） |
| 节点名非法 | 不要用 `"__start__"`、`"__end__"` 等保留字；避免特殊字符 |
| 修改已编译的图 | compile 后再 add_node 会报错——图不可变，要改就重新 build |

**循环图必须有终止路径**（到 END 的条件、或 recursion_limit 兜底），否则死循环到 25 步默认上限抛 `GraphRecursionError`：

```python
from langgraph.errors import GraphRecursionError

try:
    graph.invoke(inputs, {"recursion_limit": 50})  # 可显式调大
except GraphRecursionError:
    ...
```

## 6.3 可视化

```python
# 1. Mermaid 文本（零依赖，随处可用）
print(graph.get_graph().draw_mermaid())

# 2. PNG 图片（需要能访问 mermaid 渲染服务或本地渲染依赖）
png = graph.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png)

# 3. ASCII 结构图（pip install grandalf）
print(graph.get_graph().draw_ascii())

# 4. 带"数据流向"的结构（xray=True 展示每个节点读/写哪些状态字段）
print(graph.get_graph(xray=True).draw_mermaid())
```

Mermaid 输出示例（可贴到任何支持 mermaid 的 Markdown 里）：

```
flowchart TD
    __start__([__start__]) --> classifier
    classifier -->|refund| refund_agent
    classifier -->|faq| faq_agent
    refund_agent --> __end__([__end__])
    faq_agent --> __end__
```

## 6.4 综合示例：把第 3-6 章串起来

"多轮检索问答 Agent"骨架——包含状态设计、reducer、Command 路由、循环与终止：

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver

llm = init_chat_model("openai:gpt-4o-mini")


class AgentState(MessagesState):
    query: str
    docs: Annotated[list, operator.add]     # 每轮检索结果累加
    rounds: int                             # 检索轮数
    done: bool                              # 是否信息充分


async def search(state: AgentState) -> dict:
    # 伪代码：真实项目换成向量库/搜索 API
    new_docs = [f"doc about: {state['query']} (round {state['rounds']})"]
    return {"docs": new_docs, "rounds": state["rounds"] + 1}


def evaluate(state: AgentState) -> Command:
    """评估信息是否充分：充分 → 生成答案；不充分且 <3 轮 → 继续搜；否则放弃。"""
    if len(state["docs"]) >= 3 or state["rounds"] >= 3:
        return Command(goto="answer", update={"done": True})
    return Command(goto="search")


async def answer(state: AgentState) -> dict:
    context = "\n".join(state["docs"])
    response = await llm.ainvoke([
        {"role": "system", "content": f"根据以下资料回答问题：\n{context}"},
        {"role": "user", "content": state["query"]},
    ])
    return {"messages": [response]}


builder = StateGraph(AgentState)
builder.add_node("search", search)
builder.add_node("evaluate", evaluate)
builder.add_node("answer", answer)

builder.add_edge(START, "search")
builder.add_edge("search", "evaluate")   # search → evaluate → (search|answer) 循环
builder.add_edge("answer", END)

graph = builder.compile(checkpointer=InMemorySaver())

result = graph.invoke(
    {"query": "LangGraph 和 LangChain 的区别", "messages": [], "docs": [], "rounds": 0, "done": False},
    config={"configurable": {"thread_id": "demo-1"}},
)
print(result["messages"][-1].content)
```

流程图（`draw_mermaid()` 的输出）：

```
__start__ → search → evaluate ─┬→ search（循环）
                               └→ answer → __end__
```

## 6.5 运行接口一览

```python
graph.invoke(input, config)                # 同步，返回最终状态
graph.ainvoke(input, config)               # 异步版

graph.stream(input, config, stream_mode="updates")   # 流式（第 10 章）
graph.astream(...)

graph.get_state(config)                    # 查看当前状态
graph.update_state(config, {"done": True}) # 外部改状态（HITL 用）
graph.get_state_history(config)            # 历史快照
```

`invoke` 的输入不一定是完整 State——只需满足 input schema 的必填字段。

## 6.6 模块化组织建议（项目结构）

图变大后，推荐的文件组织：

```
my_agent/
├── state.py        # 所有 Schema 定义（一处集中）
├── nodes/
│   ├── retrieval.py
│   ├── evaluate.py
│   └── answer.py
├── graph.py        # 组装 + compile（唯一 import 全部节点的地方）
└── main.py         # 入口：invoke / serve
```

原则：**Schema 集中定义、节点按领域拆文件、组装只在 graph.py**。

## 本章小结

- `compile(checkpointer, store, interrupt_*, config_schema)` 是所有运行能力的开关
- 图不可变；compile 后不能改；循环必须有出口或 recursion_limit 兜底
- `get_graph().draw_mermaid()` 零依赖可视化，xray=True 显示读写关系
- 核心概念四章（State/Node/Edge+Command/Compile）到此闭环，可以写出任意单 Agent 流程了

> 接下来第三部分：并行、子图、Functional API、流式——从"能用"到"好用"。
