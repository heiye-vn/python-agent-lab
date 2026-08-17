# 第 3 章：状态 State —— LangGraph 的心脏

State 是 LangGraph 中唯一贯穿全图的数据结构。所有节点都从它读、向它写；持久化存的是它；时间旅行回放的是它。**搞懂 State 的读写规则 = 搞懂 LangGraph 的 80%。**

## 3.1 三种定义方式

### 方式一：TypedDict（最常用、零依赖）

```python
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]  # 追加式
    counter: int                             # 默认覆盖式
```

### 方式二：Pydantic（需要校验时用）

```python
from pydantic import BaseModel, Field

class State(BaseModel):
    messages: Annotated[list, add_messages]
    query: str = Field(description="用户原始问题")
    score: float = Field(default=0.0, ge=0, le=1)  # 自动校验范围
```

优势：类型/范围校验、可嵌套模型、配合 `response_format` 做结构化输出。
劣势：比 TypedDict 略慢；节点返回非法值会直接抛 ValidationError（生产中这是优点）。

### 方式三：dataclass（轻量、IDE 友好）

```python
from dataclasses import dataclass, field

@dataclass
class State:
    messages: Annotated[list, add_messages] = field(default_factory=list)
    query: str = ""
```

**选型建议**：默认 TypedDict；需要严格校验或字段很多时用 Pydantic。

## 3.2 内置 MessagesState

聊天场景 90% 的状态都长这样，LangGraph 直接内置了：

```python
from langgraph.graph import MessagesState

class State(MessagesState):
    # MessagesState 已包含：messages: Annotated[list[AnyMessage], add_messages]
    # 你只需追加自己的字段
    user_name: str
```

等价于：

```python
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_name: str
```

## 3.3 Reducer：状态更新的合并规则（核心中的核心）

**关键认知：节点返回的 dict 不是"替换整个 State"，而是对每个字段做"合并"。合并规则由 Reducer 决定。**

### 规则一：默认 = 覆盖（LastValue）

```python
class State(TypedDict):
    answer: str

def node_a(state): return {"answer": "第一次生成"}
def node_b(state): return {"answer": "第二次生成，覆盖了 a 的结果"}
```

如果两个节点（同一轮并行）同时写 `answer`，会抛
`InvalidUpdateError: Invalid concurrent update to same channel/action`——**没有 reducer 的字段不允许并发写**。

### 规则二：add_messages —— 消息追加 + 按 ID 去重

```python
from langgraph.graph.message import add_messages
```

它的行为：
1. 新消息 **append** 到列表尾部
2. 若新消息的 `id` 与已有消息相同 → **替换**那条（用于人工编辑消息）
3. 配合 `RemoveMessage` → **删除**指定消息（对话裁剪时用）

```python
from langchain_core.messages import AIMessage, RemoveMessage

def node(state):
    return {
        "messages": [
            RemoveMessage(id=state["messages"][0].id),  # 删掉第一条
            AIMessage(content="新的回复", id="new-1"),   # 追加
        ]
    }
```

### 规则三：operator.add —— 列表拼接

```python
import operator
from typing import Annotated

class State(TypedDict):
    docs: Annotated[list, operator.add]   # new_list 会被 extend 进去

def node_a(state): return {"docs": ["doc1"]}
def node_b(state): return {"docs": ["doc2"]}
# 并行执行后 state["docs"] == ["doc1", "doc2"]
```

这就是 map-reduce 并行汇聚的标准解法（第 7 章结合 Send API 使用）。

### 规则四：自定义 Reducer

任何 `f(existing, new) -> merged` 形态的函数都行：

```python
# 签名：f(当前值, 新值) -> 合并后的值
def keep_max(current: int, new: int) -> int:
    return max(current, new)

class State(TypedDict):
    best_score: Annotated[int, keep_max]
```

**Reducer 必须满足交换律（commutative）**——因为并行节点的写入顺序不确定。

## 3.4 输入 / 输出 / 私有状态：三个 Schema 的分工

生产级图通常把"用户可见"与"内部流转"的字段分开：

```python
from typing_extensions import TypedDict

class InputState(TypedDict):
    """用户输入 schema：invoke 时只需要提供这些字段"""
    question: str

class OutputState(TypedDict):
    """输出 schema：invoke 返回值只包含这些字段"""
    answer: str

class InternalState(InputState, OutputState):
    """完整内部状态：包含中间产物"""
    question: str
    answer: str
    retrieved_docs: list   # 私有：外部看不到
    retry_count: int       # 私有

builder = StateGraph(
    InternalState,
    input=InputState,      # 输入过滤
    output=OutputState,    # 输出过滤
)
```

效果：

```python
graph.invoke({"question": "什么是 RAG？"})
# 返回 {'answer': '...'} —— retrieved_docs、retry_count 不在结果里
```

### 节点级私有状态（进阶技巧）

节点函数的类型注解决定它能看到哪些字段——LangGraph 会按注解**裁剪**传给节点的 state：

```python
class State(TypedDict):
    query: str
    private_scratchpad: str   # 只有部分节点需要

def node_sees_all(state: State) -> dict: ...
def node_sees_public(state: InputState) -> dict:  # 看不到 private_scratchpad
    ...
```

用途：让节点职责更清晰、防止意外依赖私有字段。

## 3.5 完整示例：一个带 Reducer 的状态设计

搜索增强问答场景的典型设计：

```python
import operator
from typing import Annotated, TypedDict
from langgraph.graph import MessagesState

class AgentState(MessagesState):
    """消息历史 + 业务字段"""
    query: str                                          # 用户当前问题
    retrieved_docs: Annotated[list, operator.add]       # 多路检索结果汇聚
    search_rounds: int                                  # 覆盖：每轮 +1（节点内自己算好）
    enough_info: bool                                   # 覆盖：路由判断结果
```

设计原则总结：

| 字段类型 | 推荐 Reducer | 理由 |
|---|---|---|
| 消息历史 | `add_messages` | 追加、可替换、可删除 |
| 并行产出的列表 | `operator.add` | 安全汇聚并发写入 |
| 路由/标记/计数值 | 默认覆盖 | 只有一个节点写它 |
| 聚合值（分数等） | 自定义（如 `max`） | 满足交换律即可 |

## 3.6 读取状态：get_state / get_state_history

图运行前后都能检查状态（需 checkpointer）：

```python
config = {"configurable": {"thread_id": "t-1"}}
graph.invoke({"question": "hi"}, config)

snapshot = graph.get_state(config)
snapshot.values          # 完整状态 dict
snapshot.next            # 下一批要执行的节点（非空 = 图还"停"在那里，如被 interrupt）
snapshot.tasks           # 待执行/暂停中任务的详情（含 interrupts 信息）
snapshot.metadata        # 步数、写入来源等
snapshot.config          # 该快照的 config（含 checkpoint_id）
```

`get_state_history(config)` 返回历史快照序列（第 13 章时间旅行详解）。

## 3.7 本章常见坑

1. **忘了加 Reducer，消息只剩最后一条** —— 新手第一大坑：`messages: list` 每次覆盖。必须 `Annotated[list, add_messages]` 或直接用 `MessagesState`。
2. **并行写无 Reducer 字段** → `InvalidUpdateError`。给该字段加 `operator.add` 或改为串行写。
3. **TypedDict 里写 `messages: list` 然后节点返回字符串** —— 类型不对，用标准消息对象（`HumanMessage` / `{"role": "user", "content": ...}`）。
4. **Pydantic State 里节点返回部分字段** —— 完全合法（partial update），不是必须返回全量。

## 本章小结

- State = 全图共享数据；节点返回 **partial update**，按字段合并
- Reducer 决定合并规则：默认覆盖、`add_messages` 追加、`operator.add` 拼接、可自定义
- 三个 Schema（input / output / internal）隔离对外接口与内部实现
- `get_state` 是调试与 HITL 的关键入口

> 下一章：节点——普通函数到企业级节点的全部写法。
