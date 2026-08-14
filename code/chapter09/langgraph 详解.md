# LangGraph 知识点详解（详细版）

> 环境说明：本文所有代码示例基于以下环境验证通过：
> - Python 3.13 + langgraph 1.2.10
> - 阿里云百炼 qwen3.7-max-2026-05-20（通过 init_chat_model + OpenAI 兼容模式接入）
> - .env 文件含 `ALI_BAILIAN_API_KEY` 和 `ALI_BAILIAN_BASE_URL` 两个变量
> - Windows 环境，建议在代码头部加 `sys.stdout.reconfigure(encoding="utf-8")` 防止中文乱码

---

## 通用初始化代码

以下代码在每个示例的"代码示例"部分默认已存在，不再重复粘贴：

```python
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Windows 防中文乱码
sys.stdout.reconfigure(encoding="utf-8")

# 加载 .env（假设 .env 与本文件同目录）
load_dotenv(Path(__file__).parent / ".env")

def get_llm(temperature=0.7):
    """初始化百炼 qwen3.7-max 模型"""
    return init_chat_model(
        model="qwen3.7-max-2026-05-20",
        model_provider="openai",
        base_url=os.getenv("ALI_BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        api_key=os.getenv("ALI_BAILIAN_API_KEY"),
        temperature=temperature,
    )

llm = get_llm()
```

> 后续代码示例中，`llm` 变量均指通过上述方式初始化的百炼模型实例。

---

# 一、核心概念层

## 1.1 LangGraph 与 LangChain 的关系

**概念详解**

LangChain 最初通过 `AgentExecutor` 实现 Agent，但随着应用复杂度上升，AgentExecutor 暴露了两个核心问题：一是无法表达循环（ReAct 模式本质是一个 while 循环，但 AgentExecutor 的内部实现不透明）；二是状态管理混乱，对话历史、中间结果等散落在各处，难以追踪和持久化。

LangGraph 用"状态图"重新定义了这个问题：把 Agent 的执行过程建模为一张有向图，每个节点是一步操作（调 LLM、调工具、处理数据），边定义操作间的流转规则，所有节点共享一个 State 对象。图天然支持循环（条件边可以指向之前的节点），状态在节点间显式传递，整个执行过程完全可追踪。

**关键关系**

| 维度 | LangChain / LCEL | LangGraph |
|------|-------------------|-----------|
| 职责 | 单节点内的逻辑编排（prompt → LLM → parser） | 多节点间的状态流转和循环控制 |
| 抽象级别 | 链（Chain）/ Runnable | 图（Graph） |
| 是否可替换 | 否，互补关系 | 否，互补关系 |
| 实际配合 | 图的节点内部用 LCEL chain 实现具体逻辑 | 负责节点间的编排和状态传递 |

**注意事项**

- LangGraph 是 LangChain 生态的一部分，不是独立竞品，安装 langgraph 会自动安装 langchain-core 作为依赖
- 节点内部完全可以不用 LCEL，纯 Python 函数也能作为节点，只是用 LCEL 可以方便地组合 prompt + LLM + parser
- LangChain 1.x 的 `create_agent`（`from langchain.agents import create_agent`）底层就是用 LangGraph 构建的，可以理解为 LangChain 在 Agent 层面已经全面转向 LangGraph

---

## 1.2 StateGraph（状态图）

**概念详解**

StateGraph 是 LangGraph 的核心类。创建时需要传入一个 State 类型（TypedDict 或 Pydantic Model），这个类型定义了图在执行过程中所有节点共享的数据结构。然后通过三个核心方法构建图的结构：`add_node()` 添加节点（每个节点是一个函数），`add_edge()` 添加固定边（A 执行完必定跳到 B），`add_conditional_edges()` 添加条件边（根据当前 state 动态决定下一个节点）。最后调用 `compile()` 将结构编译为可执行对象。

编译后的图是一个不可变的执行引擎，可以多次调用。这种"先构建结构、再编译执行"的设计类似于编译器的思路——先定义好整张图的结构，编译器做一些静态检查（如是否有从 START 到 END 的路径），然后才允许执行。

**代码示例**

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# 第一步：定义 State 类型
class MyState(TypedDict):
    question: str
    answer: str

# 第二步：定义节点函数
def generate_answer(state: MyState) -> dict:
    question = state["question"]
    answer = llm.invoke(question).content  # 调用百炼模型
    return {"answer": answer}  # 只返回要更新的字段

# 第三步：构建图结构
graph_builder = StateGraph(MyState)
graph_builder.add_node("answer_node", generate_answer)
graph_builder.add_edge(START, "answer_node")  # START → answer_node
graph_builder.add_edge("answer_node", END)     # answer_node → END

# 第四步：编译并执行
graph = graph_builder.compile()
result = graph.invoke({"question": "什么是 LangGraph？"})
print(result["answer"])
```

**注意事项**

- State 类型必须用 `TypedDict`（或继承自 `MessagesState` 等 LangGraph 内置类型），不能用普通 dict
- `add_node()` 的第一个参数是节点名（字符串），第二个参数是函数引用，不要加括号调用
- 图必须从 START 出发且至少有一条到 END 的路径，否则 compile 时会报错
- `compile()` 返回的是一个新对象，原始 `graph_builder` 不会被修改

---

## 1.3 State（状态）

**概念详解**

State 是图在所有节点间共享的"数据总线"。每个节点函数接收当前 state 作为入参，执行完返回一个 dict，dict 中的 key 对应 state 的字段，value 是该字段的新值。LangGraph 会根据字段的 reducer 策略将返回值合并到当前 state 中。

理解 State 的关键在于它不是"参数传递"而是"状态累积"。每个节点看到的是经过前面所有节点处理后的完整 state，而不是上一个节点传来的片段。这使得任何一个节点都能访问到之前所有步骤产生的信息，而不需要显式地层层传递。

State 通常用 `TypedDict` 定义，也可以继承 LangGraph 提供的 `MessagesState`（预置了 `messages` 字段 + `add_messages` reducer，专门用于对话场景）。对于复杂场景，可以用 Pydantic Model 实现字段校验和默认值。

**代码示例**

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, MessagesState

# 方式一：自定义 TypedDict + Annotated
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]  # 追加策略
    user_name: str                          # 覆盖策略（默认）
    call_count: int                          # 覆盖策略

# 方式二：继承 MessagesState（自带 messages 字段）
class AgentState(MessagesState):
    user_name: str
    call_count: int
```

**注意事项**

- State 定义中，字段的类型标注不只是"文档"，LangGraph 会根据 `Annotated` 中的 reducer 确定合并策略
- 不加 `Annotated` 的字段默认用"覆盖"策略：后一个节点的返回值直接替换原值
- `messages` 字段几乎总是需要 `add_messages` reducer，否则每轮对话都会丢失历史
- State 应该只放"真正需要在节点间传递的数据"，不要把临时变量放进去

---

## 1.4 Reducer（状态合并策略）

**概念详解**

Reducer 回答了一个关键问题：当节点 B 返回 `{"messages": [new_message]}` 时，这个新 message 是"替换"掉 state 中已有的 messages 列表，还是"追加"到列表末尾？

默认策略（不指定 reducer）是覆盖。这在很多场景下是合理的——比如 `user_name` 字段，节点 B 返回新名字就应该直接替换旧名字。但对于对话历史 `messages`，每轮对话产生的新消息应该追加到已有列表，而不是替换。这时就需要用 `Annotated[list, add_messages]` 声明追加策略。

Reducer 本质上是一个函数，签名为 `reducer(current_value, new_value) -> merged_value`。`add_messages` 是 LangGraph 内置的 reducer，它知道如何合并 LangChain 的 Message 对象（去重、处理 ID 更新等）。你也可以写自定义 reducer 实现更复杂的合并逻辑。

**代码示例**

```python
from typing import TypedDict, Annotated
from operator import add
from langgraph.graph.message import add_messages

class MyState(TypedDict):
    # 追加策略：每次节点返回的 list 会拼接到已有 list 后面
    messages: Annotated[list, add_messages]

    # 列表拼接策略：用 operator.add（等同于 list1 + list2）
    tags: Annotated[list, add]

    # 数字累加策略：用 operator.add（等同于 5 + 3 = 8）
    score: Annotated[int, add]

    # 默认覆盖策略：不加 Annotated
    current_step: str

# 自定义 reducer 示例
def keep_max(current: int, new: int) -> int:
    """始终保留较大的值"""
    return max(current, new) if current is not None else new

class GameState(TypedDict):
    high_score: Annotated[int, keep_max]
```

**注意事项**

- `add_messages` 不仅做"追加"，它还会根据 message ID 做去重和更新——如果新消息的 ID 与已有消息相同，会更新而非追加
- `operator.add` 对 list 是拼接，对 int 是加法，对 str 是字符串拼接——注意不同类型的语义
- 自定义 reducer 必须能处理 `current_value=None` 的情况（第一次执行时 state 中该字段还没有值）
- 忘记给 `messages` 字段加 reducer 是最常见的"对话历史丢失"问题的根因

---

## 1.5 Node（节点）

**概念详解**

节点是图中的执行单元，本质就是一个普通 Python 函数。函数接收一个参数（当前 state），返回一个 dict（对 state 的更新）。返回值不需要包含 state 的所有字段，只需包含你想要修改的字段——LangGraph 会根据每个字段的 reducer 策略自动合并。

节点函数的设计哲学是"声明式更新"——你不直接修改 state，而是告诉 LangGraph"我想把这些字段更新成这些值"，由 LangGraph 的 reducer 机制负责安全合并。这种设计避免了多节点并发写入时的竞态条件，也让状态流转变得可追踪。

节点可以是同步函数（`def`）或异步函数（`async def`），但在同一个图中混用同步和异步节点会有性能损失。节点内部可以做任何事情：调 LLM、查数据库、调外部 API、执行 Python 代码等。

**代码示例**

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage

class State(TypedDict):
    messages: Annotated[list, add_messages]

def chat_node(state: State) -> dict:
    # 从 state 中取出对话历史
    messages = state["messages"]
    # 调用百炼模型
    response = llm.invoke(messages)
    # 只返回要更新的字段（messages 会通过 add_messages 追加）
    return {"messages": [response]}

def tool_node(state: State) -> dict:
    # 节点可以做任何事情
    last_message = state["messages"][-1]
    # 模拟工具执行
    result = f"工具执行结果：处理了 '{last_message.content}'"
    from langchain_core.messages import ToolMessage
    return {"messages": [ToolMessage(content=result, tool_call_id="call_001")]}
```

**注意事项**

- 节点函数名就是 `add_node` 时传入的名字，但函数本身的 `__name__` 不影响——以 `add_node("my_node", func)` 注册的名字为准
- 返回的 dict 中，key 必须与 State 中定义的字段名匹配，否则会被忽略（不报错但不生效）
- 节点函数不应该有副作用（如修改全局变量），所有需要传递的信息都应该通过 state
- 如果节点返回 `None` 或空 dict，表示"不更新任何字段"，图会直接走到下一个节点

---

## 1.6 Edge（边）

**概念详解**

边定义了节点间的执行顺序。LangGraph 有三种边：

普通边（`add_edge(A, B)`）：A 执行完后必定走到 B。适用于固定流程——比如"先解析输入，再调 LLM，再格式化输出"这种线性步骤。

条件边（`add_conditional_edges(A, router_func, path_map)`）：A 执行完后，调用 `router_func(state)`，根据返回值决定下一个节点。`router_func` 返回一个字符串（节点名或 END），`path_map` 是可选的映射字典，把 router_func 的返回值映射到实际的节点名。条件边是实现 Agent 循环（ReAct 模式）的核心——"如果 LLM 说要调工具就走到工具节点，否则走到 END"。

入口/出口边：从 `START` 出发的边定义图的起点（`add_edge(START, "first_node")`），指向 `END` 的边定义图的终点（`add_edge("last_node", END)`）。

**代码示例**

```python
from langgraph.graph import StateGraph, START, END

graph = StateGraph(State)

# 普通边：START → node_a → node_b
graph.add_edge(START, "node_a")
graph.add_edge("node_a", "node_b")

# 条件边：node_b 执行后根据 state 决定走向
def router(state: State) -> str:
    if state.get("needs_tool"):
        return "tool_node"
    return END

graph.add_conditional_edges("node_b", router)

# 注册所有节点
graph.add_node("node_a", node_a_func)
graph.add_node("node_b", node_b_func)
graph.add_node("tool_node", tool_func)
graph.add_edge("tool_node", "node_b")  # 工具执行完回到 node_b 形成循环
```

**注意事项**

- 条件边的 router 函数返回的字符串必须是已注册的节点名或 `END`，否则运行时会报错
- `path_map` 参数是可选的，但建议使用——它让路由逻辑更清晰，也便于多语言化（如 router 返回 "yes"/"no"，path_map 映射到具体节点名）
- 一个节点可以有多条出边，但不能同时有普通边和条件边——`add_edge(A, B)` 后又对 A 用 `add_conditional_edges` 会覆盖普通边
- 并行执行：从一个节点引出多条普通边到不同节点，这些目标节点会并行执行

---

## 1.7 START 与 END

**概念详解**

`START` 和 `END` 是 LangGraph 内置的两个特殊节点标识，分别代表图的入口和出口。

`START` 不是一个真正的节点（没有对应的函数），它是一个"虚拟起点"。当你写 `add_edge(START, "node_a")`，意思是"图开始执行时，第一个运行的节点是 node_a"。一个图必须有至少一条从 START 出发的边。

`END` 同样是虚拟终点。当某个节点的边指向 END 时，图执行完毕，`invoke()` 返回最终的 state。一个图可以有多条指向 END 的边（如条件边的多个分支都可以指向 END）。

**代码示例**

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(State)

# 入口：图从这里开始
builder.add_edge(START, "entry_node")

# 中间节点链
builder.add_node("entry_node", entry_func)
builder.add_edge("entry_node", "process_node")
builder.add_node("process_node", process_func)

# 两种出口：处理成功或失败都到 END
builder.add_conditional_edges(
    "process_node",
    lambda state: "success_node" if state.get("ok") else "fail_node"
)
builder.add_node("success_node", success_func)
builder.add_node("fail_node", fail_func)
builder.add_edge("success_node", END)
builder.add_edge("fail_node", END)

graph = builder.compile()
```

**注意事项**

- `START` 和 `END` 从 `langgraph.graph` 导入，不要自己定义
- 图必须从 START 出发，但可以有多个"第一跳"节点（`add_edge(START, "a")` + `add_edge(START, "b")` 表示 a 和 b 并行执行）
- 指向 END 的边不一定是最后一步——条件边可以在任何时候指向 END 提前结束图
- 如果图执行中没有走到任何指向 END 的边，最终会触发 recursion_limit 超时

---

## 1.8 compile（编译）

**概念详解**

`compile()` 是从"图的构建器"到"可执行图"的转换步骤。在 compile 之前，你用 `add_node` / `add_edge` 搭建的是图的结构定义；compile 后得到的是一个不可变的执行引擎，可以反复调用。

compile 时可以传入以下关键参数：

- `checkpointer`：状态持久化器，用于多轮对话记忆和断点续传
- `interrupt_before` / `interrupt_after`：在指定节点前/后暂停执行，用于 Human-in-the-loop
- `recursion_limit`：最大循环次数限制（默认 25），防止死循环

compile 会做一些基本的图结构校验，比如检查是否有从 START 到 END 的可达路径、是否有未注册的节点引用等。

**代码示例**

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

builder = StateGraph(State)
# ... 添加节点和边 ...

# 基本编译
graph = builder.compile()

# 带持久化的编译（多轮对话用）
graph_with_memory = builder.compile(checkpointer=MemorySaver())

# 带人机协作的编译
graph_with_interrupt = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["review_node"],  # 在 review_node 执行前暂停
)

# 带递归限制的编译
graph_safe = builder.compile(recursion_limit=50)
```

**注意事项**

- compile 后的图不可修改——想改图结构需要重新构建一个新的 builder
- `interrupt_before` / `interrupt_after` 必须配合 `checkpointer` 使用，因为暂停后需要持久化状态才能续传
- `recursion_limit` 计的是"节点执行次数"而非"while 循环次数"——每次执行一个节点算一次，超过限制会抛 `GraphRecursionError`
- 同一个 builder 可以 compile 多次，传入不同参数得到不同配置的图

---

## 1.9 invoke / stream / astream

**概念详解**

编译后的图有三种执行方式，区别在于返回方式和异步特性：

`invoke(input, config)`：同步执行，一次跑完整个图，返回最终 state。最简单，适合脚本和快速测试。缺点是整个过程是阻塞的，对长时间运行的图（如多轮工具调用）体验不佳。

`stream(input, config)`：同步流式执行，每执行完一个节点就 yield 一次中间结果。适合需要展示进度的场景——比如 CLI 工具中逐步打印"正在分析 → 正在调工具 → 正在生成回答"。

`astream(input, config)`：异步流式执行，与 stream 行为一致但返回 async generator，可以用 `async for` 消费。适合 FastAPI 等 async 框架，不会阻塞事件循环。

三种方法的 `input` 参数是图的初始 state（可以只包含部分字段），`config` 参数可传入 `thread_id`（配合 checkpointer）等配置。

**代码示例**

```python
from langchain_core.messages import HumanMessage

# invoke：一次性执行
result = graph.invoke({"messages": [HumanMessage(content="你好")]})
print(result["messages"][-1].content)

# stream：逐步输出
for chunk in graph.stream({"messages": [HumanMessage(content="你好")]}):
    print(chunk)  # 每个节点执行完会输出一个 dict
    # 如：{"chat_node": {"messages": [AIMessage(...)]}}

# astream：异步流式
import asyncio

async def run():
    async for chunk in graph.astream({"messages": [HumanMessage(content="你好")]}):
        print(chunk)

asyncio.run(run())
```

**注意事项**

- `stream` 和 `astream` 默认返回的是 `{"node_name": state_update}` 格式（updates 模式），不是完整 state
- 想拿到每步的完整 state，传入 `stream_mode="values"`：`graph.stream(input, stream_mode="values")`
- `invoke` 底层其实就是 `stream` 的封装——消费完所有 chunk 后返回最后一个
- `config` 参数中最常用的是 `{"configurable": {"thread_id": "xxx"}}`，配合 checkpointer 实现多轮对话

---

## 1.10 Stream Mode

**概念详解**

`stream()` 和 `astream()` 支持 `stream_mode` 参数，控制流式输出的粒度和格式。四种模式：

`values`：每次输出完整的当前 state。适合需要每一步都看到完整状态的调试场景。数据量大，但信息最全。

`updates`（默认）：每次输出 `{"node_name": {"field": new_value}}`，只包含该节点的增量更新。数据量小，适合做进度展示。

`messages`：专门用于 LLM 输出，以 token 为单位流式返回 LLM 生成的文本。适合做"打字机效果"的前端展示。需要配合 `stream_mode=["messages", "updates"]` 才能同时看到节点级和 token 级的输出。

`debug`：最详细的模式，包含每一步的输入、输出、配置等全部信息。适合开发调试阶段定位问题。

**代码示例**

```python
from langchain_core.messages import HumanMessage

input_data = {"messages": [HumanMessage(content="解释什么是递归")]}

# values 模式：每步输出完整 state
print("=== values 模式 ===")
for chunk in graph.stream(input_data, stream_mode="values"):
    print(f"当前 messages 数量: {len(chunk.get('messages', []))}")

# updates 模式：每步输出增量更新
print("\n=== updates 模式 ===")
for chunk in graph.stream(input_data, stream_mode="updates"):
    for node_name, update in chunk.items():
        print(f"节点 [{node_name}] 更新了: {list(update.keys())}")

# messages 模式：token 级流式（LLM 逐字输出）
print("\n=== messages 模式 ===")
for msg, metadata in graph.stream(input_data, stream_mode="messages"):
    print(msg.content, end="", flush=True)
print()

# 多模式组合：同时输出 updates + messages
print("\n=== 组合模式 ===")
for chunk in graph.stream(input_data, stream_mode=["updates", "messages"]):
    print(chunk)
```

**注意事项**

- `messages` 模式返回的是 `(message_chunk, metadata)` 元组，不是单个值
- `messages` 模式只能在图中有 LLM 调用节点时才能看到输出——如果当前节点不调 LLM，不会有 token 流
- 组合模式传列表如 `["updates", "messages"]`，返回的 chunk 格式会因模式不同而不同，需要用 isinstance 或 key 判断
- 生产环境推荐 `updates` 模式做进度展示，`messages` 模式做前端打字机效果，`debug` 模式只在开发时用

---

# 二、Agent 基础模式层

## 2.1 ReAct 模式

**概念详解**

ReAct（Reasoning + Acting）是当前最主流的 Agent 架构模式。核心思想是让 LLM 交替进行"推理"和"行动"：LLM 先思考用户的问题需要什么信息或操作，然后调用工具执行，拿到工具返回的结果后再继续推理，如此循环直到 LLM 认为已经可以给出最终答案。

在 LangGraph 中，ReAct 模式的图结构非常简洁：两个节点（LLM 节点 + 工具节点），两条边（LLM → 工具的条件边、工具 → LLM 的普通边），外加一条 LLM → END 的条件边。条件边检查 LLM 回复中是否包含 `tool_calls`——有就走到工具节点，没有就到 END 结束。这个"LLM → 工具 → LLM"的循环就是 ReAct 的精髓。

**图结构示意**

```
START → [LLM 节点] ←─────── [工具节点]
                ↓                    ↑
            条件边 ──(有 tool_calls)──┘
                ↓
            (无 tool_calls) → END
```

**代码示例**

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool

# 定义工具
@tool
def search(query: str) -> str:
    """搜索互联网获取信息"""
    # 实际场景接入搜索 API，这里用模拟数据
    return f"搜索结果：关于 '{query}' 的信息..."

# 将工具绑定到 LLM
llm_with_tools = llm.bind_tools([search])

# 定义 State
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# LLM 节点
def call_model(state: AgentState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# 构建图
builder = StateGraph(AgentState)
builder.add_node("llm", call_model)
builder.add_node("tools", ToolNode([search]))

# 边：START → llm → 条件判断 → (tools → llm) 或 (END)
builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", tools_condition)  # tools_condition 自动判断
builder.add_edge("tools", "llm")  # 工具执行完回到 llm

graph = builder.compile()

# 执行
from langchain_core.messages import HumanMessage
result = graph.invoke({"messages": [HumanMessage(content="帮我搜索 LangGraph 是什么")]})
print(result["messages"][-1].content)
```

**注意事项**

- `tools_condition` 是 LangGraph 预置的条件函数，它检查最后一条消息是否有 `tool_calls`，有则返回 `"tools"`，无则返回 `END`
- `ToolNode([search])` 自动处理 `tool_calls` 的执行和 `ToolMessage` 的生成，不需要手写工具执行逻辑
- 百炼 qwen3.7-max 通过 `bind_tools` 绑定工具后，会在回复中生成 OpenAI 兼容格式的 `tool_calls`
- 循环次数受 `recursion_limit` 限制（默认 25），如果 Agent 陷入反复调工具的循环，会抛 `GraphRecursionError`

---

## 2.2 Tool Node（工具节点）

**概念详解**

ToolNode 是 LangGraph 预置的工具执行节点，封装了"从 state 中取出 LLM 的 tool_calls → 依次执行对应工具 → 将结果转为 ToolMessage 放回 state"这一完整流程。

它的工作原理：当 LLM 节点输出一条带 `tool_calls` 的 AIMessage 时，ToolNode 会从这条消息中提取所有 tool_calls（LLM 可以一次调多个工具），对每个 tool_call 找到对应的工具函数执行，生成对应的 ToolMessage（包含工具执行结果），然后将所有 ToolMessage 作为返回值放回 state。`add_messages` reducer 会将它们追加到 messages 列表。

ToolNode 还处理了一些边界情况：工具执行异常时会返回错误信息作为 ToolMessage 内容（而不是中断图执行），让 LLM 有机会根据错误信息重试或调整策略。

**代码示例**

```python
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """计算数学表达式。参数 expression: 数学表达式字符串，如 '2+3*4'"""
    try:
        result = eval(expression)  # 生产环境请用 ast.literal_eval 或更安全的方式
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算失败：{e}"

@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息。参数 city: 城市名"""
    # 实际场景接入天气 API
    weather_data = {"北京": "晴 25°C", "上海": "多云 28°C"}
    return weather_data.get(city, f"未找到 {city} 的天气信息")

# 创建 ToolNode（传入工具列表）
tool_node = ToolNode([calculator, get_weather])

# ToolNode 内部处理流程等价于：
# 1. 从 state["messages"][-1] 取出 AIMessage
# 2. 遍历 message.tool_calls
# 3. 对每个 tool_call，根据 name 找到对应工具，用 args 执行
# 4. 为每个结果创建 ToolMessage，tool_call_id 对应
# 5. 返回 {"messages": [tool_msg1, tool_msg2, ...]}
```

**注意事项**

- ToolNode 内部会 catch 工具执行异常，把错误信息作为 ToolMessage 返回——LLM 能看到错误并自行调整
- 如果 LLM 生成了不存在的工具名，ToolNode 会返回一条"工具不存在"的错误消息
- 工具的 docstring 非常重要——它是 LLM 判断"什么时候用这个工具、怎么传参数"的主要依据
- 百炼 qwen3.7-max 支持一次生成多个 tool_calls，ToolNode 会全部执行后一次性返回

---

## 2.3 bind_tools

**概念详解**

`bind_tools` 是 LangChain LLM 对象的方法，用于将工具定义绑定到 LLM 上。绑定后，LLM 就能在回复中生成结构化的 `tool_calls`——不再是纯文本回复，而是包含"工具名 + 参数"的结构化调用指令。

从技术角度看，bind_tools 做了两件事：一是将工具定义（函数名、参数 schema、描述）转换为 LLM 能理解的格式（百炼/OpenAI 兼容模式下是 JSON Schema），放在 API 请求的 `tools` 参数中；二是让 LLM 知道"你有这些工具可用，当需要时可以调用"。

LLM 回复中的 `tool_calls` 是一个列表，每个元素包含 `name`（工具名）、`args`（参数字典）、`id`（唯一调用标识，用于配对 ToolMessage）。ToolNode 正是用这个 `name` 来找到对应工具函数，用 `args` 来执行。

**代码示例**

```python
from langchain_core.tools import tool

@tool
def search_web(query: str) -> str:
    """搜索互联网。参数 query: 搜索关键词"""
    return f"搜索 {query} 的结果..."

@tool
def write_file(path: str, content: str) -> str:
    """写入文件。参数 path: 文件路径, content: 文件内容"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"已写入 {path}"

# 绑定工具到百炼模型
llm_with_tools = llm.bind_tools([search_web, write_file])

# 测试：LLM 会根据问题决定是否调用工具
from langchain_core.messages import HumanMessage
response = llm_with_tools.invoke([HumanMessage(content="帮我搜索 LangGraph 教程")])

# 检查是否有 tool_calls
if response.tool_calls:
    for tc in response.tool_calls:
        print(f"工具: {tc['name']}, 参数: {tc['args']}, ID: {tc['id']}")
else:
    print(f"直接回答: {response.content}")
```

**注意事项**

- `@tool` 装饰器会自动从函数签名和 docstring 生成工具 schema——参数类型注解和 docstring 是 LLM 理解工具的关键
- 百炼 qwen3.7-max 使用 OpenAI 兼容模式，`bind_tools` 底层走的是 function calling 协议
- 一个 LLM 可以 bind_tools 多次（每次追加），但通常一次绑定所有需要的工具即可
- 绑定的工具数量不宜过多（一般 < 20 个），太多会让 LLM 选择困难，准确率下降

---

## 2.4 tools_condition（条件路由）

**概念详解**

`tools_condition` 是 LangGraph 预置的条件路由函数，专门用于 ReAct 模式的路由判断。它的工作逻辑非常简单：检查 state 中最后一条消息（应该是 LLM 的 AIMessage）是否包含 `tool_calls` 字段——如果包含，返回字符串 `"tools"`（路由到工具节点）；如果不包含，返回 `END`（结束图执行）。

这个函数是 `add_conditional_edges("llm", tools_condition)` 的标准搭配，几乎每个手搭的 ReAct Agent 都会用到。它省去了手写"检查 tool_calls 并返回节点名"的样板代码。

**代码示例**

```python
from langgraph.prebuilt import tools_condition
from langgraph.graph import StateGraph, START, END

builder = StateGraph(AgentState)
builder.add_node("llm", call_model)
builder.add_node("tools", ToolNode(tools))

# tools_condition 的等价手动实现：
def my_tools_condition(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"  # 有工具调用 → 去工具节点
    return END          # 无工具调用 → 结束

# 两种写法效果相同：
builder.add_conditional_edges("llm", tools_condition)
# 或
# builder.add_conditional_edges("llm", my_tools_condition)
# 或带 path_map（显式映射，更清晰）：
# builder.add_conditional_edges("llm", my_tools_condition, {"tools": "tools", END: END})
```

**注意事项**

- `tools_condition` 假设你的工具节点名为 `"tools"`——如果你用 `add_node("my_tool_node", ...)` 注册了不同的名字，需要用 path_map 映射：`add_conditional_edges("llm", tools_condition, {"tools": "my_tool_node", END: END})`
- 它只检查最后一条消息是否有 tool_calls，不检查 tool_calls 的内容——如果 LLM 产生了无效的工具名，路由仍然会走到工具节点，由 ToolNode 处理错误
- 在 LangGraph 1.x 中，`tools_condition` 的返回值是 `END` 常量而非字符串 `"END"`，两者在 LangGraph 内部等价

---

## 2.5 create_react_agent

**概念详解**

`create_react_agent` 是 LangGraph 预置的快捷函数，一行代码创建完整的 ReAct Agent。它内部自动完成：创建 State（MessagesState）、添加 LLM 节点和 ToolNode、连接 START→LLM→条件边→(tools→LLM 或 END) 的完整循环。

这个函数是快速搭建 Agent 的首选——当你只需要一个标准的"LLM + 工具"Agent 时，不需要手动搭图结构。但如果你需要自定义图结构（如多 Agent 协作、加入预处理/后处理节点），就需要手搭。

**代码示例**

```python
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """搜索互联网获取信息"""
    return f"搜索 {query} 的结果..."

@tool
def calculator(expression: str) -> str:
    """计算数学表达式"""
    return str(eval(expression))

# 一行创建 ReAct Agent（内部自动构建图结构）
agent = create_react_agent(
    model=llm,           # 百炼 qwen3.7-max
    tools=[search, calculator],
)

# 直接使用（与手搭的图调用方式相同）
from langchain_core.messages import HumanMessage
result = agent.invoke({
    "messages": [HumanMessage(content="帮我算一下 123 * 456")]
})
print(result["messages"][-1].content)
```

**注意事项**

- `create_react_agent` 返回的是一个编译后的图（CompiledGraph），直接可以 invoke/stream，不需要再 compile
- 它内部用了 `MessagesState` 作为 State，所以 invoke 时传入 `{"messages": [...]}` 格式
- 可以传入 `prompt` 参数自定义系统提示词：`create_react_agent(model=llm, tools=tools, prompt="你是一个助手...")`
- 项目 chapter08 的数据分析 Agent 就用了这个函数：`create_react_agent(model=llm, tools=[PythonAstREPLTool(...)])`

---

## 2.6 Checkpointer（状态持久化）

**概念详解**

Checkpointer 是 LangGraph 的状态持久化机制。它在每个节点执行完毕后，自动将当前完整的 state 保存一个快照（snapshot）。这意味着如果图执行到一半中断了（主动暂停或异常崩溃），可以从最近的快照恢复继续执行。

Checkpointer 最常见的用途是实现多轮对话记忆。没有 Checkpointer 时，每次 `invoke()` 都是独立的——图不知道上一次对话发生了什么。加了 Checkpointer 后，用相同的 `thread_id` 多次调用，第二次会自动加载第一次的完整 state（包括对话历史），实现真正的"多轮对话"。

LangGraph 提供多种 Checkpointer 实现，区别在于存储后端：`MemorySaver`（内存，开发用）、`SqliteSaver`（SQLite 文件，单机持久化）、`PostgresSaver`（PostgreSQL，生产级多实例共享）。

**代码示例**

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# 开发环境：内存 Checkpointer
memory_saver = MemorySaver()
agent = create_react_agent(
    model=llm,
    tools=[search],
    checkpointer=memory_saver,  # 传入 checkpointer
)

# 多轮对话（用 thread_id 关联会话）
config = {"configurable": {"thread_id": "user_001_session_1"}}

# 第一轮
agent.invoke(
    {"messages": [HumanMessage(content="我叫小明")]},
    config=config
)

# 第二轮（自动加载第一轮的 state，知道用户叫小明）
result = agent.invoke(
    {"messages": [HumanMessage(content="我叫什么名字？")]},
    config=config
)
print(result["messages"][-1].content)  # "你叫小明"

# 不同 thread_id 是独立会话
config2 = {"configurable": {"thread_id": "user_002_session_1"}}
result2 = agent.invoke(
    {"messages": [HumanMessage(content="我叫什么名字？")]},
    config=config2
)
print(result2["messages"][-1].content)  # 不知道用户名字
```

**注意事项**

- `MemorySaver` 的数据存在进程内存中——程序重启后全部丢失，只适合开发测试
- 生产环境需要 `SqliteSaver` 或 `PostgresSaver`，但需要额外安装包（见 2.8）
- `thread_id` 是逻辑会话标识，可以是任意字符串——通常用用户 ID + 会话 ID 组合
- Checkpointer 不仅存对话历史，还存完整 state（包括你自定义的所有字段），恢复时全部还原

---

## 2.7 thread_id（会话标识）

**概念详解**

`thread_id` 是配合 Checkpointer 使用的会话标识。它是一个字符串，通过 `config={"configurable": {"thread_id": "xxx"}}` 传入图的执行方法。相同 thread_id 的多次调用会共享同一份 state 历史，不同 thread_id 之间完全隔离。

从实现角度看，Checkpointer 内部维护了一个 `{thread_id → [snapshot1, snapshot2, ...]}` 的映射。每次 invoke 时，如果 config 中有 thread_id 且该 ID 已有快照，会先加载最新快照作为初始 state，然后执行新的输入，执行完再保存新快照。如果没有 thread_id，每次都是全新开始。

thread_id 的命名建议包含业务语义，如 `f"user_{user_id}_session_{session_id}"`，便于调试和管理。在多 Agent 系统中，每个子 Agent 可以有独立的 thread_id，也可以共享一个。

**代码示例**

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
agent = create_react_agent(model=llm, tools=[search], checkpointer=memory)

# 会话 A
config_a = {"configurable": {"thread_id": "conversation_a"}}
agent.invoke({"messages": [HumanMessage(content="记住：我的密码是1234")]}, config=config_a)

# 会话 B（完全不知道会话 A 的内容）
config_b = {"configurable": {"thread_id": "conversation_b"}}
result_b = agent.invoke({"messages": [HumanMessage(content="我的密码是什么？")]}, config=config_b)
# LLM 会说不知道，因为这是独立会话

# 会话 A 继续（加载之前的 state）
result_a = agent.invoke({"messages": [HumanMessage(content="我的密码是什么？")]}, config=config_a)
# LLM 会说"你的密码是1234"，因为会话 A 的 state 中有这个信息

# 查看 Checkpointer 中的会话历史
print(memory.storage)  # 可以看到两个 thread 的数据
```

**注意事项**

- thread_id 必须放在 `config["configurable"]["thread_id"]` 中，层级不能错
- 同一个 thread_id 的 invoke 可以只传入新增的 message（如只传最新的 HumanMessage），Checkpointer 会自动补上历史
- 如果不传 thread_id 且图有 checkpointer，invoke 仍能执行但不做持久化——相当于一次性执行
- 在 Web 应用中，通常每个用户一个 thread_id，或每个用户每个对话窗口一个 thread_id

---

## 2.8 MemorySaver vs Production Saver

**概念详解**

LangGraph 的 Checkpointer 有三种实现，对应不同环境：

`MemorySaver`：纯内存实现，数据存在 Python 进程的字典中。零依赖、零配置，开发阶段最方便。但程序重启后数据全部丢失，且多个进程/容器之间不共享，不能用于生产。

`SqliteSaver`：基于 SQLite 文件的持久化。数据存在本地 `.db` 文件中，重启不丢失。适合单机部署的场景（如个人项目、小型应用）。需要安装 `langgraph-checkpoint-sqlite` 包。

`PostgresSaver`：基于 PostgreSQL 数据库的持久化。支持多进程/多容器共享同一份状态，适合生产级多实例部署。需要安装 `langgraph-checkpoint-postgres` 包，并有可用的 PostgreSQL 实例。

**代码示例**

```python
# ===== 开发环境：MemorySaver =====
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()
agent = create_react_agent(model=llm, tools=tools, checkpointer=checkpointer)

# ===== 单机生产：SqliteSaver =====
# 先安装：pip install langgraph-checkpoint-sqlite
# from langgraph.checkpoint.sqlite import SqliteSaver
#
# # 方式一：指定数据库文件路径
# checkpointer = SqliteSaver.from_conn_string("agent_state.db")
# # 方式二：使用上下文管理器（推荐，自动管理连接）
# with SqliteSaver.from_conn_string("agent_state.db") as checkpointer:
#     agent = create_react_agent(model=llm, tools=tools, checkpointer=checkpointer)
#     # ... 使用 agent ...

# ===== 多实例生产：PostgresSaver =====
# 先安装：pip install langgraph-checkpoint-postgres
# from langgraph.checkpoint.postgres import PostgresSaver
#
# DB_URI = "postgresql://user:pass@localhost:5432/langgraph"
# with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
#     agent = create_react_agent(model=llm, tools=tools, checkpointer=checkpointer)
```

**注意事项**

- 三种 Checkpointer 的 API 完全一致，切换时只需改 import 和初始化，图的代码不需要动
- 当前项目（python-agent-lab）未安装 `langgraph-checkpoint-sqlite`，使用前需先安装
- SqliteSaver 在高并发写入时会有锁竞争（SQLite 的文件锁限制），高并发场景用 PostgresSaver
- 除了 LangGraph 官方的三种，也可以自定义 Checkpointer（继承 BaseCheckpointSaver），如用 Redis 实现

---

## 2.9 Recursion Limit（递归限制）

**概念详解**

`recursion_limit` 是图执行时的安全阀，限制节点执行的最大次数（不是"循环"次数，而是"每个节点被调用的总次数"）。默认值 25，意味着整个图执行过程中，所有节点的调用次数加起来不能超过 25 次。超过会抛出 `GraphRecursionError`。

这个限制主要是防止 Agent 陷入死循环——比如 LLM 反复调用同一个工具但参数总是错的，或者条件边的路由逻辑有 bug 导致两个节点间无限来回跳转。

**代码示例**

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

# 方式一：compile 时设置
builder = StateGraph(AgentState)
# ... 添加节点和边 ...
graph = builder.compile(
    checkpointer=MemorySaver(),
    recursion_limit=50,  # 提高到 50
)

# 方式二：create_react_agent 不支持 compile 参数，需用运行时 config
agent = create_react_agent(model=llm, tools=tools, checkpointer=MemorySaver())
result = agent.invoke(
    {"messages": [HumanMessage(content="复杂任务...")]},
    config={
        "configurable": {"thread_id": "session_1"},
        "recursion_limit": 50,  # 运行时覆盖
    }
)

# 捕获递归超限
from langgraph.errors import GraphRecursionError
try:
    result = agent.invoke(input, config={"recursion_limit": 10})
except GraphRecursionError:
    print("Agent 执行超过 10 步，可能陷入循环")
```

**注意事项**

- `recursion_limit` 计的是所有节点执行次数的总和——如果图有 3 个节点循环 8 次，总执行次数是 24
- 复杂的多 Agent 系统或需要多轮工具调用的任务，默认 25 可能不够，建议设为 50-100
- `create_react_agent` 返回的图不支持 compile 参数，只能通过运行时 config 设置 recursion_limit
- 超限不一定意味着 bug——有些任务确实需要很多步骤。但如果频繁超限，应检查 Agent 的 prompt 和工具设计是否有问题

---

# 三、高级模式层

## 3.1 Human-in-the-loop（人机协作）

**概念详解**

Human-in-the-loop（HITL）让图在执行过程中暂停，等待人工确认或修改后再继续。典型场景：Agent 在执行敏感操作（发邮件、删除数据、提交订单）前，先暂停让人工审核；或 Agent 遇到不确定性时，暂停向人工求助。

LangGraph 1.x 中实现 HITL 有两种方式：

`interrupt_before` / `interrupt_after`：在 compile 时指定哪些节点前/后暂停。适合固定位置的审核点——比如"每次执行到 `send_email` 节点前都暂停"。

`interrupt()` 函数（推荐）：在节点函数内部动态调用 `interrupt(value)`，传入需要人工查看的值。暂停后，人工通过 `Command(resume=answer)` 恢复执行，`interrupt()` 函数返回人工的输入。适合需要根据运行时状态动态决定是否暂停的场景。

两种方式都需要配合 Checkpointer 使用——暂停后必须持久化状态才能续传。

**代码示例**

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_core.tools import tool

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """发送邮件"""
    # 实际接入邮件 API
    return f"邮件已发送至 {to}"

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def review_and_send(state: AgentState) -> dict:
    """节点内部使用 interrupt 动态暂停"""
    last_ai_message = state["messages"][-1]
    # 提取 LLM 想发的邮件内容
    email_content = last_ai_message.content

    # 动态暂停，等待人工确认
    approval = interrupt({
        "question": "是否允许发送以下邮件？",
        "email_content": email_content
    })

    if approval == "yes":
        # 人工批准，执行发送
        result = "邮件已发送"
    else:
        result = f"邮件发送被拒绝，原因：{approval}"

    from langchain_core.messages import ToolMessage
    return {"messages": [ToolMessage(content=result, tool_call_id="send_email")]}

# 构建图
builder = StateGraph(AgentState)
builder.add_node("llm", call_model)
builder.add_node("review", review_and_send)
builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", tools_condition, {"tools": "review", END: END})
builder.add_edge("review", "llm")

graph = builder.compile(checkpointer=MemorySaver())

# 第一次调用：会在 review 节点暂停
config = {"configurable": {"thread_id": "hitl_1"}}
result = graph.invoke({"messages": [HumanMessage(content="给老板发邮件请假")]}, config)
# 此时图暂停在 review 节点，interrupt() 还未返回

# 人工确认后恢复执行
result = graph.invoke(
    Command(resume="yes"),  # 这个值会作为 interrupt() 的返回值
    config=config  # 用同一个 thread_id 续传
)
print(result["messages"][-1].content)
```

**注意事项**

- `interrupt()` 是 LangGraph 1.x 推荐的方式，比 `interrupt_before/after` 更灵活
- 恢复执行时，必须用同一个 `thread_id` 的 config，且输入用 `Command(resume=value)` 而非普通 dict
- 暂停时 `invoke()` 会返回当前 state（暂停点的快照），可以在返回值中看到 interrupt 传入的值
- 如果用 `interrupt_before`，恢复时直接 `graph.invoke(None, config)` 传入 None 表示继续执行

---

## 3.2 interrupt（主动中断）

**概念详解**

`interrupt()` 是 LangGraph 1.x 引入的函数，用于在节点函数内部主动触发暂停。与 `interrupt_before/after`（在 compile 时固定暂停位置）不同，`interrupt()` 允许节点根据运行时状态动态决定是否暂停。

工作流程：节点函数执行到 `interrupt(value)` 时，图执行暂停，`value` 被传递给调用方（通过 invoke 的返回值）。调用方处理完后，用 `graph.invoke(Command(resume=answer), config)` 恢复执行，`interrupt()` 函数返回 `answer`，节点继续执行后续逻辑。

这种设计让"暂停-恢复"的逻辑内聚在节点函数内部，而不是分散在图的编译配置中。对于"有时候需要审核、有时候不需要"的场景特别有用——节点内部判断是否需要暂停，而不是固定每个实例都暂停。

**代码示例**

```python
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

def smart_agent_node(state: AgentState) -> dict:
    """智能判断是否需要人工介入"""
    last_message = state["messages"][-1]
    content = last_message.content

    # 如果涉及敏感操作关键词，暂停请求确认
    sensitive_keywords = ["删除", "发送邮件", "转账", "格式化"]
    if any(kw in content for kw in sensitive_keywords):
        user_choice = interrupt({
            "type": "confirmation",
            "message": f"检测到敏感操作请求：{content}",
            "options": ["确认执行", "取消操作"],
        })
        if user_choice == "确认执行":
            # 人工批准，继续执行
            response = llm.invoke(f"用户已确认执行以下操作：{content}")
        else:
            response_text = "操作已被用户取消"
            from langchain_core.messages import AIMessage
            return {"messages": [AIMessage(content=response_text)]}
    else:
        # 非敏感操作，直接执行
        response = llm.invoke(state["messages"])

    from langchain_core.messages import AIMessage
    return {"messages": [AIMessage(content=response.content)]}

# 使用
graph = builder.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "interrupt_demo"}}

# 非敏感操作：直接执行完毕
result = graph.invoke({"messages": [HumanMessage(content="今天天气怎么样")]}, config)

# 敏感操作：会暂停
config = {"configurable": {"thread_id": "interrupt_demo_2"}}
result = graph.invoke({"messages": [HumanMessage(content="帮我删除所有文件")]}, config)
# 图暂停，返回值中可以看到 interrupt 传入的值

# 恢复执行
result = graph.invoke(Command(resume="取消操作"), config)
```

**注意事项**

- `interrupt()` 只能在节点函数内部调用，不能在边/路由函数中调用
- 恢复时必须用 `Command(resume=value)` 作为 invoke 的输入，不是普通的 state dict
- `interrupt()` 的参数可以是任意 JSON 可序列化的值（dict、list、str 等）
- 如果图有多个 `interrupt()` 点，每次 invoke 只会停在第一个遇到的 interrupt，恢复后继续到下一个

---

## 3.3 多 Agent 架构总览

**概念详解**

当单个 Agent 的工具数量过多、或任务需要多种不同的专业能力时，单个 ReAct Agent 会变得低效——LLM 在太多工具中选择困难，或一个 Agent 难以兼顾所有领域。这时可以把多个 Agent 组成协作网络。

多 Agent 架构需要解决两个核心问题：路由（用户请求该交给哪个子 Agent）和通信（子 Agent 之间如何传递信息和结果）。

LangGraph 的图结构天然适合多 Agent 架构——每个子 Agent 可以是一个子图，主 Agent 或 Supervisor 负责路由。常见的两种模式是 Supervisor（一个主管统一调度）和 Hierarchical（多层调度）。

**代码示例（Supervisor 模式骨架）**

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

# ===== 子 Agent 1：搜索专家 =====
@tool
def web_search(query: str) -> str:
    """搜索互联网"""
    return f"搜索结果：{query}"

search_agent = create_react_agent(model=llm, tools=[web_search], prompt="你是搜索专家")

# ===== 子 Agent 2：计算专家 =====
@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    return str(eval(expression))

calc_agent = create_react_agent(model=llm, tools=[calculate], prompt="你是计算专家")

# ===== Supervisor：决定路由到哪个子 Agent =====
from pydantic import BaseModel
from typing import Literal

class Route(BaseModel):
    agent: Literal["search", "calc", "FINISH"]

def supervisor(state: dict) -> dict:
    """主管节点：分析用户请求，决定交给哪个子 Agent"""
    messages = state["messages"]
    # 让 LLM 决定路由
    structured_llm = llm.with_structured_output(Route)
    route = structured_llm.invoke([
        {"role": "system", "content": "根据用户请求决定交给哪个专家：search（搜索）/calc（计算）/FINISH（完成）"},
        *messages
    ])
    if route.agent == "FINISH":
        return {"messages": []}  # 不追加，结束
    return {"next_agent": route.agent}

# 构建图
class MultiAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    next_agent: str

builder = StateGraph(MultiAgentState)
builder.add_node("supervisor", supervisor)
builder.add_node("search", search_agent)
builder.add_node("calc", calc_agent)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", lambda s: s.get("next_agent", END),
    {"search": "search", "calc": "calc", END: END})
builder.add_edge("search", "supervisor")  # 子 Agent 完成后回到 supervisor
builder.add_edge("calc", "supervisor")

multi_agent = builder.compile()
```

**注意事项**

- 多 Agent 系统的 State 设计更复杂——需要考虑子 Agent 间共享哪些信息、如何避免上下文污染
- 子 Agent 完成后回到 Supervisor 是标准模式，让 Supervisor 决定是否需要再调另一个子 Agent
- `with_structured_output` 是让 LLM 输出结构化路由决策的好方法——百炼 qwen3.7-max 支持此功能
- 多 Agent 系统的 recursion_limit 通常需要设高一些，因为 Supervisor + 子 Agent 的循环会消耗较多步数

---

## 3.4 Supervisor 模式

**概念详解**

Supervisor 模式是最常见的多 Agent 架构。一个"主管"Agent（通常不绑定工具，只用 LLM 推理）负责接收用户请求，分析请求类型，决定交给哪个子 Agent 处理。子 Agent 执行完后，结果回到 Supervisor，Supervisor 决定是否需要再调其他子 Agent，或直接给用户返回最终答案。

优势：结构清晰，子 Agent 间解耦（各自独立，不直接通信），扩展性强（加新 Agent 只需加节点和路由选项）。

劣势：所有请求都经过 Supervisor，有额外的 LLM 调用开销；如果 Supervisor 的路由判断不准，会导致请求被送到错误的子 Agent。

**注意事项**

- Supervisor 通常用 `with_structured_output` 让 LLM 输出结构化路由决策（如 `{"agent": "search"}`），比解析文本更可靠
- 子 Agent 完成后应该回到 Supervisor 而非直接返回用户——Supervisor 需要判断是否完成、是否需要补充
- 多个同类型子 Agent（如 3 个搜索 Agent 并行搜索不同关键词）可以用 Send 机制实现扇出

---

## 3.5 Hierarchical 模式

**概念详解**

Hierarchical（层级）模式是 Supervisor 的递归嵌套：顶层 Supervisor 调度中层 Supervisor，中层 Supervisor 再调度底层执行 Agent。适合任务需要多级分解的复杂场景。

例如，一个"企业知识助手"可能这样分层：顶层 Supervisor 决定请求属于哪个部门（HR / 财务 / 技术），中层 HR Supervisor 再决定交给哪个 HR 子 Agent（招聘 / 薪酬 / 培训），底层薪酬 Agent 执行具体任务。

**注意事项**

- 层级不宜过深（通常 2-3 层），否则 LLM 路由的准确率会逐层下降
- 每一层都是用 LangGraph 子图实现的——子图作为节点嵌入父图
- 状态设计要分清哪些信息需要向上传递（如最终答案），哪些只在本层流转
- 层级越深，recursion_limit 需要越高

---

## 3.6 子图（Subgraph）

**概念详解**

子图是将一段完整的图流程封装为一个"超级节点"，嵌入更大的图中。子图有自己的 State、节点和边，编译后作为一个节点函数注册到父图中。

子图的核心价值是"封装和复用"：一个复杂的搜索+总结流程可以封装成子图，在多个场景中复用；多 Agent 系统中每个 Agent 本身就是一个子图。

**代码示例**

```python
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated

# ===== 子图：搜索 + 总结 =====
class SearchState(TypedDict):
    query: str
    search_result: str
    summary: str

def search_node(state: SearchState) -> dict:
    result = f"搜索 {state['query']} 的详细结果..."
    return {"search_result": result}

def summarize_node(state: SearchState) -> dict:
    summary = llm.invoke(f"请总结以下内容：{state['search_result']}").content
    return {"summary": summary}

search_subgraph_builder = StateGraph(SearchState)
search_subgraph_builder.add_node("search", search_node)
search_subgraph_builder.add_node("summarize", summarize_node)
search_subgraph_builder.add_edge(START, "search")
search_subgraph_builder.add_edge("search", "summarize")
search_subgraph_builder.add_edge("summarize", END)
search_subgraph = search_subgraph_builder.compile()

# ===== 父图：将子图作为节点 =====
class ParentState(TypedDict):
    user_query: str
    final_answer: str

def run_search_subgraph(state: ParentState) -> dict:
    # 调用子图
    result = search_subgraph.invoke({"query": state["user_query"]})
    return {"final_answer": result["summary"]}

parent_builder = StateGraph(ParentState)
parent_builder.add_node("search_and_summarize", run_search_subgraph)
parent_builder.add_edge(START, "search_and_summarize")
parent_builder.add_edge("search_and_summarize", END)
parent_graph = parent_builder.compile()

# 使用
result = parent_graph.invoke({"user_query": "LangGraph 的最新特性"})
print(result["final_answer"])
```

**注意事项**

- 子图的 State 类型不需要和父图一致——父图节点函数负责在父 State 和子 State 之间做转换
- 子图编译后就是一个可调用对象，在父图中就像普通节点函数一样使用
- 子图也可以有自己的 Checkpointer，独立于父图的持久化
- 如果需要在父图的 stream 输出中看到子图内部步骤，需要设置 `subgraphs=True`

---

## 3.7 并行执行

**概念详解**

LangGraph 支持节点并行执行。当一个节点的多条普通边指向不同节点时，这些目标节点会并行执行，各自完成后将结果通过 reducer 合并回 State。

并行执行的前提是目标节点之间没有数据依赖——如果 B 需要 A 的输出，就不能并行。如果 B 和 C 都只依赖 A 的输出且互不依赖，就可以并行。

并行节点的结果合并通过 State 的 reducer 完成。例如，如果 B 和 C 都返回 `{"results": [some_result]}`，且 `results` 字段用 `operator.add` 或 `add_messages` reducer，则两个结果会自动合并。

**代码示例**

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from operator import add

class ParallelState(TypedDict):
    query: str
    results: Annotated[list, add]  # 并行结果通过追加合并

def search_web(state: ParallelState) -> dict:
    return {"results": [f"Web 搜索结果：{state['query']}"]}

def search_db(state: ParallelState) -> dict:
    return {"results": [f"数据库查询结果：{state['query']}"]}

def search_cache(state: ParallelState) -> dict:
    return {"results": [f"缓存结果：{state['query']}"]}

def merge_results(state: ParallelState) -> dict:
    # 三个并行节点的结果已经在 results 列表中了
    combined = "\n".join(state["results"])
    answer = llm.invoke(f"综合以下信息回答问题：\n{combined}").content
    return {"results": [answer]}  # 追加最终答案

builder = StateGraph(ParallelState)
builder.add_node("web", search_web)
builder.add_node("db", search_db)
builder.add_node("cache", search_cache)
builder.add_node("merge", merge_results)

# 三路并行：START 同时指向 web/db/cache
builder.add_edge(START, "web")
builder.add_edge(START, "db")
builder.add_edge(START, "cache")

# 三路汇合：web/db/cache 都完成后执行 merge
builder.add_edge("web", "merge")
builder.add_edge("db", "merge")
builder.add_edge("cache", "merge")
builder.add_edge("merge", END)

graph = builder.compile()
result = graph.invoke({"query": "LangGraph 性能如何"})
```

**注意事项**

- 并行节点必须通过 reducer 合并结果——如果目标字段是默认覆盖策略，后执行完的节点会覆盖先执行完的
- 并行执行的节点数受 LangGraph 内部线程池限制，过多并行节点不会提升性能
- 并行节点内部不应该修改共享的全局变量——所有状态传递都通过 State

---

## 3.8 Send（动态路由）

**概念详解**

`Send` 机制允许条件边在运行时动态生成多个并行目标节点及各自独立的 state。与固定并行边不同，Send 的目标数量和每个目标的 state 内容都是在运行时才确定的。

典型场景是 map-reduce：一个节点产生 N 个子任务，每个子任务需要不同的输入 state 并行处理，最后汇总。比如"搜索 5 个关键词"——先产生 5 个 Send，每个 Send 携带不同的关键词到搜索节点并行执行，最后汇总。

**代码示例**

```python
from langgraph.types import Send
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from operator import add

class MapReduceState(TypedDict):
    topics: list[str]          # 原始主题列表
    results: Annotated[list, add]  # 汇总结果

class WorkerState(TypedDict):
    topic: str                 # 单个主题
    result: str

def fan_out(state: MapReduceState) -> list[Send]:
    """将主题列表扇出为多个并行任务"""
    # 为每个主题创建一个 Send，指定目标节点和独立 state
    return [
        Send("worker", {"topic": topic})
        for topic in state["topics"]
    ]

def worker(state: WorkerState) -> dict:
    """并行工作节点：处理单个主题"""
    result = llm.invoke(f"请用一句话解释：{state['topic']}").content
    # 注意：worker 的返回值会合并回父图的 state
    return {"results": [result]}

def fan_in(state: MapReduceState) -> dict:
    """汇总所有并行结果"""
    combined = "\n".join(state["results"])
    summary = llm.invoke(f"综合总结以下信息：\n{combined}").content
    return {"results": [summary]}

builder = StateGraph(MapReduceState)
builder.add_node("worker", worker)
builder.add_node("summarize", fan_in)

# fan_out 作为条件边，返回 Send 列表
builder.add_conditional_edges(START, fan_out)
# 所有 worker 完成后到 summarize
builder.add_edge("worker", "summarize")
builder.add_edge("summarize", END)

graph = builder.compile()
result = graph.invoke({"topics": ["LangGraph", "RAG", "Agent", "RabbitMQ", "Gradio"]})
```

**注意事项**

- `Send` 在 LangGraph 1.x 中从 `langgraph.types` 导入（旧版从 `langgraph.constants` 导入已废弃）
- Send 指定的目标节点（如 `"worker"`）必须已通过 `add_node` 注册
- Worker 节点的 State 类型可以和父图不同，但返回值会合并回父图的 State——字段名和 reducer 必须匹配
- Send 列表可以是空列表（相当于不扇出），但此时需要有一条到 END 的备用边

---

## 3.9 Command（指令对象）

**概念详解**

`Command` 是 LangGraph 1.x 引入的新 API，让节点函数可以同时声明"状态更新 + 下一个跳转的节点"，将路由逻辑内聚在节点内部，而不是分散到 `add_conditional_edges` 中。

传统方式下，节点的路由逻辑在 `add_conditional_edges` 的 router 函数中定义，与节点函数分离。用 Command 后，节点函数直接返回 `Command(update={...}, goto="next_node")`，更新和路由在一处。

Command 也用于 HITL 的恢复——`Command(resume=value)` 用于从 `interrupt()` 恢复执行。

**代码示例**

```python
from langgraph.types import Command
from langgraph.graph import StateGraph, START, END

class FlowState(TypedDict):
    step: str
    data: str

def step_a(state: FlowState) -> Command:
    data = process_a(state["data"])
    # 用 Command 同时更新状态和指定下一个节点
    return Command(
        update={"data": data, "step": "a_done"},
        goto="step_b"  # 跳转到 step_b
    )

def step_b(state: FlowState) -> Command:
    data = process_b(state["data"])
    if needs_retry(state):
        return Command(
            update={"data": data, "step": "b_retry"},
            goto="step_a"  # 回到 step_a 重试
        )
    return Command(
        update={"data": data, "step": "done"},
        goto=END  # 结束
    )

builder = StateGraph(FlowState)
builder.add_node("step_a", step_a)
builder.add_node("step_b", step_b)
builder.add_edge(START, "step_a")
# 注意：不需要 add_conditional_edges，路由在节点内部用 Command 声明
graph = builder.compile()
```

**注意事项**

- 使用 Command 的节点不需要（也不应该）为该节点添加 `add_edge` 或 `add_conditional_edges`——路由已在节点内部声明
- `Command` 的 `update` 参数和普通节点返回的 dict 作用相同——更新 state
- `goto` 可以是字符串（节点名）或 `END`，也可以是列表（并行跳转多个节点）
- 如果节点函数返回了 `Command`，就不要再返回 dict——两者是互斥的返回方式

---

## 3.10 流式输出的三个层级

**概念详解**

LangGraph 的流式输出从粗到细分为三个层级，满足不同场景的需求：

节点级（`stream_mode="updates"`）：每个节点执行完输出一次，格式为 `{"node_name": {"field": value}}`。最粗粒度，适合做进度条或日志——"正在执行节点 A → 正在执行节点 B"。数据量小，网络开销低。

消息级（`stream_mode="messages"`）：在 LLM 调用节点内部，以 token 为单位输出 LLM 生成的文本。适合做"打字机效果"的前端实时展示。用户看到 AI 正在逐字写回复，体验更好。返回的是 `(AIMessageChunk, metadata)` 元组。

事件级（`astream_events(version="v2")`）：输出图执行过程中的所有内部事件，包括 LLM 调用开始/结束、工具调用开始/结束、节点进入/退出等。最细粒度，适合做复杂的 UI 展示和调试。但事件量大，解析逻辑复杂。

**代码示例**

```python
from langchain_core.messages import HumanMessage

input_data = {"messages": [HumanMessage(content="用三句话解释量子计算")]}

# ===== 节点级流式 =====
print("--- 节点级 ---")
for chunk in graph.stream(input_data, stream_mode="updates"):
    for node, update in chunk.items():
        print(f"[{node}] 更新字段: {list(update.keys())}")

# ===== 消息级流式（打字机效果）=====
print("\n--- 消息级（打字机）---")
for chunk, metadata in graph.stream(input_data, stream_mode="messages"):
    # chunk 是 AIMessageChunk，content 是当前 token
    print(chunk.content, end="", flush=True)
print()

# ===== 事件级流式（异步）=====
import asyncio

async def stream_events():
    print("\n--- 事件级 ---")
    async for event in graph.astream_events(input_data, version="v2"):
        kind = event["event"]
        if kind == "on_chat_model_start":
            print(f"[LLM 开始] {event['name']}")
        elif kind == "on_chat_model_stream":
            # 逐 token 输出
            print(event["data"]["chunk"].content, end="", flush=True)
        elif kind == "on_tool_start":
            print(f"\n[工具开始] {event['name']}")
        elif kind == "on_tool_end":
            print(f"[工具结束] 输出: {event['data'].get('output', '')[:50]}")

asyncio.run(stream_events())

# ===== 组合模式：同时看节点级和消息级 =====
print("\n--- 组合模式 ---")
for chunk in graph.stream(input_data, stream_mode=["updates", "messages"]):
    # chunk 格式取决于哪个 stream_mode 产生了输出
    if "updates" in str(chunk):
        print(f"[节点更新] {chunk}")
    else:
        # messages 级的 token
        print(chunk, end="", flush=True)
```

**注意事项**

- `astream_events(version="v2")` 是异步方法，必须用 `async for` 消费，不能用于同步代码
- `stream_mode="messages"` 只在图中有 LLM 调用节点时才有输出——纯数据处理节点不会产生 messages 流
- 组合模式 `stream_mode=["updates", "messages"]` 返回的 chunk 格式不统一，需要根据 chunk 结构判断来源
- 生产环境推荐：节点级做进度展示，消息级做打字机效果，事件级只在调试时用

---

## 3.11 Subgraph Streaming

**概念详解**

默认情况下，父图通过 `stream()` / `astream()` 输出的流式数据只包含父图节点的执行信息，子图内部的状态流转对外不可见。

如果需要在父图的流式输出中看到子图内部的每一步更新，有两种方式：

1. `stream_mode` 传列表：如 `stream_mode=["updates", "messages"]` 可以同时看到父图和子图的节点级更新。

2. `subgraphs=True` 参数：`graph.stream(input, subgraphs=True)` 会输出嵌套结构，包含子图内部每步的更新。返回的 chunk 格式为 `(namespace, chunk)`，`namespace` 标识来自父图还是子图。

**代码示例**

```python
# 假设 graph 包含 search_subgraph 作为节点

# 方式一：stream_mode 列表（同时看父图和子图的消息级流式）
for chunk in graph.stream(
    input_data,
    stream_mode=["updates", "messages"]
):
    print(chunk)

# 方式二：subgraphs=True（显式开启子图流式）
for namespace, chunk in graph.stream(
    input_data,
    stream_mode="updates",
    subgraphs=True
):
    if namespace:
        # namespace 非空表示来自子图
        print(f"[子图 {namespace}] {chunk}")
    else:
        # namespace 为空表示来自父图
        print(f"[父图] {chunk}")
```

**注意事项**

- `subgraphs=True` 会增加流式输出的数据量，在子图层级深时尤为明显
- `namespace` 的格式是 tuple，如 `("search_subgraph:node_id",)`，可以据此判断来自哪个子图的哪个节点
- 如果不需要子图内部细节，不用开启 `subgraphs=True`，默认行为更简洁

---

# 四、工程实战层

## 4.1 State 设计原则

**概念详解**

State 设计是 LangGraph 项目中最考验功底的环节。好的 State 设计让图结构清晰、节点职责明确；差的 State 设计会导致数据冗余、reducer 冲突、难以调试。

核心原则：精简（只放真正需要在节点间传递的数据）、分清累加和覆盖（对话历史累加、临时变量覆盖）、复杂状态拆分（不要把所有字段塞一个 State，可以分层管理）。

**代码示例**

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages, MessagesState

# ===== 反面示例：什么都往 State 塞 =====
class BadState(TypedDict):
    messages: list                     # 忘了 reducer！对话历史会丢失
    temp_var: str                      # 临时变量不该放 State
    api_response_cache: str            # 缓存数据不该放 State
    user_input_processed: str          # 中间处理结果，只在两个节点间用

# ===== 正面示例：精简 + 正确 reducer =====
class GoodState(MessagesState):        # 继承 MessagesState 自动有 messages
    # 只放需要在多个节点间传递的字段
    user_name: str                     # 用户信息（覆盖策略即可）
    current_task: str                  # 当前任务（覆盖策略）
    search_results: Annotated[list, add]  # 搜索结果需要累加

# ===== 进阶：多 State 分层 =====
# 对话层 State（所有 Agent 共享）
class ConversationState(MessagesState):
    user_id: str

# 任务层 State（特定流程的临时状态）
class TaskState(TypedDict):
    task_id: str
    status: str                        # pending / running / done / failed
    retry_count: int                   # 覆盖策略
```

**注意事项**

- 临时变量（只在相邻两个节点间用的数据）如果必须放 State，考虑加一个 `scratchpad: str` 字段统一管理，而不是为每个临时变量建字段
- 对话历史 messages 字段几乎总是需要 `add_messages` reducer——这是最常见的新手 bug
- State 字段越多，图的调试越复杂——遵循 YAGNI 原则，初始只放必需字段，后续按需添加
- 在多 Agent 系统中，考虑每个子 Agent 用独立的 State 类型，通过父图节点做转换

---

## 4.2 Annotated 类型标注

**概念详解**

`Annotated` 是 Python 标准库 `typing` 模块提供的类型标注工具，LangGraph 用它来为 State 字段附加 reducer 信息。语法是 `Annotated[类型, reducer函数]`。

LangGraph 在编译图时会扫描 State 类型的 `__annotations__`，对带 `Annotated` 的字段提取 reducer 函数，对不带 `Annotated` 的字段使用默认覆盖策略。

**代码示例**

```python
from typing import TypedDict, Annotated
from operator import add
from langgraph.graph.message import add_messages

class State(TypedDict):
    # 1. 追加策略：LangGraph 内置的 message 专用 reducer
    messages: Annotated[list, add_messages]

    # 2. 追加策略：通用 list 拼接（operator.add = list.__add__）
    search_results: Annotated[list, add]

    # 3. 累加策略：数字累加（operator.add = int.__add__）
    total_score: Annotated[int, add]

    # 4. 覆盖策略：不加 Annotated
    current_question: str

    # 5. 自定义 reducer
    def keep_latest_if_not_none(current, new):
        """新值非 None 才覆盖，否则保持原值"""
        return new if new is not None else current

    cached_data: Annotated[str, keep_latest_if_not_none]
```

**注意事项**

- `Annotated` 的第一个参数是类型（不影响运行时行为，仅用于类型检查），第二个参数是 reducer 函数
- 多个元数据可以叠加：`Annotated[list, add_messages, "some_metadata"]`，LangGraph 只取第一个 callable 作为 reducer
- 忘记标注是最常见的 bug——如果对话历史莫名丢失，第一步检查 messages 字段是否有 `add_messages`
- `add_messages` 比简单的 list append 更智能：它会根据 message ID 去重和更新

---

## 4.3 错误处理

**概念详解**

LangGraph 图执行中，某个节点函数抛出未捕获的异常会中断整个图。这与普通 Python 函数调用链的行为一致——异常会沿调用栈向上传播，如果没有 try/except 捕获，程序崩溃。

处理方式有两种：

节点内部 try/except：在节点函数内部捕获异常，将错误信息写入 State（如 `{"error": str(e)}`），让后续节点可以看到错误并处理（如 LLM 根据错误信息调整策略重试）。适合可恢复的错误。

Fallback 节点：在条件边中检查 state 是否有 error 字段，有则路由到错误处理节点。错误处理节点可以做日志记录、通知用户、执行回滚等操作。适合需要专门处理流程的错误。

**代码示例**

```python
import traceback
from langchain_core.messages import AIMessage

def safe_llm_call(state: AgentState) -> dict:
    """节点内部 try/except 处理 LLM 调用异常"""
    try:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}
    except Exception as e:
        error_msg = f"LLM 调用失败：{type(e).__name__}: {e}"
        # 将错误信息作为 AI 消息返回，LLM 在下一轮可以看到
        return {
            "messages": [AIMessage(content=f"[系统提示] 发生错误：{error_msg}")],
            "error": error_msg,
        }

def error_handler(state: AgentState) -> dict:
    """专门的错误处理节点"""
    error = state.get("error", "未知错误")
    print(f"[错误处理] {error}")
    # 可以做日志记录、发送告警、执行回滚等
    return {
        "messages": [AIMessage(content=f"抱歉，处理过程中出现问题：{error}。请重试。")],
        "error": None,  # 清除错误状态
    }

# 构建带 fallback 的图
builder = StateGraph(AgentState)
builder.add_node("llm", safe_llm_call)
builder.add_node("error_handler", error_handler)
builder.add_edge(START, "llm")

# 条件边：有 error 就走错误处理，否则正常路由
def route_after_llm(state: AgentState) -> str:
    if state.get("error"):
        return "error_handler"
    return tools_condition(state)

builder.add_conditional_edges("llm", route_after_llm,
    {"error_handler": "error_handler", "tools": "tools", END: END})
builder.add_edge("error_handler", "llm")  # 错误处理完回到 LLM 重试
```

**注意事项**

- LLM API 调用最常见的异常是 RateLimitError（限流）和 APITimeoutError（超时），建议配合 Retry 策略使用
- 百炼 qwen3.7-max 的 `init_chat_model` 支持 `max_retries` 参数：`init_chat_model(..., max_retries=3)`
- 在节点内部 catch 异常时，不要吞掉异常（静默返回空结果）——应该将错误信息写入 state 让下游感知
- 多 Agent 系统中，子 Agent 的错误应该向上传递到 Supervisor，由 Supervisor 决定重试还是换一个子 Agent

---

## 4.4 Retry 策略

**概念详解**

LangGraph 本身不提供节点级的自动重试机制——如果节点函数抛异常，图就中断。重试需要在节点函数内部自行实现。

两种推荐方式：

1. LLM 层面重试：`init_chat_model` 支持 `max_retries` 参数，由底层 SDK 自动重试 API 调用失败（网络超时、限流等）。最简单，覆盖最常见的失败场景。

2. tenacity 库：更灵活的重试控制，支持指数退避、条件重试、重试回调等。适合需要精细控制重试策略的场景。

**代码示例**

```python
# 方式一：init_chat_model 内置 max_retries
from langchain.chat_models import init_chat_model

llm = init_chat_model(
    model="qwen3.7-max-2026-05-20",
    model_provider="openai",
    base_url=os.getenv("ALI_BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    api_key=os.getenv("ALI_BAILIAN_API_KEY"),
    max_retries=3,  # API 调用失败自动重试 3 次
)

# 方式二：tenacity 实现指数退避重试
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

@retry(
    stop=stop_after_attempt(5),                              # 最多重试 5 次
    wait=wait_exponential(multiplier=1, min=2, max=30),     # 指数退避：2s, 4s, 8s, 16s, 30s
    retry=retry_if_exception_type((TimeoutError, ConnectionError)),  # 只重试特定异常
)
def call_llm_with_retry(messages):
    """带指数退避的 LLM 调用"""
    return llm.invoke(messages)

def llm_node(state: AgentState) -> dict:
    response = call_llm_with_retry(state["messages"])
    return {"messages": [response]}
```

**注意事项**

- `max_retries=3` 对应的是底层 OpenAI SDK 的自动重试，主要处理 HTTP 5xx 和 429（限流）
- tenacity 的 `retry_if_exception_type` 要明确指定重试哪些异常——不要对所有异常都重试（如参数错误重试也没用）
- 指数退避（exponential backoff）是对待 API 限流的标准策略——立即重试只会加重限流
- 对于工具调用的重试，ToolNode 内部已经有异常处理（返回错误信息作为 ToolMessage），不需要额外重试

---

## 4.5 LangSmith 集成

**概念详解**

LangSmith 是 LangChain 生态的可观测性和调试平台。与 LangGraph 配合使用时，它能自动追踪图的每一步执行——包括每个节点的输入/输出、每次 LLM 调用的 prompt 和 response、每次工具调用的参数和结果、执行耗时等。

配置方式极简：设置两个环境变量 `LANGCHAIN_TRACING_V2=true` 和 `LANGCHAIN_API_KEY=xxx`，LangGraph 的所有执行就会自动上报到 LangSmith。不需要改一行代码。

在 LangSmith 面板上，你能看到图的执行流程以可视化方式呈现：节点间的流转路径、每步的 state 变化、LLM 调用的完整 prompt/response。这对调试 Agent 行为（如"为什么 LLM 选择了错误的工具"）非常有价值。

**代码示例**

```python
# .env 文件中添加：
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=lsv2_xxxxxxxxxxxxxxxx
# LANGCHAIN_PROJECT=my-langgraph-project  # 可选，默认 default 项目

# 或在代码中设置环境变量
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "lsv2_xxxxxxxxxxxxxxxx"
os.environ["LANGCHAIN_PROJECT"] = "my-langgraph-project"

# 之后的图执行会自动上报，无需改代码
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(model=llm, tools=[search])
result = agent.invoke({"messages": [HumanMessage(content="你好")]})
# 这次执行的所有细节都会出现在 LangSmith 面板上
```

**注意事项**

- LangSmith 有免费额度（5000 traces/月），对于个人学习和开发足够
- 如果不想上报某些敏感数据（如 API Key），可以在 invoke 时传 `run_name` 标记或用 `@chain` 装饰器的 `metadata` 过滤
- LangSmith 追踪会增加少量网络开销（异步上报），对性能影响极小但不是零
- 如果网络不通 LangSmith 服务，可以设 `LANGCHAIN_TRACING_V2=false` 关闭追踪

---

## 4.6 部署：LangGraph Server / Cloud

**概念详解**

LangGraph Server 是 LangChain 官方提供的部署方案，将编译后的图包装为生产级 HTTP 服务。它提供了 LangGraph 本身不包含的工程能力：持久化任务队列（异步执行长时间任务）、REST API（供前端调用）、SSE 流式输出、自动扩缩容、身份认证等。

使用方式有两种：LangGraph Cloud（托管服务，零运维）和 LangGraph Server（自托管 Docker 镜像）。两者 API 一致，区别在于运维责任。

**注意事项**

- LangGraph Server 需要 `langgraph-cli` 工具和 `langgraph.json` 配置文件定义图的入口
- 典型配置文件包含 graphs（图入口路径）、dependencies（依赖）、env（环境变量）等字段
- 对于学习阶段，自部署用 FastAPI 更简单直接（见 4.7），LangGraph Server 更适合正式生产
- 百炼 qwen3.7-max 的 API Key 通过环境变量注入，不要硬编码在配置文件中

---

## 4.7 自部署：FastAPI + LangGraph

**概念详解**

对于不想用 LangGraph Cloud 或有定制化需求的场景，可以用 FastAPI 将编译后的图包装为 HTTP 服务。核心要点：Checkpointer 用持久化后端、流式输出转为 SSE、thread_id 作为 API 参数传入。

**代码示例**

```python
# app.py — FastAPI + LangGraph 最小部署示例
import os
import sys
import uuid
from pathlib import Path
from dotenv import load_dotenv
from typing import Annotated, TypedDict

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent, ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
import json

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv(Path(__file__).parent / ".env")

# 初始化百炼模型
llm = init_chat_model(
    model="qwen3.7-max-2026-05-20",
    model_provider="openai",
    base_url=os.getenv("ALI_BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    api_key=os.getenv("ALI_BAILIAN_API_KEY"),
)

@tool
def search(query: str) -> str:
    """搜索互联网"""
    return f"搜索 {query} 的结果..."

# 创建 Agent（内存 Checkpointer，生产环境换 Sqlite/Postgres）
agent = create_react_agent(model=llm, tools=[search], checkpointer=MemorySaver())

app = FastAPI(title="LangGraph Agent API")

class ChatRequest(BaseModel):
    message: str
    thread_id: str = None  # 可选，不传则随机生成

@app.post("/chat")
async def chat(req: ChatRequest):
    """同步对话接口"""
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [HumanMessage(content=req.message)]},
        config=config
    )
    return {
        "reply": result["messages"][-1].content,
        "thread_id": thread_id,
    }

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE 流式对话接口"""
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    async def event_stream():
        async for chunk, metadata in agent.astream(
            {"messages": [HumanMessage(content=req.message)]},
            config=config,
            stream_mode="messages"
        ):
            data = json.dumps({"content": chunk.content, "done": False})
            yield f"data: {data}\n\n"
        yield f"data: {json.dumps({'content': '', 'done': True, 'thread_id': thread_id})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# 运行：uvicorn app:app --reload --port 8000
```

**注意事项**

- 生产环境务必将 `MemorySaver` 换成 `SqliteSaver` 或 `PostgresSaver`，否则重启后对话历史丢失
- SSE 流式接口前端用 `EventSource` 或 `fetch` 消费，注意跨域配置（FastAPI 加 `CORSMiddleware`）
- 多并发请求时，MemorySaver 的字典读写需要注意线程安全（SqliteSaver/PostgresSaver 自带事务保护）
- 可以加 `/health` 端点做健康检查、加 `/history/{thread_id}` 端点查看对话历史

---

## 4.8 异步执行

**概念详解**

LangGraph 的图可以同步或异步执行。异步模式下，节点函数定义为 `async def`，内部 LLM 调用用 `ainvoke` 而非 `invoke`。图的方法对应换为 `ainvoke` / `astream` / `astream_events`。

异步执行的核心优势是并发——当图有并行节点时，异步模式可以让多个节点真正并发执行（同步模式下并行节点是在线程池中模拟的）。在 FastAPI 等 async 框架中，异步执行不会阻塞事件循环，能处理更多并发请求。

百炼 qwen3.7-max 通过 `init_chat_model` 初始化后，同时支持同步（`invoke`）和异步（`ainvoke`）调用，不需要额外配置。

**代码示例**

```python
import asyncio
from langchain_core.messages import HumanMessage

# 异步节点函数
async def async_llm_node(state: AgentState) -> dict:
    # 用 ainvoke 而非 invoke
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}

# 异步执行图
async def main():
    # ainvoke：异步一次性执行
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="你好")]}
    )
    print(result["messages"][-1].content)

    # astream：异步流式执行
    async for chunk in agent.astream(
        {"messages": [HumanMessage(content="解释递归")]},
        stream_mode="updates"
    ):
        print(chunk)

    # astream_events：异步事件流
    async for event in agent.astream_events(
        {"messages": [HumanMessage(content="你好")]},
        version="v2"
    ):
        if event["event"] == "on_chat_model_stream":
            print(event["data"]["chunk"].content, end="", flush=True)

asyncio.run(main())
```

**注意事项**

- 异步图中可以混合同步和异步节点，但混用时同步节点会被包装在线程池中执行，有性能损失
- `create_react_agent` 创建的图默认同时支持同步和异步调用，不需要额外配置
- 在 Jupyter Notebook 中运行异步代码需要 `await` 直接调用或用 `nest_asyncio`（Jupyter 自带事件循环）
- 百炼 API 的异步调用底层走 `httpx.AsyncClient`，与同步的 `requests` 是不同的 HTTP 客户端

---

## 4.9 条件边的实现方式

**概念详解**

条件边有三种实现方式，从旧到新：

1. `add_conditional_edges(source, router_func, path_map)`：经典方式。router_func 接收 state 返回字符串（节点名），path_map 是可选映射字典。

2. `Command(goto="node")`：LangGraph 1.x 新方式。节点函数直接返回 Command 对象声明跳转目标，不需要 add_conditional_edges。

3. `Send(node, state)`：动态扇出方式。条件边返回多个 Send 对象，并行扇出到不同节点。

**代码示例**

```python
# ===== 方式一：add_conditional_edges（经典）=====
def router(state: AgentState) -> str:
    if state.get("needs_search"):
        return "search"
    elif state.get("needs_calc"):
        return "calc"
    return END

builder.add_conditional_edges("llm", router, {
    "search": "search_node",
    "calc": "calc_node",
    END: END
})

# ===== 方式二：Command（1.x 推荐）=====
from langgraph.types import Command

def llm_node(state: AgentState) -> Command:
    response = llm.invoke(state["messages"])
    if "搜索" in response.content:
        return Command(update={"messages": [response]}, goto="search_node")
    elif "计算" in response.content:
        return Command(update={"messages": [response]}, goto="calc_node")
    return Command(update={"messages": [response]}, goto=END)

# 不需要 add_conditional_edges，路由在节点内部声明

# ===== 方式三：Send（动态扇出）=====
from langgraph.types import Send

def fan_out_router(state: AgentState) -> list[Send]:
    keywords = extract_keywords(state["messages"][-1].content)
    return [Send("search_node", {"keyword": kw}) for kw in keywords]
```

**注意事项**

- `Command` 方式让路由逻辑内聚在节点内部，代码更清晰，LangGraph 1.x 推荐
- `add_conditional_edges` 仍然完全可用，旧代码不需要迁移
- `path_map` 参数虽然可选，但建议显式写出——它既是文档也是类型约束，能防止 router 返回未注册的节点名
- 一个节点只能用一种方式——要么用 `add_conditional_edges`，要么用 `Command`，不能混用

---

## 4.10 图的可视化

**概念详解**

LangGraph 编译后的图可以生成可视化输出，帮助理解图的结构。`graph.get_graph()` 返回图的内部结构表示，可以进一步调用 `draw_mermaid()` 生成 Mermaid 格式文本，或 `draw_mermaid_png()` 直接生成 PNG 图片。

Mermaid 格式可以在 Markdown、GitHub、Notion 等支持 Mermaid 的编辑器中直接渲染。PNG 输出需要安装 `grandalf` 或 `pygraphviz` 可选依赖。

**代码示例**

```python
# 生成 Mermaid 文本（可粘贴到 Markdown 中渲染）
mermaid_text = graph.get_graph().draw_mermaid()
print(mermaid_text)
# 输出示例：
# %%{init: {flowchart: {curve: linear}}}%%
# graph
#     __start__([__start__]):::first
#     llm(llm)
#     tools(tools)
#     __end__([__end__]):::last
#     __start__ -.-> llm
#     llm -.-> tools
#     llm -.-> __end__
#     tools -.-> llm

# 生成 PNG 图片（需要 grandalf 或 pygraphviz）
# pip install grandalf
png_bytes = graph.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png_bytes)

# 在 Jupyter 中直接显示
from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))
```

**注意事项**

- `draw_mermaid_png()` 依赖可选包，安装方式：`pip install grandalf`（纯 Python，无需编译）或 `pip install pygraphviz`（需要 Graphviz 系统依赖）
- 项目已安装 grandalf（在 langchain 相关依赖中包含），可以正常使用
- Mermaid 文本可以直接粘贴到 GitHub 的 Markdown 文件中，GitHub 会自动渲染
- 复杂的图（多 Agent 系统）生成的 Mermaid 可能很大，建议分段查看或用子图方式简化

---

# 五、百炼 qwen3.7-max 对接详解

## 5.1 初始化方式

**概念详解**

阿里云百炼（DashScope）提供 OpenAI 兼容模式 API，可以用 LangChain 的 `init_chat_model` 工厂函数通过 OpenAI 兼容方式接入。这种方式的好处是：不需要安装阿里云专用 SDK，用标准的 `langchain-openai` 包即可，切换其他 OpenAI 兼容模型（如 DeepSeek、Moonshot）只需改 model 名和 base_url。

**代码示例**

```python
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv(Path(__file__).parent / ".env")

llm = init_chat_model(
    model="qwen3.7-max-2026-05-20",       # 百炼模型标识（带日期版本号）
    model_provider="openai",              # 使用 OpenAI 兼容模式
    base_url=os.getenv("ALI_BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    api_key=os.getenv("ALI_BAILIAN_API_KEY"),
    temperature=0.7,                      # 可选，0=确定性输出，1=高随机性
    max_retries=3,                        # 可选，API 调用失败自动重试次数
)

# .env 文件内容：
# ALI_BAILIAN_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
# ALI_BAILIAN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

**注意事项**

- 模型名 `qwen3.7-max-2026-05-20` 是带日期版本号的标识，使用不带日期的 `qwen3.7-max` 也可以（自动指向最新版本）
- `model_provider="openai"` 不是说模型是 OpenAI 的，而是用 OpenAI 兼容协议接入百炼
- API Key 是敏感信息，必须在 .env 中配置，不要硬编码在代码中或提交到版本控制
- 项目中每个 chapter 目录下都有独立的 .env 文件，需要确保对应目录下有正确的 Key

---

## 5.2 绑定工具

**概念详解**

百炼 qwen3.7-max 支持 OpenAI 兼容的 function calling 协议。通过 `llm.bind_tools(tools)` 绑定工具后，LLM 回复中会包含结构化的 `tool_calls` 字段，而非纯文本回复。

`tool_calls` 是一个列表，每个元素包含：`name`（工具函数名）、`args`（参数字典，由 LLM 根据 schema 生成）、`id`（唯一调用标识，用于配对 ToolMessage）。

**代码示例**

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气。参数 city: 城市名称，如"北京"
    """
    weather_map = {"北京": "晴 25C", "上海": "多云 28C"}
    return weather_map.get(city, f"暂无 {city} 的天气数据")

@tool
def calculate(expression: str) -> str:
    """计算数学表达式。参数 expression: 数学表达式，如 '2+3*4'
    """
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误: {e}"

# 绑定工具
llm_with_tools = llm.bind_tools([get_weather, calculate])

# 测试
from langchain_core.messages import HumanMessage
response = llm_with_tools.invoke([HumanMessage(content="北京今天天气怎么样？")])

if response.tool_calls:
    for tc in response.tool_calls:
        print(f"工具: {tc['name']}")  # get_weather
        print(f"参数: {tc['args']}")  # {'city': '北京'}
        print(f"ID: {tc['id']}")      # call_xxxxx
else:
    print(f"直接回答: {response.content}")
```

**注意事项**

- `@tool` 装饰器从函数的类型注解和 docstring 自动生成 JSON Schema——docstring 的质量直接影响 LLM 调用工具的准确率
- 百炼 qwen3.7-max 可以在一条回复中生成多个 tool_calls（如同时查天气和算数），ToolNode 会全部执行
- 如果工具参数是必填的，在 docstring 中标注"必填"；可选参数标注"可选"
- 绑定的工具数量建议 < 20 个，过多会让 LLM 选择困难，降低准确率

---

## 5.3 流式输出

**概念详解**

百炼 qwen3.7-max 通过 `init_chat_model` 初始化后支持流式输出——LLM 逐 token 生成回复而非一次性返回完整文本。在 LangGraph 中，配合 `stream_mode="messages"` 可以让图的流式输出包含 LLM 的实时 token 流。

流式输出的核心价值是用户体验——用户看到 AI 正在"打字"而非长时间空白等待。在 Web 应用中，配合 SSE（Server-Sent Events）可以实现类似 ChatGPT 的实时打字机效果。

**代码示例**

```python
from langchain_core.messages import HumanMessage, SystemMessage

# 方式一：LLM 直接流式
print("=== LLM 直接流式 ===")
for chunk in llm.stream([HumanMessage(content="用三句话解释什么是 Agent")]):
    print(chunk.content, end="", flush=True)
print()

# 方式二：LLM 异步流式
import asyncio

async def async_stream():
    async for chunk in llm.astream([HumanMessage(content="什么是递归")]):
        print(chunk.content, end="", flush=True)
    print()

asyncio.run(async_stream())

# 方式三：LangGraph 图的 messages 模式流式
# 假设 agent 是用 create_react_agent 创建的图
print("=== 图流式 ===")
for chunk, metadata in agent.stream(
    {"messages": [HumanMessage(content="解释什么是 RAG")]},
    stream_mode="messages"
):
    # chunk 是 AIMessageChunk，content 是当前 token
    print(chunk.content, end="", flush=True)
print()

# 方式四：图异步流式 + SSE（配合 FastAPI）
# 参见 4.7 节的 /chat/stream 端点实现
```

**注意事项**

- `stream()` 返回的是 `AIMessageChunk` 对象（而非完整 AIMessage），`content` 属性是当前 token 的文本片段
- 流式输出的 chunk 之间可能有极短延迟（取决于百炼 API 的生成速度），但用户端看起来是连续的
- `stream_mode="messages"` 也会输出 ToolMessage 的内容——可以用 `metadata` 判断是 LLM token 还是工具结果
- 在前端配合 SSE 时，每个 token 发一个 SSE event，前端逐个追加到聊天窗口

---

## 5.4 环境变量

**概念详解**

百炼 qwen3.7-max + LangGraph 的完整环境变量配置。建议统一放在项目 `.env` 文件中，通过 `python-dotenv` 加载，避免硬编码和泄露。

**代码示例**

```python
# .env 文件内容（放在 code/chapterXX/ 目录下）

# ===== 必填：百炼模型 =====
ALI_BAILIAN_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ALI_BAILIAN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# ===== 推荐：LangSmith 追踪（调试用）=====
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
LANGCHAIN_PROJECT=langgraph-learning

# ===== 可选：其他模型（项目中有多个模型配置）=====
DEEP_SEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
DEEP_SEEK_BASE_URL=https://api.deepseek.com/v1
SILICON_API_KEY=sk-xxxxxxxxxxxxxxxx
SILICON_BASE_URL=https://api.siliconflow.cn/v1

# Python 代码中加载
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")  # 加载同目录 .env

# 验证关键变量
assert os.getenv("ALI_BAILIAN_API_KEY"), "请在 .env 中设置 ALI_BAILIAN_API_KEY"
```

**注意事项**

- 项目中每个 chapter 目录下有独立的 .env，内容可能相同——确保你要运行的 chapter 目录下有正确的 Key
- `.env.example` 文件是模板（Key 值为占位符），新建 chapter 时复制 .env.example 并填入真实 Key
- `LANGCHAIN_TRACING_V2=true` 开启后所有 LangChain/LangGraph 调用都会上报 LangSmith，生产环境注意隐私和费用
- 如果代码在多处调用 `load_dotenv`，重复调用无害——已加载的变量不会被覆盖（除非传 `override=True`）

---

# 六、学习路径建议

以下是从零到生产级 LangGraph 应用的推荐学习路径，每步约需 2-4 小时：

**第 1 步：手搭最简两节点图**

用 `StateGraph` + `add_node` + `add_edge` + `compile` 搭一个"输入 → LLM → 输出"的线性图。重点理解 State 定义、节点函数的返回值格式、invoke 的输入输出。这个练习让你建立"图 = 节点 + 边 + 状态"的直觉。

**第 2 步：手搭 ReAct Agent（不用 create_react_agent）**

用 `bind_tools` + `ToolNode` + `tools_condition` + 条件边手搭完整的 ReAct 循环。理解"LLM → 条件边 → 工具节点 → LLM"的循环结构。完成后与 `create_react_agent` 对比，理解快捷函数内部做了什么。

**第 3 步：加 Checkpointer 做多轮对话记忆**

用 `MemorySaver` + `thread_id` 实现多轮对话。重点理解 Checkpointer 的快照机制、thread_id 的会话隔离作用。测试两个 thread_id 的对话互不干扰。

**第 4 步：加 Human-in-the-loop 暂停**

用 `interrupt()` 在节点内部实现"工具执行前人工确认"。重点理解暂停-恢复的完整流程：`interrupt(value)` → 人工处理 → `Command(resume=answer)` 续传。

**第 5 步：尝试多 Agent Supervisor 架构**

搭一个 Supervisor + 2 个子 Agent 的多 Agent 系统。重点理解 Supervisor 的路由决策（`with_structured_output`）、子 Agent 的结果回到 Supervisor 的循环。

**第 6 步：配置 LangSmith + 异步执行**

设置 `LANGCHAIN_TRACING_V2=true`，在 LangSmith 面板查看图的执行 trace。将图的调用改为 `ainvoke` / `astream`，配合 FastAPI 搭建最简 API 服务。

**第 7 步：生产化部署**

将 MemorySaver 换成持久化 Checkpointer（SqliteSaver 或 PostgresSaver），用 FastAPI + SSE 搭建流式对话 API，加入错误处理和 Retry 策略，完成从开发到生产的闭环。

---

> 本文档基于 langgraph 1.2.10 + 百炼 qwen3.7-max-2026-05-20 编写，所有代码示例的 API 导入路径已在 python-agent-lab 项目 venv 中验证通过。
