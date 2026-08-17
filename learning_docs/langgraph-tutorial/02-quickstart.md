# 第 2 章：环境准备与第一个应用

## 2.1 安装

```bash
# 核心：只要编排能力
pip install langgraph

# 推荐：加上 LangChain 生态（init_chat_model、create_react_agent 都需要）
pip install langgraph langchain langchain-openai

# 用 Claude 的话
pip install langgraph langchain langchain-anthropic

# 持久化相关（按需）
pip install langgraph-checkpoint-sqlite    # SQLite checkpointer
pip install langgraph-checkpoint-postgres  # Postgres checkpointer
```

验证安装：

```bash
python -c "import langgraph; print(langgraph.__version__)"
```

## 2.2 模型与 API Key

本教程统一用 `init_chat_model` 初始化模型——一个入口切换所有供应商：

```python
from langchain.chat_models import init_chat_model

# OpenAI
llm = init_chat_model("openai:gpt-4.1")

# Anthropic
llm = init_chat_model("anthropic:claude-sonnet-4-5")

# 国产/兼容 OpenAI 协议的服务（如 DeepSeek、通义、智谱）
import os
os.environ["OPENAI_API_KEY"] = "sk-xxx"
os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com/v1"  # 以 DeepSeek 为例
llm = init_chat_model("openai:deepseek-chat")
```

环境变量建议放在 `.env` 文件中：

```bash
# .env
OPENAI_API_KEY=sk-xxx
LANGSMITH_TRACING=true        # 开启 LangSmith tracing（强烈建议）
LANGSMITH_API_KEY=lsv2-xxx
```

```python
# 在代码入口加载 .env
from dotenv import load_dotenv
load_dotenv()
```

## 2.3 Hello World：不依赖 LLM 的第一张图

先用纯 Python 逻辑理解图的骨架（不花一分钱 token）：

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


# 1. 定义状态：整个图共享的数据结构
class State(TypedDict):
    count: int
    message: str


# 2. 定义节点：普通函数，接收 state，返回「部分更新」
def add_one(state: State) -> dict:
    return {"count": state["count"] + 1}


def make_message(state: State) -> dict:
    return {"message": f"当前计数：{state['count']}"}


# 3. 组装图
builder = StateGraph(State)
builder.add_node("add_one", add_one)
builder.add_node("make_message", make_message)

builder.add_edge(START, "add_one")            # 入口 → add_one
builder.add_edge("add_one", "make_message")   # add_one → make_message
builder.add_edge("make_message", END)         # make_message → 结束

graph = builder.compile()

# 4. 运行
result = graph.invoke({"count": 0})
print(result)
# {'count': 1, 'message': '当前计数：1'}
```

这就是 LangGraph 的全部骨架：**定义状态 → 定义节点 → 连边 → compile → invoke**。

## 2.4 第一个聊天机器人（LLM + 循环）

下面加入 LLM 和一个"是否继续对话"的简单循环，感受图真正的力量：

```python
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

llm = init_chat_model("openai:gpt-4o-mini")


# add_messages 是内置 reducer：新消息 append 而不是覆盖
class State(TypedDict):
    messages: list  # 完整写法：Annotated[list, add_messages]，见第 3 章


def chatbot(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}  # reducer 决定它是「追加」


builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)
graph = builder.compile()

result = graph.invoke({"messages": [{"role": "user", "content": "你好，介绍一下你自己"}]})
print(result["messages"][-1].content)
```

> 注：上面为了入门简化了 reducer 写法。`messages` 字段要正确追加必须用 `Annotated[list, add_messages]`（或直接用内置 `MessagesState`），第 3 章会严格展开。本章先跑通直觉。

## 2.5 加上"记忆"：三行代码获得持久化

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "user-001"}}  # 一个 thread = 一段会话

# 第一轮
graph.invoke({"messages": [{"role": "user", "content": "我叫小明，我喜欢爬山"}]}, config)

# 第二轮：新的一句输入，但同一个 thread —— 它记得你
result = graph.invoke({"messages": [{"role": "user", "content": "我叫什么名字？"}]}, config)
print(result["messages"][-1].content)  # "你叫小明"
```

发生了什么：checkpointer 在**每一步之后**把整个状态存档。第二次 invoke 时，LangGraph 找到 `thread_id=user-001` 的最新存档，把你的新消息合并进去再执行。这就是"对话记忆"的本质——不是魔法，是检查点。

## 2.6 开启 LangSmith，看见图的每一步

设置环境变量后无需改任何代码：

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=lsv2-xxx
```

再运行上面的例子，打开 [smith.langchain.com](https://smith.langchain.com)，你会看到：
- 每个 `invoke` 是一个 trace
- trace 里每个节点是一条 span，包含输入、输出、耗时、token 数

**学 LangGraph 强烈建议全程开着 tracing**，它是理解"图到底怎么跑"的最佳工具。

## 2.7 可视化你的图

```python
# 生成 mermaid 描述（文本，任何环境可用）
print(graph.get_graph().draw_mermaid())

# 生成 PNG（需要 pip install pymermaid 或使用网络渲染，见第 6 章）
# png_bytes = graph.get_graph().draw_mermaid_png()
```

## 2.8 常用开发工具链一览

| 工具 | 用途 | 何时用 |
|---|---|---|
| `graph.invoke()` / `.stream()` | 直接运行 | 学习、单测 |
| `langgraph dev` | 本地起 LangGraph Server | 集成前端、调试 HITL（第 23 章） |
| LangGraph Studio | 浏览器可视化调试 | 看图、回放、时间旅行 |
| LangSmith tracing | 生产观测 | 永远开着 |

## 本章小结

- 安装：`pip install langgraph langchain langchain-openai`
- 图四步曲：State → nodes → edges → compile
- `thread_id` + checkpointer = 对话记忆
- 全程开 LangSmith tracing 学习效率翻倍

> 下一章进入全书最重要的部分：State。理解了 State 的读写规则，LangGraph 就懂了 80%。
