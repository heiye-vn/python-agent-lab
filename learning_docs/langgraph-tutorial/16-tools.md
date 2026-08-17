# 第 16 章：工具 Tools 深度解析

工具是 Agent 的手脚。本章讲透 LangChain/LangGraph 体系的工具体系：定义、绑定、执行（ToolNode）、错误处理、隐藏参数注入——这些细节决定 Agent 的可靠性。

## 16.1 定义工具：@tool 装饰器

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气。

    Args:
        city: 城市名，如 "上海"、"Beijing"
    """
    return fetch_from_weather_api(city)
```

**铁律（模型只认识两样东西）**：
1. 函数名（要动词化、见名知义：`get_weather` 而非 `weather_tool`）
2. docstring + 参数注解（这就是模型的"工具说明书"）——写得越清楚，调用越准

`args_schema` 自动从类型注解生成，也可显式用 Pydantic 控制：

```python
from pydantic import BaseModel, Field

class SearchArgs(BaseModel):
    query: str = Field(description="搜索关键词，中英文均可")
    top_k: int = Field(default=5, ge=1, le=20, description="返回条数")

@tool(args_schema=SearchArgs)
def search(query: str, top_k: int = 5) -> str:
    """联网搜索最新信息。"""
    ...
```

其他定义方式：`StructuredTool.from_function(func)`、Runnable 直接转 `ToolNode` 兼容、甚至任意 `{"name","description","schema"}` 格式（如 OpenAI 格式、MCP 工具）都可直接绑定。

## 16.2 绑定与调用消息流

```python
llm_with_tools = llm.bind_tools([get_weather, search])

response = llm_with_tools.invoke([("user", "上海今天热吗？")])
response.tool_calls
# [{'name': 'get_weather', 'args': {'city': '上海'}, 'id': 'call_xxx', 'type': 'tool_call'}]
```

Agent 循环中的消息协议（务必记住，手写 Agent 全靠它）：

```
HumanMessage("上海天气") 
→ AIMessage(tool_calls=[{get_weather, {city:上海}, id}])     # 模型要调工具
→ ToolMessage(content="25°C 晴", tool_call_id=id)            # 执行结果回填
→ AIMessage("上海今天 25 度，晴。")                            # 模型基于结果回答
```

## 16.3 ToolNode：官方工具执行器

```python
from langgraph.prebuilt import ToolNode

tool_node = ToolNode([get_weather, search])

builder.add_node("agent", call_model)        # 模型节点：bind_tools + invoke
builder.add_node("tools", tool_node)         # 工具节点：解析 tool_calls 并执行

builder.add_conditional_edges("agent",
    lambda s: "tools" if s["messages"][-1].tool_calls else END)
builder.add_edge("tools", "agent")
```

ToolNode 内置能力：
- 并行执行同一消息里的**多个 tool_calls**
- 参数校验失败/工具抛异常 → 自动把 traceback 作为 ToolMessage 返回给模型（模型会自我纠正）
- `handle_tool_errors=True/False/自定义函数` 控制错误行为

```python
tool_node = ToolNode(
    [get_weather],
    handle_tool_errors=lambda e: f"工具暂时不可用，请稍后重试或换一种方式。原始错误：{e}",
)
```

**"把错误还给模型"是 Agent 设计的重要哲学**：与其崩溃，不如让模型看到错误自己想办法。

## 16.4 隐藏参数：InjectedState 与 InjectedToolArg

有些参数不想（也不该）让模型填，而是运行时注入：

```python
from langgraph.prebuilt import InjectedState, InjectedToolArg
from langchain_core.tools import tool
from langgraph.config import get_config

@tool
def query_orders(
    state: Annotated[dict, InjectedState("messages")],  # 注入图状态（模型看不到此参数）
    tool_call_id: Annotated[str, InjectedToolArg],      # 完全隐藏的参数
    order_id: str,                                       # 模型填的正常参数
) -> str:
    """根据订单号查询订单状态。"""
    user_id = get_config()["configurable"]["user_id"]    # 运行时上下文注入
    return db.get_order(user_id, order_id)
```

- `InjectedState`：把图的 State（可指定字段）作为参数传给工具，但**不进模型说明书**（模型不知道也填不了）
- `InjectedToolArg`：ToolNode 自动填充（如 tool_call_id），模型不可见
- 权限类信息（user_id、tenant_id）**永远从 config/store 注入，绝不交给模型填**——这是安全底线

## 16.5 工具返回值工程学

返回 `str` 最稳（所有模型都吃）。需要结构化时返回 JSON 字符串并控制长度：

```python
@tool
def search_orders(keyword: str) -> str:
    """搜索订单，返回 JSON 数组。"""
    rows = db.search(keyword, limit=10)          # 1. 限量
    slim = [{"id": r.id, "status": r.status, "amount": r.amount} for r in rows]
    return json.dumps(slim, ensure_ascii=False)   # 2. 只留必要字段
    # 大结果集 → 返回总数 + 前 N 条 + "用 search_orders_page 翻页"
```

多模态工具（返回图片给多模态模型）：返回 `list[ContentBlock]` 格式（`{"type": "image_url", ...}`），ToolNode 原样透传。

## 16.6 工具管理的进阶模式

### 动态启用工具（按权限/租户）

```python
def call_model(state, config):
    user = config["configurable"]["user"]
    tools = base_tools if user.tier == "free" else base_tools + pro_tools
    return {"messages": [llm.bind_tools(tools).invoke(state["messages"])]}
```

### 工具数量太多？

模型说明书塞 50 个工具会显著降低准确率。方案：
- **分组路由**：前置一个 router LLM 选择本次启用的工具子集（成本换准确）
- **MCP**：统一协议管理外部工具（第 19 章）
- 合并"细粒度 CRUD 工具"为一个"带操作参数"的工具

### 测试工具

```python
def test_get_weather():
    assert "25" in get_weather.invoke({"city": "上海"})   # 单测工具函数本身
    # 集成测：用 LangSmith 数据集跑"用户问题 → 期望调用的工具"（第 27 章）
```

## 16.7 手写一个完整 ReAct Agent（理解原理）

后面会讲 `create_react_agent` 一行搞定，但**手写一遍是理解 Agent 的最佳方式**：

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver

tools = [get_weather, search]
llm_with_tools = llm.bind_tools(tools)

def call_model(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

def should_continue(state):
    return "tools" if state["messages"][-1].tool_calls else END

builder = StateGraph(MessagesState)
builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, ["tools", END])
builder.add_edge("tools", "agent")

agent = builder.compile(checkpointer=InMemorySaver())

result = agent.invoke(
    {"messages": [("user", "上海和北京今天哪个更热？")]},
    config={"configurable": {"thread_id": "1"}},
)
print(result["messages"][-1].content)
```

这个 30 行的骨架就是 ReAct：**模型 →（有 tool_calls？）→ 工具 → 模型 → … → 无 tool_calls → END**。`create_react_agent` 是它的官方封装。

## 本章小结

- 工具 = 函数名 + docstring + 参数注解；说明书写给模型看，精度全靠它
- 消息协议：AIMessage(tool_calls) → ToolMessage(tool_call_id) 循环
- ToolNode：并行执行、错误回传模型、支持隐藏参数注入
- 权限信息永远运行时注入（InjectedState / config），不交给模型
- 工具多到影响准确率：分组路由 / 合并 / MCP

> 下一章：create_react_agent——官方预构建 Agent 的完整参数手册。
