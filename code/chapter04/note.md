# LangChain Tools 复习总结

> **核心关键词**：`Tool`、`@tool`、`Tool Calling`、`ToolRuntime`、`State`、`Context`、`Store`、`ToolNode`、`Command`、`MCP`
>
> **一句话**：LLM 负责"决定做什么"，Tool 负责"真正做什么"，Runtime 负责"告诉 Tool 当前环境"，LangGraph 负责"组织整个执行过程"。

---

## 一、Tool 是什么

Tool 是 Agent 连接外部世界的标准接口——LLM 本身不执行操作，而是产生 `tool_call`，由 Agent / LangGraph 执行对应 Tool。

```text
用户 → LLM → 判断是否需要 Tool → 生成 tool_call → Tool 执行 → 结果回传 LLM → 最终回答
```

典型能力：查询天气、查询数据库、调用 HTTP API、搜索互联网、读取文件、发送邮件等。

---

## 二、Tool 的核心组成

| 要素 | 说明 |
|---|---|
| **name** | Tool 名称，推荐 `snake_case`，避免中文/空格/连字符 |
| **description** | LLM 据此判断何时调用该 Tool，必须清晰、准确、描述实际能力 |
| **input schema** | 由类型注解自动生成，LLM 据此决定传什么参数 |
| **function** | 真正执行逻辑 |
| **return value** | 返回给 LLM 的结果（字符串 / 结构化数据 / `Command`） |

---

## 三、`@tool` 定义方式

最推荐的 Tool 定义方式：

```python
from langchain.tools import tool

@tool
def add(a: int, b: int) -> int:
    """计算两个数字的和。"""
    return a + b

# add.name / add.description / add.args_schema 均自动生成
```

**类型注解至关重要**——不仅用于 Python 类型检查，更是 Tool Calling Schema 的来源。LLM 依赖 Schema 决定如何调用 Tool。

可显式指定名称：`@tool("weather_query")`

---

## 四、复杂 Schema：Pydantic

参数较多/需要 `Field(description=...)` 时，使用 Pydantic 定义 Schema：

```python
from pydantic import BaseModel, Field

class WeatherInput(BaseModel):
    location: str = Field(description="城市名称")
    units: str = Field(default="celsius", description="温度单位")
    include_forecast: bool = Field(default=False, description="是否返回预报")

@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """查询天气。"""
    return "Sunny"
```

适用场景：参数较多、嵌套、有复杂约束、需要复用 Schema。

---

## 五、Tool 返回值

| 返回类型 | 适用场景 |
|---|---|
| `str` | 简单文本结果 |
| `dict` | 结构化业务数据（数据库 / API 返回） |
| `Command` | 不仅返回数据，还可修改 Agent State / 控制 Graph 流转 |

```python
from langgraph.types import Command

@tool
def set_language(language: str) -> Command:
    """设置当前对话语言。"""
    return Command(update={"language": language})
```

---

## 六、Tool Calling 流程

LLM **不直接执行** Tool，而是先生成调用指令：

```text
HumanMessage → LLM → AIMessage(tool_calls) → Tool 执行 → ToolMessage → LLM → 最终 AIMessage
```

LLM 产生的 `tool_call` 示例：

```json
{ "name": "get_weather", "args": { "city": "成都" } }
```

> **Tool Calling vs Function Calling**：Function Calling 是早期 Provider 层面的概念，Tool Calling 是现代更通用的概念。学习时聚焦 Tool Calling 即可。

---

## 七、Tool vs Structured Output

| | Structured Output | Tool |
|---|---|---|
| 核心目标 | 得到结构化数据 | 执行外部能力 |
| 执行函数 | ❌ | ✅ |
| Agent 核心能力 | 辅助 | 核心 |

简记：**Structured Output = LLM → 数据**，**Tool Calling = LLM → 动作**

---

## 八、`create_agent`

```python
from langchain.agents import create_agent

agent = create_agent(model=model, tools=[get_weather, search_database])
```

Agent 组成：`Model` + `Tools` + `Prompt` + `Runtime` + `Middleware`

---

## 九、ToolRuntime

Tool 与 Agent Runtime 之间的桥梁。`runtime` 参数 **不会暴露给 LLM**，由程序注入。

```python
from langchain.tools import tool, ToolRuntime

@tool
def get_user_info(runtime: ToolRuntime) -> str:
    """获取当前用户信息。"""
    return f"当前用户：{runtime.context.user_id}"
```

ToolRuntime 可访问的资源：

| 资源 | 说明 |
|---|---|
| **State** | 当前 Agent 的短期状态（消息、购物车等） |
| **Context** | 本次调用的运行时上下文（user_id、tenant_id 等） |
| **Store** | 跨会话的长期持久化数据（用户偏好、历史配置等） |
| stream_writer | 流式输出 |
| execution info | 执行元信息 |

---

## 十、State / Context / Store 对比

| 概念 | 含义 | 生命周期 | 示例 |
|---|---|---|---|
| **State** | 当前 Agent 执行状态 | 短期（单次对话） | `messages`、`cart`、`language` |
| **Context** | 调用时传入的运行时上下文 | 当前调用 | `user_id`、`tenant_id` |
| **Store** | 持久化数据 | 长期（跨会话） | 用户偏好、历史配置 |

Context 的重要性：身份、权限、租户隔离等 **不应由 LLM 生成**，应由后端通过 Context 注入。

```python
@tool
def query_order(order_id: str, runtime: ToolRuntime[UserContext]):
    # LLM 只控制 order_id，user_id/tenant_id 由程序控制
    user_id = runtime.context.user_id
    ...
```

---

## 十一、ToolNode 与 Agent Loop

`ToolNode` 是 LangGraph 中负责执行 Tool Calls 的底层组件：

```python
from langgraph.prebuilt import ToolNode
tool_node = ToolNode([get_weather, search_database])
```

经典 Agent 循环：

```text
LLM → 有 tool_calls? → Yes → ToolNode → Tool 执行 → 结果回传 LLM（循环）
                      → No  → END
```

`tools_condition` 负责判断 AIMessage 是否包含 `tool_calls`，实现路由：

```python
builder.add_conditional_edges("llm", tools_condition)
```

---

## 十二、Tool 来源生态

| 来源 | 说明 |
|---|---|
| **自定义 Tool** | `@tool` 自行编写 |
| **Toolkit** | 一组相关 Tool 的集合（如 `SQLDatabaseToolkit`） |
| **第三方 Tool** | Tavily、Exa（搜索）、Python REPL（代码）、GitHub / Slack 等 |
| **MCP Tool** | Model Context Protocol，标准化的外部 Tool 能力提供方式 |
| **Server-side Tool** | Provider 原生提供（Web Search、Code Interpreter），由 Provider 执行 |

```text
MCP Server → MCP Tools → LangChain Tools → Agent
```

---

## 十三、Tool 安全与工程实践

### 隐藏参数

LLM 可控制：业务参数（`order_id`、`query`）
程序控制：`user_id`、`tenant_id`、`permission`、`database`（通过 `ToolRuntime`）

### 错误处理

Tool 可能失败（API 超时 / 数据库异常 / 权限错误等），Agent 不应因此崩溃，需要 Error Handling。

### 并行调用

LLM 可同时产生多个 Tool Call 并行执行，需注意并发写 State 时的 Reducer / 状态冲突。

### 安全原则

```text
LLM → Tool → 参数验证 → 权限验证 → 业务验证 → 执行
```

重点关注：身份验证、权限控制、租户隔离、参数校验、SQL 注入、Prompt Injection、敏感数据保护。

---

## 十四、完整示例

```python
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime

@dataclass
class UserContext:
    user_id: str
    tenant_id: str

@tool
def get_order(order_id: str, runtime: ToolRuntime[UserContext]) -> str:
    """根据订单号查询当前用户的订单。

    Args:
        order_id: 订单编号。
    """
    user_id = runtime.context.user_id
    tenant_id = runtime.context.tenant_id
    # 实际项目：SELECT * FROM orders WHERE id = ? AND user_id = ? AND tenant_id = ?
    return f"订单 {order_id} 属于用户 {user_id}，租户为 {tenant_id}，当前状态：已发货"
```

涵盖：`@tool` + Tool Name + Description + 类型注解 + Runtime + Context + 数据隔离。

---

## 十五、知识体系总图

```text
                    LangChain Tools
                           │
            ┌──────────────┴──────────────┐
         Tool 定义                      Tool 执行
            │                             │
      ┌─────┼─────┐               ┌───────┴───────┐
    @tool Schema Description   create_agent     ToolNode
                                       │
                                  Tool Calling
                                       │
                                  ToolRuntime
                                       │
                          ┌────────────┼────────────┐
                        State       Context       Store
                          │
                       Command → 修改 Agent State

外部生态：自定义 Tool │ Toolkit │ MCP │ 第三方 Tool │ Server-side
```

---

## 十六、Tool 与 LangGraph 的关系

> Tool 是 Agent 的**能力**，LangGraph 是 Agent 工作流的**底层运行框架**。

```text
LangChain（Model / Prompt / Tool / Agent）
    ↓
LangGraph Runtime（State / Node / Edge / ToolNode / Command / Checkpoint）
```

---

## 十七、速查速记

| 概念 | 一句话 |
|---|---|
| Tool | 给 Agent 提供外部能力 |
| `@tool` | 把 Python 函数转换成 LangChain Tool |
| Schema | 告诉 LLM Tool 需要什么参数 |
| Description | 告诉 LLM 什么时候应该使用这个 Tool |
| Tool Calling | LLM 决定调用哪个 Tool，并生成参数 |
| ToolRuntime | Tool 获取 Agent 运行时信息的入口 |
| State | 当前 Agent 的短期状态 |
| Context | 当前调用携带的运行时上下文 |
| Store | 跨调用/跨会话的长期数据 |
| ToolNode | LangGraph 中负责执行 Tool 的节点 |
| Command | Tool 不仅返回数据，还可更新 State / 控制 Graph |
| Toolkit | 一组相关 Tool 的集合 |
| MCP | 标准化的外部 Tool / 能力提供方式 |

---

## 十八、推荐学习路线

1. **Tool 基础**：`@tool` → Name / Description → 类型注解 → Schema → Pydantic → 返回值
2. **Tool Calling**：`tool_calls` → AIMessage / ToolMessage → 多 Tool / 并行调用
3. **Agent**：`create_agent`（Model + Tools + Prompt + Runtime）
4. **Runtime**：ToolRuntime → State vs Context vs Store
5. **LangGraph**：ToolNode → `tools_condition` → Command → Reducer
6. **实战 Tool**：Weather → HTTP API → Database → Search → RAG → File
7. **高级**：Error Handling / Retry / Timeout / Streaming / Human-in-the-loop / Dynamic Tools / MCP