# MCP（Model Context Protocol）学习总结

> MCP = Model Context Protocol（模型上下文协议）
>
> 核心目标：**让 AI 应用以统一、标准的方式连接外部工具、数据和能力。**

## 一、MCP 是什么？

MCP 是一种开放协议，用于标准化：

**AI 应用 ↔ 外部工具 / 数据 / 服务**

之间的连接方式。

可以把 MCP 理解成：

> **AI 世界里的“USB 接口”。**

```text
              ┌── Weather MCP
              │
AI Agent ─ MCP ├── GitHub MCP
              │
              ├── Database MCP
              │
              └── Filesystem MCP
```

---

## 二、为什么需要 MCP？

### 没有 MCP

LangChain 中可以直接定义 Tool：

```python
from langchain.tools import tool

@tool
def get_weather(city: str):
    ...
```

这种方式通常和当前应用耦合。如果 LangChain、Claude、Cursor、VS Code 等多个 AI 应用都需要同一个能力，就需要分别适配。

### 有了 MCP

可以把工具独立为 MCP Server：

```text
                MCP Protocol
                     │
       ┌─────────────┼─────────────┐
       ↓             ↓             ↓
   LangChain       Cursor        Claude
       │             │             │
       └─────────────┼─────────────┘
                     ↓
                MCP Server
                     ↓
                Weather API
```

MCP 将“工具的实现”和“使用工具的 AI 应用”解耦。

---

## 三、MCP 核心架构

```text
MCP Host / AI Application
        │
        │ MCP Client
        ↓
   MCP Protocol
        │
        ↓
   MCP Server
        │
   ┌────┼────┐
   ↓    ↓    ↓
 Tools Resources Prompts
```

例如：

```text
LangChain Agent
      │
      ↓
  MCP Client
      │
      ↓
 Weather MCP Server
      │
      ↓
 Weather API
```

---

## 四、MCP 的三个核心能力

```text
MCP
├── Tools
├── Resources
└── Prompts
```

### 1. Tools

Tool 表示 AI 可以调用、执行的能力。

例如：

```text
get_weather()
search_user()
query_database()
send_email()
create_order()
search_github()
```

Tool 更偏向：

```text
执行
操作
产生副作用
```

### 2. Resources

Resource 可以理解为向 AI 提供的数据或上下文资源。

例如：

```text
database://users
file://README.md
docs://api
config://system
```

简单区分：

```text
Tool
 ↓
做事情

Resource
 ↓
提供数据
```

### 3. Prompts

Prompt 是由 MCP Server 提供的可复用 Prompt 模板。

例如：

```text
review_code
```

可以对应一个代码审查模板：

```text
请分析下面代码：

{code}

重点检查：
1. 性能
2. 安全
3. 可维护性
4. Bug
```

---

## 五、MCP ≠ Function Calling

### Function Calling

主要解决：

> LLM 如何调用一个函数？

```text
LLM
 ↓
Function Schema
 ↓
程序
 ↓
执行函数
```

### MCP

解决：

> AI 应用如何标准化连接外部能力？

```text
LLM
 ↓
Agent
 ↓
MCP Client
 ↓
MCP Protocol
 ↓
MCP Server
 ↓
Tool / Resource / Prompt
 ↓
外部系统
```

因此：

```text
Function Calling
    ↓
函数调用机制

MCP
    ↓
AI ↔ 外部能力的标准通信协议
```

---

## 六、MCP 与 LangChain 的关系

### LangChain

解决：

> 如何开发 LLM / Agent 应用？

包括：

```text
LLM
Prompt
Tool
Agent
Memory
Runnable
Graph
```

### MCP

解决：

> 如何标准化连接外部能力？

关系可以理解为：

```text
             LangChain
                 │
                 │ MCP Client
                 ↓
             MCP Server
             /    |    \
            /     |     \
         Tool  Resource Prompt
```

MCP 不依赖 LangChain，LangChain 也不是 MCP。

---

## 七、MCP Tool 与 LangChain Tool

传统 LangChain Tool：

```python
from langchain.tools import tool

@tool
def get_weather(city: str):
    ...
```

MCP Tool：

```text
MCP Server
 ↓
MCP Tool
```

通过 `langchain-mcp-adapters` 可以转换成：

```text
MCP Tool
   ↓
langchain-mcp-adapters
   ↓
LangChain Tool
   ↓
Agent
```

因此：

> **MCP Tool 最终可以被 LangChain Agent 当成普通 Tool 使用。**

---

## 八、LangChain 接入 MCP

核心适配包：

```text
langchain-mcp-adapters
```

安装：

```bash
uv add langchain
uv add langgraph
uv add langchain-mcp-adapters
```

如果使用 OpenAI：

```bash
uv add langchain-openai
```

---

## 九、MultiServerMCPClient

LangChain 接入 MCP 最核心的 API：

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
```

创建 Client：

```python
client = MultiServerMCPClient(
    {
        "weather": {
            "transport": "stdio",
            "command": "python",
            "args": ["weather_server.py"],
        }
    }
)
```

获取 MCP Tools：

```python
tools = await client.get_tools()
```

核心转换链路：

```text
MCP Server
    ↓
MCP Tools
    ↓
MultiServerMCPClient
    ↓
LangChain Tools
```

---

## 十、stdio Transport

stdio 是本地 MCP 常见的通信方式。

```python
{
    "transport": "stdio",
    "command": "python",
    "args": ["weather_server.py"],
}
```

含义：

```text
LangChain
   │
   │ 启动进程
   ↓
python weather_server.py
   │
   ↓
MCP Server
```

通过 stdin / stdout 通信。

适合：

- 本地 MCP Server
- 开发环境
- CLI 工具
- 本地文件系统
- 本地数据库
- 桌面应用

---

## 十一、简单的 MCP Server

概念结构：

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather")


@mcp.tool()
def get_weather(city: str) -> str:
    return f"{city}：晴天，28°C"


if __name__ == "__main__":
    mcp.run()
```

服务器提供：

```text
get_weather
```

> 实际开发时，应以当前 MCP Python SDK 版本对应的 API 文档为准。

---

## 十二、LangChain 获取 MCP Tool

```python
from langchain_mcp_adapters.client import MultiServerMCPClient


client = MultiServerMCPClient(
    {
        "weather": {
            "transport": "stdio",
            "command": "python",
            "args": ["weather_server.py"],
        }
    }
)

tools = await client.get_tools()
```

得到：

```text
get_weather
```

转换关系：

```text
MCP Tool
    ↓
langchain-mcp-adapters
    ↓
LangChain Tool
```

---

## 十三、MCP Tool 接入 Agent

例如：

```python
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient


model = init_chat_model(
    "openai:gpt-5.5"
)

client = MultiServerMCPClient(
    {
        "weather": {
            "transport": "stdio",
            "command": "python",
            "args": ["weather_server.py"],
        }
    }
)

tools = await client.get_tools()

agent = create_agent(
    model=model,
    tools=tools,
)
```

调用：

```python
result = await agent.ainvoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "成都今天天气怎么样？"
            }
        ]
    }
)
```

---

## 十四、完整调用链

```text
User
 ↓
LangChain Agent
 ↓
判断需要天气工具
 ↓
LangChain Tool
 ↓
MCP Client
 ↓
MCP Protocol
 ↓
Weather MCP Server
 ↓
get_weather("成都")
 ↓
Weather API
 ↓
返回天气结果
 ↓
MCP Server
 ↓
MCP Client
 ↓
LangChain Agent
 ↓
LLM
 ↓
最终回答
```

---

## 十五、多个 MCP Server

MCP 可以同时连接多个 Server：

```text
Weather MCP
GitHub MCP
Database MCP
Search MCP
Filesystem MCP
```

配置思路：

```python
client = MultiServerMCPClient(
    {
        "weather": {
            ...
        },

        "github": {
            ...
        },

        "database": {
            ...
        },
    }
)
```

获取：

```python
tools = await client.get_tools()
```

可能得到：

```text
get_weather
search_github
create_issue
query_database
```

然后：

```python
agent = create_agent(
    model=model,
    tools=tools,
)
```

架构：

```text
                    Agent
                      │
        ┌─────────────┼─────────────┐
        ↓             ↓             ↓
   Weather MCP    GitHub MCP    Database MCP
        │             │             │
        ↓             ↓             ↓
   Weather API      GitHub         MySQL
```

---

## 十六、Remote MCP

MCP 也可以连接远程 Server：

```text
LangChain Agent
      │
      │ HTTP
      ↓
Remote MCP Server
      │
      ├── Tool
      ├── Resource
      └── Prompt
```

现代 MCP 学习重点应放在：

```text
Streamable HTTP
```

传统的：

```text
HTTP + SSE
```

已经属于旧的传输方式，不应作为新项目的主要学习重点。

---

## 十七、stdio vs Streamable HTTP

| 特性         | stdio  | Streamable HTTP |
| ------------ | ------ | --------------- |
| 本地         | ⭐⭐⭐⭐⭐  | ⭐⭐              |
| 远程         | ❌      | ⭐⭐⭐⭐⭐           |
| 网络服务     | ❌      | ✅               |
| 启动进程     | ✅      | ❌               |
| 开发简单     | ⭐⭐⭐⭐⭐  | ⭐⭐⭐             |
| Docker       | ⭐⭐⭐    | ⭐⭐⭐⭐⭐           |
| 微服务       | ⭐      | ⭐⭐⭐⭐⭐           |
| 远程生产服务 | 不适合 | 推荐            |

简单记：

```text
本地
 ↓
stdio

远程 / 服务端
 ↓
Streamable HTTP
```

---

## 十八、MCP 与 REST API

MCP 并不是 REST API 的替代品。

传统：

```text
Frontend
   ↓
REST API
   ↓
Backend
```

MCP：

```text
Agent
   ↓
MCP
   ↓
MCP Server
   ↓
REST API / Database / Service
```

例如：

```python
@mcp.tool()
def get_weather(city: str):

    response = requests.get(
        "https://weather-api.com",
        params={"city": city}
    )

    return response.json()
```

可以理解为：

> **MCP 将 REST API、数据库、文件系统等能力包装成 AI Agent 可以理解和调用的能力。**

---

## 十九、MCP 的跨语言能力

MCP 最大的优势之一是：

> **客户端和服务端不需要使用同一种编程语言。**

例如：

```text
Python LangChain
      │
      │ MCP
      ↓
TypeScript MCP Server
```

或者：

```text
Python Agent
      │
      │ MCP
      ↓
Go MCP Server
```

因为双方通过 MCP Protocol 通信。

---

## 二十、MCP Server 应该负责什么？

MCP Server 主要负责：

- 业务能力
- 数据访问
- API 调用
- 数据库操作
- 权限控制
- 参数校验

例如：

```text
search_users()
get_user()
search_orders()
create_order()
query_statistics()
```

---

## 二十一、Agent 应该负责什么？

Agent 主要负责：

- 理解用户
- 任务拆解
- 规划
- 推理
- 选择 Tool
- 多步执行
- 结果总结

合理架构：

```text
Agent
├── 理解用户需求
├── 规划任务
├── 选择 MCP Tool
├── 执行多步任务
└── 汇总结果

MCP Server
├── 查询数据库
├── 调用 API
├── 执行业务操作
└── 返回结果
```

不要把复杂 Agent 推理逻辑全部塞进 MCP Server。

---

## 二十二、MCP Tool 设计原则

核心原则：

> **一个 Tool 表达一个清晰能力。**

推荐：

```text
search_users()
get_user()
search_orders()
get_order()
create_order()
cancel_order()
```

不推荐：

```text
execute_business_operation(params: dict)
```

因为参数过于模糊，LLM 很难正确使用。

---

## 二十三、Tool Schema

Tool Schema 对 Agent 非常重要。

例如：

```python
get_weather(
    city: str
)
```

Agent 可以理解：

```text
Tool:
get_weather

参数:
city

类型:
string

必填:
yes
```

一个好的 Tool 应具有：

```text
name
description
input schema
output
```

尤其是：

```text
description
input schema
```

会直接影响 LLM 是否能够正确选择和调用 Tool。

---

## 二十四、Tool 的两种类型

### Read-only Tool

```text
search_user()
get_weather()
query_order()
search_github()
```

特点：

```text
只读
风险较低
```

### Side-effect Tool

```text
delete_user()
create_order()
send_email()
update_database()
deploy()
```

特点：

```text
修改数据
产生副作用
风险较高
```

推荐：

```text
读取操作
 ↓
可以自动执行

写入操作
 ↓
严格权限控制

危险操作
 ↓
Human-in-the-loop
```

---

## 二十五、MCP 安全

MCP Server 可能拥有：

```text
文件权限
数据库权限
API 权限
GitHub 权限
Shell 权限
邮件权限
```

所以：

> **MCP 本身不代表安全。**

生产环境需要考虑：

- Authentication
- Authorization
- 权限控制
- 参数校验
- 工具白名单
- 审计日志
- 限流
- 凭证管理
- Human-in-the-loop

尤其需要关注：

```text
delete
update
send
create
deploy
```

等有副作用的操作。

---

## 二十六、MCP Registry

MCP 生态中存在 MCP Server 的注册和发现机制。

可以类比：

```text
npm
 ↓
JavaScript Package

PyPI
 ↓
Python Package

MCP Registry
 ↓
MCP Server
```

可以发现各种 MCP Server，例如：

```text
GitHub MCP
Database MCP
Search MCP
Filesystem MCP
Documentation MCP
```

---

## 二十七、如何理解 MCP？

可以从四个角度理解：

### 1. USB

```text
MCP = AI 世界的 USB
```

### 2. 插件系统

```text
MCP Server = AI Plugin
```

### 3. 通信协议

```text
MCP = AI 应用连接外部能力的标准协议
```

### 4. Tool 标准

```text
MCP Tool
=
标准化的 AI Tool
```

---

## 二十八、MCP + LangChain 核心 API

重点记住：

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
```

创建：

```python
client = MultiServerMCPClient(...)
```

获取工具：

```python
tools = await client.get_tools()
```

接入 Agent：

```python
agent = create_agent(
    model=model,
    tools=tools,
)
```

核心链路：

```text
MultiServerMCPClient
        ↓
    get_tools()
        ↓
 LangChain Tools
        ↓
      Agent
```

---

## 二十九、LangChain / LangGraph / Tool / MCP 的区别

| 技术            | 主要解决的问题           |
| --------------- | ------------------------ |
| LLM             | 理解和生成自然语言       |
| LangChain       | 构建 LLM 应用            |
| LangGraph       | 编排复杂 Agent Workflow  |
| Tool            | Agent 调用函数/能力      |
| MCP             | 标准化连接外部能力       |
| MCP Client      | 连接 MCP Server          |
| MCP Server      | 对外提供 AI 可调用的能力 |
| Resource        | 提供数据/上下文          |
| Prompt          | 提供可复用 Prompt        |
| Streamable HTTP | 远程 MCP 通信方式        |
| stdio           | 本地 MCP 通信方式        |

---

## 三十、推荐学习路线

### 第一阶段：MCP 基础

掌握：

```text
MCP
MCP Client
MCP Server
Tool
Resource
Prompt
Transport
```

### 第二阶段：自己写 MCP Server

实现：

```text
Calculator MCP
```

提供：

```text
add()
subtract()
multiply()
divide()
```

### 第三阶段：LangChain 接 MCP

重点：

```python
MultiServerMCPClient
```

掌握：

```python
get_tools()
```

理解：

```text
MCP Tool
 ↓
LangChain Tool
 ↓
Agent
```

### 第四阶段：多个 MCP Server

实现：

```text
Weather MCP
Database MCP
Search MCP
```

### 第五阶段：远程 MCP

学习：

```text
Streamable HTTP
Authentication
Authorization
OAuth
API Key
Security
```

### 第六阶段：完整 Agent 项目

最终实现：

```text
                    AI Assistant
                         │
                  LangGraph Agent
                         │
          ┌──────────────┼──────────────┐
          ↓              ↓              ↓
     Weather MCP    Database MCP    Search MCP
          │              │              │
          ↓              ↓              ↓
      Weather API      MySQL          Web/API
```

例如用户：

> 查询成都天气，如果明天下雨，帮我查询数据库中成都地区的巡查任务，并统计未完成任务。

Agent：

```text
1. 调用 Weather MCP
2. 判断天气
3. 调用 Database MCP
4. 查询巡查任务
5. 统计未完成任务
6. 汇总结果
7. 返回用户
```

---

# 三十一、最终认知模型

```text
                         User
                           │
                           ↓
                    ┌─────────────┐
                    │    Agent    │
                    │  LangChain  │
                    │  LangGraph  │
                    └──────┬──────┘
                           │
                           ↓
                      MCP Client
                           │
             ┌─────────────┼─────────────┐
             │             │             │
             ↓             ↓             ↓
        Weather MCP    GitHub MCP    Database MCP
             │             │             │
             ↓             ↓             ↓
        Weather API      GitHub       MySQL
```

核心思想：

```text
LangChain
=
Agent 应用开发框架

LangGraph
=
复杂 Agent 工作流编排

Tool
=
Agent 可以调用的能力

MCP
=
AI 连接外部能力的标准协议

MCP Server
=
提供外部能力

MCP Client
=
连接 MCP Server

langchain-mcp-adapters
=
把 MCP Tool 接入 LangChain

MultiServerMCPClient
=
连接一个或多个 MCP Server
```

最终形成：

```text
Agent
  ↓
MCP Client
  ↓
MCP Protocol
  ↓
MCP Server
  ↓
Tool / Resource / Prompt
  ↓
API / Database / FileSystem / Service
```

> **一句话总结：**
>
> **MCP 不是 Agent，也不是 Function Calling；它是一套让 Agent / AI 应用能够以统一方式发现、访问和调用外部工具、数据和服务的标准协议。**