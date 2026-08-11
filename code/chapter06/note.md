# Chapter 06 - MCP（Model Context Protocol）

## 一、MCP 是什么？

**MCP（Model Context Protocol，模型上下文协议）** 是由 Anthropic 于 2024 年底开源的一套标准化协议，旨在为 AI 应用（Host）提供一种**统一的方式**来连接外部数据源和工具。

> 💡 **一句话理解**：MCP 就是 AI 世界的 "USB-C 接口" — 让任何 AI 模型都能通过统一协议即插即用地访问各种外部服务和数据。

### 核心价值

| 传统方式 | MCP 方式 |
|---------|---------|
| 每个工具需要单独编写适配代码 | 统一协议，一次接入，处处可用 |
| AI 与工具紧耦合 | AI 与工具松耦合，可互换 |
| 难以复用和共享集成方案 | 社区共建，开箱即用的 MCP Server 生态 |

### MCP 的定位

- **不是** 一个具体的工具或框架
- **是** 一套通信协议标准（类似 HTTP、WebSocket）
- 定义了 AI 应用（Client）与外部服务（Server）之间的**通信格式、能力发现、调用方式**

---

## 二、MCP 架构

### 2.1 核心角色

MCP 架构包含三个核心角色：

| 角色 | 说明 | 示例 |
|------|------|------|
| **Host（宿主）** | 发起连接的 AI 应用 | Claude Desktop、IDE 插件、自定义 Agent 应用 |
| **Client（客户端）** | Host 内部维护的协议客户端，与 Server 保持 1:1 连接 | MCP Client 实例 |
| **Server（服务端）** | 对外暴露能力（工具/资源/提示词）的轻量服务 | 天气 MCP Server、GitHub MCP Server |

### 2.2 架构图

```
                        ┌─────────────────────────┐
                        │          User            │
                        └────────────┬────────────┘
                                     │
                                     ▼
                        ┌─────────────────────────┐
                        │     Host / Agent         │
                        │  ┌───────────────────┐   │
                        │  │    LangChain /     │   │
                        │  │    LangGraph       │   │
                        │  └───────────────────┘   │
                        └────────────┬────────────┘
                                     │
                                     ▼
                        ┌─────────────────────────┐
                        │      MCP Client          │
                        │   (协议客户端层)           │
                        └────────────┬────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
                    ▼                ▼                ▼
           ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
           │ Weather MCP  │ │ GitHub MCP   │ │   DB MCP     │
           │   Server     │ │   Server     │ │   Server     │
           └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
                  │                │                │
                  ▼                ▼                ▼
           ┌──────────┐   ┌──────────┐    ┌──────────┐
           │Weather API│   │  GitHub  │    │  MySQL   │
           └──────────┘   └──────────┘    └──────────┘
```

### 2.3 通信流程

```
Agent                    MCP Client                MCP Server
  │                          │                          │
  │  1. "查天气"              │                          │
  │ ─────────────────────►   │                          │
  │                          │  2. 发现可用工具            │
  │                          │ ─────────────────────►   │
  │                          │  3. 返回工具列表            │
  │                          │ ◄─────────────────────   │
  │                          │                          │
  │  4. 调用 get_weather      │                          │
  │ ─────────────────────►   │                          │
  │                          │  5. tools/call            │
  │                          │ ─────────────────────►   │
  │                          │  6. 返回天气数据            │
  │                          │ ◄─────────────────────   │
  │  7. 格式化回复             │                          │
  │ ◄─────────────────────   │                          │
```

---

## 三、三大原语（Server Primitives）

MCP Server 通过三种**原语（Primitives）** 向 Client 暴露能力：

### 3.1 Tools（工具）

> **可执行的函数**，LLM 可以主动调用，通常有**副作用**。

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather-server")

@mcp.tool()
async def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    # 调用外部天气 API
    return f"{city} 今天晴，25°C"
```

**特点：**
- 由 **LLM 主动决定** 何时调用（Model-controlled）
- 类似 Agent 中的 Tool / Function Calling
- 可以有副作用（写数据库、发邮件、调用 API 等）
- 需要用户确认（安全敏感操作）

**Tool Annotations（工具注解）：**

| 注解 | 说明 |
|------|------|
| `readOnlyHint` | 是否只读（不修改数据） |
| `destructiveHint` | 是否具有破坏性（删除数据） |
| `idempotentHint` | 是否幂等（重复调用结果相同） |
| `openWorldHint` | 是否与外部世界交互 |

### 3.2 Resources（资源）

> **只读的数据源**，为 LLM 提供上下文信息。

```python
@mcp.resource("file://project/readme")
async def get_readme() -> str:
    """获取项目 README 文件内容"""
    with open("README.md", "r") as f:
        return f.read()

# 动态资源模板
@mcp.resource("db://users/{user_id}")
async def get_user(user_id: str) -> str:
    """根据用户 ID 查询用户信息"""
    return query_user_by_id(user_id)
```

**特点：**
- **只读**，不产生副作用
- 由 **应用程序控制**（Application-controlled），而非 LLM 主动调用
- 类似 RAG 中的上下文注入
- 使用 URI 模式标识（如 `file://`、`db://`、`api://`）

### 3.3 Prompts（提示词模板）

> **可复用的提示词模板**，为常见任务提供标准化的交互结构。

```python
@mcp.prompt()
def code_review(code: str, language: str = "python") -> str:
    """代码审查提示词模板"""
    return f"""请对以下 {language} 代码进行审查，关注：
1. 代码质量和可读性
2. 潜在的 bug 和安全问题
3. 性能优化建议

```{language}
{code}
```"""
```

**特点：**
- 由 **用户控制**（User-controlled），类似"斜杠命令"
- 定义标准化的交互模式
- 可以接受参数，动态生成提示词
- 适合封装复杂的、多步骤的提示逻辑

### 3.4 三大原语对比

| 特性 | Tools（工具） | Resources（资源） | Prompts（提示词） |
|------|:---:|:---:|:---:|
| **控制方** | LLM 主动调用 | 应用程序控制 | 用户触发 |
| **副作用** | ✅ 可以有 | ❌ 只读 | ❌ 无 |
| **类比** | Function Calling | RAG 上下文 | 斜杠命令 |
| **用途** | 执行操作 | 提供数据 | 标准化交互 |
| **示例** | 发邮件、查天气 | 读文件、查数据库 | 代码审查模板 |

---

## 四、传输方式（Transport）

MCP 支持两种主要传输方式：

### 4.1 Stdio（标准输入输出）

适用于**本地部署**，Client 将 Server 作为子进程启动。

```python
# Server 端启动
mcp.run(transport="stdio")
```

```json
// Client 配置示例（如 Claude Desktop）
{
  "mcpServers": {
    "weather": {
      "command": "python",
      "args": ["weather_server.py"]
    }
  }
}
```

**适用场景：** 本地 IDE 插件、桌面应用、开发调试

### 4.2 Streamable HTTP（流式 HTTP）

适用于**远程/云端部署**，2025 年 3 月引入，取代旧版 SSE。

```python
# Server 端启动
mcp = FastMCP("remote-server")

@mcp.tool()
async def hello(name: str) -> str:
    return f"Hello, {name}!"

mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)
```

**核心特点：**
- 单一 HTTP endpoint（`/mcp`）
- 使用标准 `POST`/`GET` 请求
- 可选使用 SSE 进行服务端推送
- **无状态设计**（2026 年 7 月规范）— 每个请求自描述
- 兼容负载均衡、Serverless 部署

### 4.3 传输方式对比

| 特性 | Stdio | Streamable HTTP |
|------|:---:|:---:|
| 部署位置 | 本地 | 远程/云端 |
| 启动方式 | 子进程 | 独立服务 |
| 网络要求 | 无 | 需要 HTTP 网络 |
| 扩展性 | 单机 | 水平扩展 |
| 认证方式 | OS 进程隔离 | OAuth 2.1 |
| 适用场景 | 开发调试、桌面应用 | 生产环境、多租户 |

### 4.4 传输方式演进

```
2024 年底   →   Stdio + HTTP+SSE（有状态，双端点）
2025.03    →   Stdio + Streamable HTTP（单端点，可选 SSE）
2026.07    →   完全无状态协议（请求自描述，无需 Session 握手）
```

> ⚠️ **注意**：旧版 `HTTP + SSE` 传输方式已被**官方弃用**，新项目应使用 Streamable HTTP。

---

## 五、HTTP 传输 + 认证（OAuth 2.1）

MCP 规范要求远程 Server 使用 **OAuth 2.1** 进行安全认证。

### 5.1 认证架构

```
┌──────────┐     1. 请求授权      ┌─────────────────┐
│MCP Client│ ──────────────────► │ Authorization    │
│          │ ◄────────────────── │ Server           │
│          │     2. 返回 Token    │ (OAuth 2.1)      │
└────┬─────┘                    └─────────────────┘
     │
     │  3. 携带 Bearer Token 调用
     ▼
┌──────────────┐
│  MCP Server  │
│ (Resource    │  4. 验证 Token → 返回结果
│  Server)     │
└──────────────┘
```

### 5.2 核心安全要求

| 要求 | 说明 |
|------|------|
| **强制 PKCE** | 所有授权流程必须使用 Proof Key for Code Exchange，防止授权码拦截攻击 |
| **Resource Indicators** | 客户端必须在请求中包含 `resource` 参数（RFC 8707），确保 Token 受众绑定 |
| **AS Discovery** | Server 必须通过 `/.well-known/` 端点暴露授权服务器元数据（RFC 8414） |
| **Protected Resource Metadata** | Server 暴露自身作为受保护资源的元数据（RFC 9728） |
| **禁用不安全流程** | 禁止 Implicit Flow 和 ROPC（Resource Owner Password Credentials） |

### 5.3 认证流程代码示例

```python
# Server 端 - 配置 OAuth 认证
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "secure-server",
    auth={
        "provider": "oauth2",
        "issuer": "https://auth.example.com",
        "audience": "https://mcp.example.com",
        "scopes": ["tools:read", "tools:execute"]
    }
)

@mcp.tool()
async def sensitive_operation(data: str) -> str:
    """需要认证的敏感操作"""
    return f"已处理: {data}"

mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)
```

```python
# Client 端 - 携带认证信息连接
from mcp.client.streamable_http import streamablehttp_client

async with streamablehttp_client(
    url="https://mcp.example.com/mcp",
    headers={"Authorization": "Bearer <access_token>"}
) as (read, write, _):
    # 已认证的 MCP 通信
    ...
```

### 5.4 不同传输的安全策略

| 传输方式 | 安全机制 | 说明 |
|---------|---------|------|
| **Stdio** | OS 进程隔离 | 通过环境变量传递凭据，进程边界提供安全 |
| **Streamable HTTP** | OAuth 2.1 | 强制使用标准化的授权流程 |

---

## 六、LangChain 接入 MCP

### 6.1 核心依赖

```bash
pip install langchain-mcp-adapters langgraph langchain-openai
```

`langchain-mcp-adapters` 是官方提供的适配层，将 MCP 的 Tools/Resources/Prompts 转换为 LangChain 可识别的格式。

### 6.2 接入方式一：Stdio 本地 Server

```python
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

async def main():
    # 配置 MCP Server（Stdio 方式）
    server_config = {
        "weather": {
            "transport": "stdio",
            "command": "python",
            "args": ["weather_server.py"],
        }
    }

    async with MultiServerMCPClient(server_config) as client:
        # 自动发现并加载所有工具
        tools = client.get_tools()

        # 创建 Agent
        llm = ChatOpenAI(model="gpt-4o")
        agent = create_react_agent(llm, tools)

        # 调用 Agent
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "北京今天天气怎么样？"}]}
        )
        print(result)

asyncio.run(main())
```

### 6.3 接入方式二：Streamable HTTP 远程 Server

```python
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

async def main():
    # 配置远程 MCP Server
    server_config = {
        "github": {
            "transport": "streamable_http",
            "url": "https://mcp.example.com/github/mcp",
            "headers": {
                "Authorization": "Bearer <your_token>"
            }
        }
    }

    async with MultiServerMCPClient(server_config) as client:
        tools = client.get_tools()

        llm = ChatOpenAI(model="gpt-4o")
        agent = create_react_agent(llm, tools)

        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "列出我最近的 GitHub 仓库"}]}
        )
        print(result)

asyncio.run(main())
```

### 6.4 接入方式三：多 Server 聚合

```python
async def main():
    # 同时连接多个 MCP Server
    server_config = {
        "weather": {
            "transport": "stdio",
            "command": "python",
            "args": ["weather_server.py"],
        },
        "github": {
            "transport": "streamable_http",
            "url": "https://mcp.example.com/github/mcp",
        },
        "database": {
            "transport": "stdio",
            "command": "python",
            "args": ["db_server.py"],
        }
    }

    async with MultiServerMCPClient(server_config) as client:
        # 自动聚合所有 Server 的工具
        tools = client.get_tools()  # 包含所有 Server 的工具

        llm = ChatOpenAI(model="gpt-4o")
        agent = create_react_agent(llm, tools)

        # Agent 可以跨 Server 调用工具
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "查询数据库中的用户列表，然后查看每个用户的 GitHub 仓库"}]}
        )
```

### 6.5 接入流程总结

```
┌────────────────────────────────────────────────────────────┐
│                    LangChain Agent                         │
│                                                            │
│   1. MultiServerMCPClient(config)   ← 配置 Server 信息     │
│   2. client.get_tools()             ← 自动发现工具          │
│   3. create_react_agent(llm, tools) ← 创建 Agent           │
│   4. agent.ainvoke(messages)        ← 执行任务              │
│                                                            │
│   适配层自动完成：                                           │
│   • MCP Tool → LangChain BaseTool 转换                     │
│   • 参数 Schema 映射（JSON Schema → Pydantic）              │
│   • 调用结果格式转换                                        │
└────────────────────────────────────────────────────────────┘
```

---

## 七、MCP 面试题精选

### Q1：什么是 MCP？它解决了什么问题？

**答：** MCP（Model Context Protocol）是 Anthropic 开源的一套标准化协议，用于统一 AI 应用与外部工具/数据的连接方式。

**解决的核心问题：** 在 MCP 出现之前，每个 AI 应用要对接外部服务（如数据库、API、文件系统），都需要编写独立的集成代码，形成 **M×N 的对接复杂度**。MCP 通过定义统一协议，将问题降低为 **M+N**：
- M 个 AI 应用只需实现 MCP Client
- N 个外部服务只需实现 MCP Server
- 任意组合即插即用

---

### Q2：MCP 的三大原语是什么？它们有什么区别？

**答：**

| 原语 | 控制方 | 是否有副作用 | 类比 |
|------|--------|:---:|------|
| **Tools** | LLM 主动调用 | ✅ | Function Calling |
| **Resources** | 应用程序控制 | ❌ 只读 | RAG 上下文注入 |
| **Prompts** | 用户触发 | ❌ | 斜杠命令模板 |

- **Tools** 是 Agent 的"手"，让 LLM 能执行动作
- **Resources** 是 Agent 的"眼睛"，让 LLM 能看到数据
- **Prompts** 是预设的"操作手册"，标准化常见交互

---

### Q3：MCP 与 Function Calling 有什么区别？

**答：**

| 维度 | Function Calling | MCP |
|------|-----------------|-----|
| **层级** | 模型能力 | 通信协议 |
| **范围** | 单个工具的定义与调用 | 完整的能力发现、调用、认证体系 |
| **标准化** | 各厂商格式不同 | 统一开放标准 |
| **生态** | 仅限单个应用 | 跨应用共享 Server |
| **能力** | 仅工具调用 | 工具 + 资源 + 提示词 |

> MCP **包含** Function Calling 的能力，但远不止于此。Function Calling 是"调用一个函数"，MCP 是"发现、连接、认证、调用一整套服务"。

---

### Q4：Stdio 和 Streamable HTTP 传输方式的区别？

**答：**
- **Stdio**：Client 将 Server 作为本地子进程启动，通过标准输入/输出通信。适合本地开发和桌面应用，简单但无法远程部署。
- **Streamable HTTP**：Server 作为独立 HTTP 服务运行，Client 通过 HTTP 请求通信。支持云端部署、负载均衡、Serverless，是生产环境推荐方式。

关键演进：2026 年 7 月规范实现了**完全无状态**设计，每个请求在 `_meta` 字段中自描述，无需 Session 握手，可以被任意服务器实例处理。

---

### Q5：MCP 如何保证远程调用的安全性？

**答：** MCP 规范强制要求远程 Server 使用 **OAuth 2.1** 认证：
1. **强制 PKCE** — 防止授权码拦截
2. **Resource Indicators（RFC 8707）** — Token 受众绑定，防止 Token 滥用
3. **Authorization Server Discovery** — 通过 `/.well-known/` 自动发现认证端点
4. **禁止不安全流程** — 移除 Implicit Flow 和 ROPC

对于本地 Stdio 传输，则依赖 OS 进程隔离和环境变量传递凭据。

---

### Q6：如何用 LangChain 接入 MCP Server？

**答：** 使用 `langchain-mcp-adapters` 库，核心步骤：

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

# 1. 配置 Server
config = {"my_server": {"transport": "stdio", "command": "python", "args": ["server.py"]}}

# 2. 连接并获取工具
async with MultiServerMCPClient(config) as client:
    tools = client.get_tools()  # MCP Tool → LangChain Tool 自动转换

    # 3. 创建 Agent 并调用
    agent = create_react_agent(llm, tools)
    result = await agent.ainvoke({"messages": [...]})
```

适配层自动完成 MCP Tool Schema → LangChain BaseTool 的转换。

---

### Q7：MCP Server 的三大原语分别适合什么场景？

**答：**

- **Tools** — 需要 Agent **执行动作** 的场景：
  - 调用外部 API（天气、搜索、发邮件）
  - 数据库写入操作
  - 文件系统操作（创建、修改、删除）

- **Resources** — 需要为 Agent **提供上下文** 的场景：
  - 将项目文档注入 LLM 上下文
  - 读取数据库记录作为参考
  - 提供配置信息或环境状态

- **Prompts** — 需要 **标准化用户交互** 的场景：
  - 代码审查模板
  - 报告生成模板
  - 固定格式的数据分析流程

---

### Q8：MCP 与 LangChain Tools 的关系是什么？

**答：** MCP Tools 和 LangChain Tools 是**不同层级**的概念：

- **LangChain Tool** 是框架内置的工具抽象，与特定 Agent 框架绑定
- **MCP Tool** 是协议级的工具定义，框架无关

通过 `langchain-mcp-adapters`，MCP Tool 可以被自动转换为 LangChain Tool，实现了**协议层与框架层的解耦**。这意味着：
- 同一个 MCP Server 可以同时被 LangChain、OpenAI Agents SDK、Claude 等不同框架使用
- 更换 Agent 框架时，MCP Server 无需修改

---

### Q9：MCP 的无状态设计有什么优势？

**答：** 2026 年 7 月的规范更新将 MCP 从有状态协议转变为**完全无状态**：

1. **水平扩展** — 任何服务器实例都能处理任何请求，无需 Session 亲和
2. **负载均衡** — 可以使用简单的 Round-Robin 负载均衡器
3. **Serverless 友好** — 服务器可以在空闲时关闭，无需维护长连接
4. **容错性** — 单个实例故障不影响整体服务
5. **运维简化** — 无需管理 Session 存储和同步

---

### Q10：如何自己开发一个 MCP Server？

**答：** 以 Python FastMCP 为例，最小实现：

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("my-server")

# 定义工具
@mcp.tool()
async def add(a: int, b: int) -> int:
    """计算两个数的和"""
    return a + b

# 定义资源
@mcp.resource("info://server/status")
async def get_status() -> str:
    """获取服务器状态"""
    return "running"

# 定义提示词模板
@mcp.prompt()
def summarize(text: str) -> str:
    """文本摘要模板"""
    return f"请用 3 句话总结以下内容：\n{text}"

# 启动（本地）
mcp.run(transport="stdio")

# 启动（远程）
# mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)
```

开发流程：定义能力（Tools/Resources/Prompts）→ 选择传输方式 → 配置认证（远程）→ 测试验证（MCP Inspector）→ 部署上线。
