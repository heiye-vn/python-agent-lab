# 第 19 章：上下文工程与 MCP 集成

本章讲两件决定 Agent 上限的事：**上下文工程**（往模型窗口里放什么）与 **MCP**（工具生态的标准协议）。

## 19.1 上下文工程：Agent 时代的"特征工程"

模型能力固定的前提下，**上下文质量决定输出质量**。一个 Agent 的上下文构成：

```
[system prompt] [长期记忆] [对话历史(裁剪后)] [检索/工具结果] [当前任务说明]
```

上下文工程的四条军规：

### 军规一：预算思维

不同内容按"每 token 价值"分配：

| 内容 | 价值密度 | 策略 |
|---|---|---|
| system prompt / 指令 | 极高 | 常驻，精写 |
| 长期记忆摘要 | 高 | 每次注入，限 top-k 条 |
| 最近对话 | 高 | 滑窗保留（trim） |
| 检索文档 | 中 | 只留最相关 N 段，做引用编号 |
| 工具结果 | 中 | 限长、截断、结构化 |
| 远期历史 | 低 | 压缩成摘要（第 11 章） |

### 军规二：动态 system prompt

静态提示是原型做法。生产级 prompt 是**函数**（第 17 章 17.3）：注入日期、用户身份、租户配置、记忆——让模型每次都在"最新世界"里工作。

### 军规三：工具结果的"再加工"

原始工具输出（整个网页、全表数据）直接塞上下文是灾难：

```python
@tool
def web_fetch(url: str) -> str:
    """抓取网页并返回正文摘要。"""
    html = requests.get(url, timeout=10).text
    text = trafilatura.extract(html) or ""
    return text[:4000] + ("\n...[截断]" if len(text) > 4000 else "")
```

原则：**工具负责把数据加工成模型友好形态**（提取正文、限量、转 markdown、引用编号）。

### 军规四：子 Agent 隔离上下文（预告）

多 Agent 的隐藏收益是**上下文隔离**：研究子 Agent 读 50 个页面污染自己的窗口，只把结论带回主 Agent。第 21-22 章展开。

## 19.2 结构化输出：让模型输出可编程

除了 `response_format`（第 17 章），任意 LLM 调用都可以结构化：

```python
from pydantic import BaseModel

class Route(BaseModel):
    intent: str          # "refund" | "faq" | "complaint"
    confidence: float

router_llm = llm.with_structured_output(Route)
decision = router_llm.invoke("你们价格太贵了我要退款")
# Route(intent='refund', confidence=0.92) —— 直接是 Python 对象
```

**路由判断必须结构化**（第 7 章的教训）：枚举字段 + 校验，杜绝字符串解析。

## 19.3 MCP：Model Context Protocol

MCP 是 Anthropic 发起的开放协议，把"模型 ⇄ 工具/数据源"标准化。生态里有大量现成 MCP Server（GitHub、Postgres、文件系统、浏览器、Slack…），**接一次协议，全生态工具可用**。

### 核心概念

```
MCP Host（你的 Agent 应用）
   └── MCP Client（协议客户端）
         ├── MCP Server A（提供 tools / resources / prompts）
         └── MCP Server B（…）
```

MCP Server 提供三种能力：**tools**（可调用函数）、**resources**（可读数据）、**prompts**（提示模板）。我们主要用 tools。

### 在 LangGraph 中接入（langchain-mcp-adapters）

```bash
pip install langchain-mcp-adapters
```

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

# 同时挂多个 MCP Server：stdio（本地进程）+ streamable http（远程服务）
client = MultiServerMCPClient({
    "math": {                                   # 本地 stdio server
        "transport": "stdio",
        "command": "python",
        "args": ["-m", "mcp_server_calculator"],
    },
    "filesystem": {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/data"],
    },
    "company_tools": {                          # 远程 HTTP server
        "transport": "streamable_http",
        "url": "https://mcp.internal.mycompany.com/mcp",
        "headers": {"Authorization": "Bearer xxx"},
    },
})

tools = await client.get_tools()     # MCP tools → LangChain 工具对象
agent = create_react_agent(llm, tools, prompt="你可以使用计算器、文件系统和内部系统工具。")
```

MCP 工具在模型眼中与普通 `@tool` 完全一致（name/description/schema 协议转换自动完成）。

### 自建 MCP Server（对外暴露公司能力）

```python
# pip install "mcp[cli]"
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("orders")

@mcp.tool()
def query_order(order_id: str) -> str:
    """查询订单状态与物流信息。"""
    return db.get_order(order_id).to_json()

if __name__ == "__main__":
    mcp.run()          # stdio；加参数可跑 HTTP
```

**何时选 MCP 而非 @tool**：能力要**跨团队/跨产品复用**（一个订单查询，客服 Agent、BI Agent、外部伙伴都要用）；单应用内私有逻辑用 @tool 更轻。

### MCP 工程注意

- stdio server 是**子进程**：部署要保证镜像里装好、命令可执行；HTTP server 注意鉴权与网络策略
- 工具数量同样要节制（MCP 很容易挂出几十个工具，回到 16.6 的问题）
- 生产建议**在公司层搭"内部 MCP 网关"**统一鉴权、审计、限流——LangGraph Server 自身的 API 也可以作为 MCP endpoint 暴露给其他 Agent 使用（第 24 章）

## 19.4 提示工程落地：Agent 提示模板

Agent system prompt 的推荐结构（配套第 17 章动态注入）：

```python
SYSTEM_TEMPLATE = """你是 {company} 的 {role}。

## 身份与边界
- 只回答与 {scope} 相关的问题
- 涉及 {taboo} 的话题一律礼貌拒答

## 工具使用规范
- 不确定的事实先调用搜索工具，不要编造
- 涉及钱款/账户的操作必须先确认用户身份

## 输出规范
- 中文，先给结论再给依据，总长不超过 {max_len} 字
- 引用资料时标注 [编号]

## 当前上下文
- 今天：{today}
- 用户：{user_profile}
- 已知记忆：
{memories}
"""
```

配合 LangSmith Prompt Hub（第 27 章）做版本管理与 A/B，提示即代码资产。

## 本章小结

- 上下文工程四军规：预算分配、动态 prompt、工具结果再加工、子 Agent 隔离
- 路由判断必须 `with_structured_output`
- MCP = 工具生态标准协议；`langchain-mcp-adapters` 三个工具无缝接入；复用型能力自建 MCP Server
- Agent 提示模板化 + 版本管理，提示是资产不是字符串

> 第六部分完成。接下来：多 Agent 系统。
