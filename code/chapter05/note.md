# 第05章 多智能体浏览器自动化

## 📌 上节回顾：AgentExecutor 与 现代 Agent 架构演进

在上一章中，我们学习了使用 `create_tool_calling_agent` 与 `AgentExecutor` 来实现工具调用和串联。但在实际的生产级（Production-grade）开发中，这套范式正逐渐被更先进的架构所取代。

---

### 一、 为什么 `AgentExecutor` 在生产中被逐渐弃用？

1. **控制力差（黑盒化严重）**：
   `AgentExecutor` 内部封装了固定的循环机制（Prompt $\rightarrow$ LLM $\rightarrow$ Tool Call $\rightarrow$ Tool $\rightarrow$ Context）。当业务需要加入**条件分支/路由**、**并发工具调用**或**人工干预/审批（Human-in-the-loop）**时，极难灵活扩展。
2. **状态管理（State Management）脆弱**：
   `AgentExecutor` 的上下文仅存在于内存中的 `agent_scratchpad`。缺乏原生的持久化（Persistence）、Checkpointer 和会话断点恢复能力。
3. **官方演进方向转移**：
   LangChain 官方在 0.2+ / 0.3+ 版本中已明确将 `AgentExecutor` 标记为**旧版/遗留（Legacy）架构**，未来主推基于图状态机的 LangGraph。

---

### 二、 现代 Agent 开发的主流替代方案

#### 1. 方案一：LangGraph（工业级标准）

LangChain 官方推出的基于有向图（Graph）的状态机框架：

- **显式状态管理**：状态在节点（Nodes）与边（Edges）之间透明传递。
- **内置持久化**：原生支持 SQLite/PostgreSQL 存盘与会话恢复。
- **Human-in-the-loop**：支持在任意节点中断并挂起，等待外部审批后再恢复执行。
- **快速替代**：提供了 `from langgraph.prebuilt import create_react_agent` 极简替换原生的 `AgentExecutor`。

#### 2. 方案二：原生 `llm.bind_tools()` + 自定义循环（轻量场景）

直接使用 LangChain 的 `bind_tools` 或大模型原生 SDK，通过手写 `while` 循环显式控制工具调用：

```python
# 绑定工具
llm_with_tools = llm.bind_tools(tools)
messages = [SystemMessage(...), HumanMessage("...")]

while True:
    response = llm_with_tools.invoke(messages)
    messages.append(response)

    if not response.tool_calls:
        break  # 无工具调用，结束循环

    for tool_call in response.tool_calls:
        tool_result = execute_tool(tool_call)
        messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))
```

---

### 三、 总结与技术选型建议

| 框架/模式                                         | 适用场景                                        | 生产推荐度                           |
| :------------------------------------------------ | :---------------------------------------------- | :----------------------------------- |
| **`create_tool_calling_agent` + `AgentExecutor`** | 10 行代码快速 PoC 验证、教学入门                | ❌ 不推荐用于生产                    |
| **`llm.bind_tools()` + 自定义循环**               | 逻辑清晰、追求轻量与零依赖的 Agent              | ⭐⭐⭐⭐ 推荐（中小型项目）          |
| **LangGraph (`create_react_agent` / Graph)**      | 复杂工作流、多 Agent 协作、状态持久化、人工审批 | ⭐⭐⭐⭐⭐ 强力推荐（复杂/大型项目） |

---

### 四、 PlayWrightBrowserToolkit 生产评估与现代替代

#### 1. 为什么生产环境不推荐 `PlayWrightBrowserToolkit`？
- **Token 浪费严重**：直接提取全页纯文本（`extract_text`），噪声极多，易超出模型上下文限制。
- **感知能力有限**：缺乏 Vision（视觉）与 DOM 序号标注，难以处理复杂 SPA 单页应用与无语义按钮。
- **防封锁能力差**：默认配置易触发 Cloudflare 验证码及人机阻断。

#### 2. 工业界主流浏览器 Agent 替代方案

| 方案 / 工具 | 核心优势与适用场景 | 生产推荐度 |
| :--- | :--- | :--- |
| **`browser-use` / `Stagehand`** | 视觉 + DOM 序号标注，支持复杂多步骤交互、自主规划与纠错 | ⭐⭐⭐⭐⭐ **强烈推荐 (Agent)** |
| **`Crawl4AI` / `Firecrawl`** | 专为 LLM 设计的爬虫，自动清洗转化为 Markdown/JSON | ⭐⭐⭐⭐⭐ **强烈推荐 (爬虫)** |
| **原生 Playwright + LLM** | 流程确定性 100%，极低 Token 成本，适合固定后台自动化 | ⭐⭐⭐⭐ **推荐 (固定流程)** |

