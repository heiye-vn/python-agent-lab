# 章节主题与关键内容说明

> 本文件用于快速定位 `code/` 目录下每个章节的学习主题、关键知识点与代表文件，方便回顾与复习。

## 📚 章节总览

| 章节 | 主题 | 核心关键词 |
|------|------|-----------|
| [Chapter 01](#chapter-01-llm-客户端创建方式) | LLM 客户端创建方式 | `init_chat_model`、官方 SDK、多厂商接入、消息系统 |
| [Chapter 02](#chapter-02-链与-lcel) | 链与 LCEL | LCEL、复合链、自定义节点、结构化输出 |
| [Chapter 03](#chapter-03-对话系统) | 对话系统 | 单轮/多轮对话、流式输出、Gradio 聊天机器人 |
| [Chapter 04](#chapter-04-工具调用-tool) | 工具调用 Tool | `@tool`、Tool Calling、ToolNode、Agent Loop |
| [Chapter 05](#chapter-05-浏览器自动化) | 浏览器自动化 | Playwright、多智能体、AgentExecutor 替代 |
| [Chapter 06](#chapter-06-mcp-协议) | MCP 协议 | Tools/Resources/Prompts、Stdio/HTTP、OAuth |
| [Chapter 07](#chapter-07-pdf-rag-系统) | PDF RAG 系统 | FAISS、向量检索、检索工具、Agent |
| [Chapter 08](#chapter-08-数据分析-agent) | 数据分析 Agent | pandas、matplotlib、代码执行工具 |
| [Chapter 09](#chapter-09-langgraph) | LangGraph | StateGraph、Reducer、条件边、预构建图 |
| [Chapter 10](#chapter-10-langgraph-多工具调用) | LangGraph 多工具调用 | `create_agent`、多工具绑定、递归限制 |
| [Chapter 11](#chapter-11-langgraph-智能体服务化与-langsmith-全链路监控) | LangGraph 部署与 LangSmith 监控 | `langgraph dev`、Studio 调试、LangSmith Traces、全链路追踪 |
| [Chapter 12](#chapter-12-langgraph-智能数据分析-agent) | LangGraph 智能数据分析 Agent | NL2SQL、DataFrame 缓存、代码执行 REPL、Seaborn 可视化、双 Graph 架构 |
| [Chapter 12 (UI)](#chapter-12-ui-langgraph-前端聊天交互界面-agent-chat-ui) | LangGraph 前端交互客户端 | Next.js、LangGraph Server 对接、流式响应、工具调用可视化 |
| [Chapter 13](#chapter-13-langgraph-基础图构建与-pydantic-状态管理) | LangGraph 基础图构建与 Pydantic 状态管理 | StateGraph、Pydantic State、START/END、draw_mermaid |
| [Chapter 14](#chapter-14-langgraph-条件分支与循环图) | LangGraph 条件分支与循环图 | 条件边 `add_conditional_edges`、循环控制、计数器 |
| [Chapter 15](#chapter-15-langgraph-对话持久化与-react-预构建图复现) | LangGraph 对话持久化与 ReAct 复现 | `InMemorySaver`、`add_messages` Reducer、ToolNode、Agent Loop |
| [Chapter 16](#chapter-16-langgraph-进阶记忆管理与长期记忆系统) | LangGraph 进阶记忆管理与长期记忆 | Checkpointer(Memory/SQLite/PG/Redis)、Store 长期记忆、滚动摘要、Mem0、Zep、综合交互系统 |

---

## Chapter 01 — LLM 客户端创建方式

**主题**：LangChain 模型接入入门，对比不同方式创建 LLM 客户端，并接入多家厂商模型。

**关键内容**：
- `langchain.chat_models.init_chat_model` 工厂函数 vs 厂商官方 SDK（`openai.OpenAI`）的对比与取舍。
- 多厂商接入：**DeepSeek**（`deepseek-v4-pro`）、**硅基流动**（`Qwen3.6-35B-A3B`）、**阿里百炼**（`qwen3.7-max`）。
- LangChain 消息系统：`SystemMessage` / `HumanMessage` / `AIMessage` 类与 `("system", ...)` 元组简写的适用场景。
- 面试高频：为什么推荐 LangChain 统一封装、`init_chat_model` 体现的设计模式、可观测性优势。

**代表文件**：`hello_langchain.py`、`LangChain_DeepSeek.py`、`LangChain_Silicon.py`、`note.md`

---

## Chapter 02 — 链与 LCEL

**主题**：LangChain Expression Language（LCEL）——链的构造、组合与输出处理。

**关键内容**：
- **简单链**：`prompt | model | parser` 管道式组合。
- **复合链**：
  - 串行复合（链中套链）；
  - `RunnableParallel` 并行执行多个分支；
  - `RunnablePassthrough.assign` 透传输入并追加新字段；
  - `RunnableBranch` 条件路由。
- **自定义节点**：`RunnableLambda` 包装函数、`@chain` 装饰器、继承 `Runnable` 写组件类。
- **输出处理**：`OutputParser` 基础解析、`with_structured_output` + Pydantic 结构化输出（推荐）、自定义后处理。
- 面试高频：LCEL vs 旧版 `LLMChain`、流式输出如何被节点阻断等。

**代表文件**：`01_简单链.py`、`02_加入提示词.py`、`03_复合链(推荐).py`、`04_复合链与自定义节点.py`、`note.md`

---

## Chapter 03 — 对话系统

**主题**：从单轮对话到多轮、流式，再到可交互的聊天机器人。

**关键内容**：
- **单轮对话**：一次性请求-响应。
- **多轮对话**：消息历史的拼接与上下文管理。
- **流式输出**：`stream` / `astream` 逐 token 返回。
- **智能聊天机器人**：基于 **Gradio** 构建 Web UI；`MessagesPlaceholder` 注入历史、`gr.State` 维护消息对象列表、`astream` 实现打字机效果、历史裁剪（最近 50 条）。

**代表文件**：`01_单轮对话.py`、`02_多轮对话.py`、`03_流式输出.py`、`04_智能聊天机器人.py`、`05_智能聊天机器人.py`

---

## Chapter 04 — 工具调用 Tool

**主题**：LangChain 工具（Tool）的定义、调用流程与工程实践。

**关键内容**：
- `@tool` 装饰器定义工具，`Pydantic` 定义复杂 `args_schema`。
- Tool 的核心组成：name、description、args_schema、返回值。
- **Tool Calling 流程**：模型输出 tool call → 执行工具 → 回传结果。
- 现代 Agent API：`create_agent`、`ToolRuntime`、`ToolNode` 与 Agent Loop。
- State / Context / Store 三者对比；多工具串联调用。
- 实战：天气查询等外部 API 工具；工具安全与工程实践（隐藏参数、错误处理、并行调用）。

**代表文件**：`01_工具调用.py` ~ `05_多工具串联调用.py`、`global_cities_data.csv`、`note.md`

---

## Chapter 05 — 浏览器自动化

**主题**：基于 Playwright 的多智能体浏览器自动化。

**关键内容**：
- Playwright 浏览器自动化操作。
- `AgentExecutor` 在生产中被逐渐弃用的原因，以及现代替代方案（`create_agent` / LangGraph）。
- 多智能体协作完成浏览器任务的技术选型建议。

**代表文件**：`01_浏览器自动化_new.py`、`01_浏览器自动化_old.py`、`02_浏览器自动化.py`、`note.md`

---

## Chapter 06 — MCP 协议

**主题**：Model Context Protocol（MCP）——大模型连接外部工具/数据/提示词的开放协议。

**关键内容**：
- **三大原语**：Tools（工具）、Resources（资源）、Prompts（提示词模板）。
- **传输方式**：Stdio（本地标准输入输出）vs Streamable HTTP（远程流式）。
- **认证安全**：OAuth 2.1 认证流程与不同传输的安全策略。
- **LangChain 接入**：`langchain-mcp-adapters` 三种接入方式（Stdio 本地、HTTP 远程、多 Server 聚合）。
- 面试高频：MCP vs Function Calling、无状态设计优势等。

**代表文件**：`01_langchain_mcp.py`、`MCP_LangChain_文档.md`、`servers_config.json`、`note.md`

---

## Chapter 07 — PDF RAG 系统

**主题**：基于 PDF 文档的检索增强生成（RAG）系统。

**关键内容**：
- PDF 读取（`PyPDF2`）与文本切分（`RecursiveCharacterTextSplitter`）。
- 向量化（`DashScopeEmbeddings`）+ **FAISS** 向量库存储。
- 将检索能力封装为 `retriever_tool`，交给 Agent 调用。
- 经典版（`AgentExecutor` + `create_tool_calling_agent`）与 v1 现代化版本的对比。
- 附 `DeepSeek-R1 高性能部署实战.pdf` 参考资料。

**代表文件**：`langchain_classic_pdf_rag_system.py`、`langchain_v1_pdf_rag_system.py`、`DeepSeek-R1高性能部署实战.pdf`

---

## Chapter 08 — 数据分析 Agent

**主题**：用自然语言驱动的数据分析 Agent（NL2Code）。

**关键内容**：
- `pandas` 数据处理 + `matplotlib` 可视化。
- `PythonAstREPLTool` 代码解析执行工具。
- `langgraph.prebuilt.create_react_agent` 替代过时的 `AgentExecutor`。
- Streamlit 前端交互；经典版 vs v1 版的演进对比。

**代表文件**：`langchain_classic_data_analysis.py`、`langchain_v1_data_analysis.py`

---

## Chapter 09 — LangGraph

**主题**：LangGraph 状态图编排框架的系统学习。

**关键内容**：
- 核心概念：`StateGraph`、`State`、`Reducer`（状态合并策略）、`Node`、`Edge`、`START`/`END`、`compile`。
- 条件边、循环控制、checkpoint 持久化。
- 预构建图的使用（如 `create_react_agent`）。
- 配套文档：`langgraph 详解.md`（深度讲解）+ `langgraph 知识点速查.md`（速记手册）。

**代表文件**：`langgraph预构建图.py`、`langgraph 详解.md`、`langgraph 知识点速查.md`

---

## Chapter 10 — LangGraph 多工具调用

**主题**：LangGraph Agent 中多工具与内部工具的集成调用。

**关键内容**：
- `langchain.agents.create_agent`（新）与 `create_react_agent`（已弃用）的对比。
- 多工具绑定：天气查询（心知天气 API）、文件写入等工具。
- 内部工具调用。
- `GraphRecursionError` 递归限制的处理。

**代表文件**：`01_langgraph多工具调用.py`、`02_langgraph内部工具调用.py`

---

## Chapter 11 — LangGraph 智能体服务化与 LangSmith 全链路监控

**主题**：LangGraph 本地服务化部署、LangGraph Studio 可视化调试与 LangSmith 链路追踪。

**关键内容**：
- `langgraph.json` 架构与服务声明。
- `langgraph dev` 启动开发服务与 Web Studio 可视化单步调试。
- **LangSmith** 链路追踪（Traces）：Token 消耗、延迟分析、节点流转树与调用异常监控。

**代表文件**：`graph.py`、`langgraph.json`、`README.md`

---

## Chapter 12 — LangGraph 智能数据分析 Agent

**主题**：全流程智能数据分析 Agent（NL2SQL + DataFrame 内存提取 + Python 代码计算 + 数据可视化）。

**关键内容**：
- 四大核心分析工具协同：`sql_inter`、`extract_data`、`python_inter`、`fig_inter`。
- 全局内存共享：将 SQL 提取的 Pandas DataFrame 直接沉淀在内存中供后续 Python/绘图复用。
- **双 Graph 模式支持**：
  - `data_agent`：标准 MySQL 关系型数据库支持。
  - `data_agent_sqlite`：内置 SQLite 开箱即用模式（自带电信客户流失数据集 `telco.db`）。

**代表文件**：`graph.py`、`graph_sqlite.py`、`init_db.py`、`langgraph.json`、`README.md`

---

## Chapter 12 (UI) — LangGraph 前端聊天交互界面 (Agent Chat UI)

**主题**：基于 Next.js 的现代化 LangGraph 前端聊天客户端，为本地及云端部署的 LangGraph 服务提供开箱即用的 Web 对话与调试界面。

**关键内容**：
- **服务对接与协议交互**：通过 `NEXT_PUBLIC_API_URL`（默认 `http://localhost:2024`）和 `NEXT_PUBLIC_ASSISTANT_ID` 无缝连接本地 `langgraph dev` 暴露的 Graph 服务。
- **流式响应与工具调用可视化**：实时流式渲染 LLM 生成内容，折叠/展开呈现 Agent 内部工具调用（Tool Calls）输入与输出。
- **多会话与状态管理**：支持多 Thread 对话切换、上下文恢复与消息历史管理。
- **技术栈**：Next.js (App Router) + React + Tailwind CSS + Radix UI / shadcn/ui。

**代表文件**：`code/chapter12_agent-chat-ui/package.json`、`src/`、`.env.example`、`README.md`

---

## Chapter 13 — LangGraph 基础图构建与 Pydantic 状态管理

**主题**：深入 LangGraph 底层图构建机制，掌握 StateGraph 节点流转、Mermaid 结构可视化，以及基于 Pydantic BaseModel 的强类型状态管理。

**关键内容**：
- **基础状态图构建**：基于 `StateGraph(dict)` 搭建图工作流，使用 `add_node` 注册计算节点、`add_edge` 配置 `START -> addition -> subtraction -> END` 确定性串行流转。
- **图结构可视化**：调用 `graph.get_graph().draw_mermaid()` 导出标准 Mermaid 流程图代码，掌握实线（固定边）/虚线（条件边）的渲染与节点链路检查。
- **Pydantic 结构化状态**：使用 `pydantic.BaseModel` 和 `Field` 替代原生字典，实现状态字段自动类型校验、默认值设置与属性访问安全。

**代表文件**：`01_create_graph.py`、`02_pydantic_state.py`

---

## Chapter 14 — LangGraph 条件分支与循环图

**主题**：掌握 LangGraph 的条件路由判定、循环控制与递归限制处理。

**关键内容**：
- **条件边判定（Conditional Edge）**：使用 `add_conditional_edges` 结合路由函数，实现分支流转（如根据状态值跳转到不同计算节点或 `END`）。
- **循环图控制（Cycle Graph）**：在状态图中构建节点回环（如 `node_a -> node_b -> node_a`），利用 State 中的计数器（Counter）控制终止条件，避免无限死循环。
- **复合判定与循环图**：将条件判断与循环逻辑结合，实现具备自我调节和重试能力的动态图工作流。

**代表文件**：`01_judge_graph.py`、`02_cycle_graph.py`、`03_judge_cycle_graph.py`

---

## Chapter 15 — LangGraph 对话持久化与 ReAct 预构建图复现

**主题**：深入 LangGraph 对话持久化机制，手写底层节点与条件边复现 `create_react_agent`。

**关键内容**：
- **基础多轮对话与 Reducer**：使用 `add_messages` 实现消息状态的追加合并而非覆盖。
- **会话持久化与 Checkpointer**：使用 `InMemorySaver` 绑定图编译，通过 `configurable={"thread_id": ...}` 实现多会话隔离与上下文自动维护。
- **手写复现 ReAct 预构建图**：
  - 自定义 `AgentState`；
  - 构造 `call_model` 节点绑定心知天气 API 工具（`@tool` + Pydantic Schema）；
  - 构造 `ToolNode` 与 `should_continue` 条件边，完整跑通 `agent -> tools -> agent -> END` 闭环。

**代表文件**：`01_多轮对话机器人.py`、`02_持久化多轮对话机器人.py`、`03_复现ReAct预构建图.py`

---

## Chapter 16 — LangGraph 进阶记忆管理与长期记忆系统

**主题**：掌握工业级 Agent 多层记忆架构，覆盖单会话短期持久化（Checkpointer）、跨会话长期画像（Store）、上下文裁剪与滚动摘要、时间旅行状态干预、第三方前沿记忆引擎（Mem0 / Zep）以及企业级混合记忆交互系统。

**关键内容**：
- **短期会话持久化（Checkpointer 演进）**：
  - **内存级**：`InMemorySaver` 适用于单机开发与单元测试；
  - **本地持久化**：`SqliteSaver` 实现轻量文件级会话断点保存；
  - **分布式云端**：Supabase `PostgresSaver` 实现生产级多实例高并发状态共享；
  - **高速缓存与 TTL**：Upstash `Redis` 检查点管理与会话级自动过期淘汰（TTL 回收）。
- **长期跨会话记忆（Store 架构与用户画像）**：
  - `InMemoryStore`、`SqliteStore`、Supabase `PostgresStore` 的存储模型；
  - 多租户 Namespace 分层设计（如 `("users", user_id, "profile")`）；
  - 冷启动画像召回（Recall）注入 System Prompt 与热路径工具（`save_user_memory`）自主沉淀。
- **短程上下文预算控制与滚动摘要**：
  - `trim_messages` 锁定 `SystemMessage` 的精准 Token 预算裁剪；
  - `Summarization Node` 触发阈值动态生成对话摘要，配合 `RemoveMessage` 修剪历史冗余。
- **时间旅行（Time Travel）与状态干预**：
  - `get_state_history` 追溯快照调用链；
  - **Replay 时光重放**：回退并修改历史节点状态，沿原 Thread 重新执行；
  - **Fork 平行分叉**：基于历史检查点克隆至新 `thread_id` 探索平行分支。
- **前沿第三方记忆引擎集成**：
  - **Mem0**：多层级（User/Session/Agent）记忆沉淀、自动冲突更新与语义相似度召回；
  - **Zep (Graphiti)**：时序知识图谱（Temporal Knowledge Graph），节点与实体关系的动态演进。
- **企业级综合多层记忆交互系统**：
  - 融合 **Supabase PostgreSQL**（长期记忆 Store）+ **Upstash Redis**（短期记忆 TTL Checkpointer）+ **滚动摘要** + **CLI 控制台调试指令** 的完整闭环应用。

**代表文件**：
- 短期记忆：`01_langgraph_InMemorySaver_短期记忆.py`、`02_langgraph_sqlite_短期记忆.py`、`03_langgraph_postgres_短期记忆.py`、`07_langgrapg_redis_记忆系统.py`、`08_langgraph_redis_test.py`
- 长期记忆：`04_langgraph_InMemoryStore_长期记忆.py`、`05_langgraph_sqlite_长期记忆.py`、`06_langgraph_postgres_长期记忆.py`、`10_长期记忆Store与用户画像.py`
- 裁剪与时间旅行：`09_消息裁剪与滚动摘要记忆.py`、`11_时间旅行与状态分支干预.py`
- 第三方记忆库：`12_mem0_test.py`、`13_langgraph_mem0多维度记忆系统.py`、`14_mem0_对话机器人.py`、`15_zep_时序知识图谱记忆.py`
- 综合交互系统与笔记：`16_综合记忆管理交互系统.py`、`note.md`

---

## 📎 其他学习资料

| 路径 | 说明 |
|------|------|
| `interview/agent 评估体系.md` | 大模型/Agent 评估体系面试知识点速查（四层评估框架、LLM-as-a-Judge、Ragas、Benchmark） |
| `learning_docs/llamaindex_learning_guide.md` | LlamaIndex 框架学习路线图（第 0~9 站，从五分钟 RAG 到生产化） |
