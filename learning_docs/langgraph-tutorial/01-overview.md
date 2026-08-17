# 第 1 章：LangGraph 全景

## 1.1 LangGraph 是什么

LangGraph 是 LangChain 团队推出的**低层级 Agent 编排框架 + 运行时**，用于构建可靠的、有状态的 LLM 应用。它把你的应用建模为一张**图（Graph）**：

- **节点（Node）**：一个处理步骤，通常是调用 LLM、执行工具或一段业务逻辑
- **边（Edge）**：步骤之间的跳转关系，支持条件分支
- **状态（State）**：贯穿全图共享的数据，每个节点读它、写它

一句话总结：**LangGraph = 状态机 + LLM，外加一整套让状态机能在生产环境跑起来的运行时能力**（持久化、恢复、流式、人机协同）。

它最初是为解决 LangChain 链式表达式（LCEL）无法表达**循环**而诞生的——Agent 的本质是"模型 ⇄ 工具"的循环，链（Chain）是单向无环的，图（Graph）才有循环和分支。

## 1.2 LangChain 生态中的位置

很多人分不清 LangGraph 和 LangChain 的关系。官方在 1.0 后给出了清晰的三层定位：

| 层 | 名字 | 职责 | 类比 |
|---|---|---|---|
| 框架层 | **LangChain** | 接入模型/工具的标准接口、预构建架构（如 ReAct 循环、RAG） | Web 框架里的路由和 ORM |
| 运行时层 | **LangGraph** | 图编排、状态、持久化、流式、HITL、部署 | Web 框架里的应用服务器 |
| 智能体层 | **Deep Agents** 等 | 在 LangGraph 上封装的高层 Agent 范式（规划、子 Agent、文件系统） | 脚手架/全家桶 |

关键事实：

1. **LangGraph 可以完全脱离 LangChain 单独使用**——`pip install langgraph` 就够了，节点里调什么模型、什么 SDK 都随你。
2. 反过来，LangChain 的预构建 Agent（`create_react_agent`）底层就是用 LangGraph 实现的。
3. LangChain 1.x 官方推荐的 Agent 构建路径就是"LangChain 接口 + LangGraph 编排"。

## 1.3 为什么用图而不是链

一个会查天气再回答问题的 Agent，执行流程是循环的：

```
用户提问 → LLM 思考 → 要调工具？──是──> 执行工具 → 把结果喂回 LLM ─┐
                ↑                                          │
                └────────────────── 否 ────────────────────┘
```

用链（顺序执行）无法表达"回到上一步"；用裸写 `while True:` 循环可以，但你很快会遇到一系列工程问题：

| 裸写循环的痛点 | LangGraph 的解法 |
|---|---|
| 进程重启，对话状态丢失 | **Checkpointer 持久化**，每个 superstep 自动存档，可恢复 |
| 无法在中间暂停等人审批 | **interrupt 机制**，节点内随时暂停/恢复 |
| 想边生成边推给前端 | **多种 stream mode**，token 级流式开箱即用 |
| 多 Agent 协作的状态管理混乱 | 图嵌图（子图）、共享状态、Command 路由 |
| 无法回看"刚才第 3 步发生了什么" | 完整执行历史（时间旅行）、LangSmith tracing |
| 长任务占着 HTTP 连接 | LangGraph Server 的后台任务 + 任务队列 |

也就是说，LangGraph 的价值 = **图编程模型 + 持久化运行时**。前者是写法，后者才是企业真正买单的东西。

## 1.4 生态地图（2026 年现状）

```
┌──────────────────────────── 开发 ────────────────────────────┐
│  langgraph (OSS)      图编排核心库（本教程主角）              │
│  langchain            模型/工具接口 + create_react_agent     │
│  deepagents           深度 Agent 范式（规划/子Agent/文件系统）│
│  langgraph-cli        langgraph dev / new / build            │
│  LangGraph Studio     可视化调试界面（浏览器）                │
├──────────────────────────── 部署 ────────────────────────────┤
│  LangGraph Platform   Server 运行时规范 + 云托管             │
│  langgraph-api        自托管 Server 实现（Docker 镜像）       │
│  langgraph-sdk        Python / JS / REST 客户端              │
├────────────────────────── 观测质量 ───────────────────────────┤
│  LangSmith            tracing、评估、Prompt 管理             │
│  LangSmith Engine     从 trace 自动发现 Agent 缺陷并提修复   │
│  LangSmith Fleet      无代码 Agent 构建/托管                 │
└──────────────────────────────────────────────────────────────┘
```

## 1.5 版本演进与稳定性

| 时间 | 版本 | 大事 |
|---|---|---|
| 2024-01 | 0.0.x | 开源发布 |
| 2024-2025 | 0.0.x → 0.4.x | API 频繁变动（`MemorySaver`、条件边为主流），痛点是"不稳定" |
| 2025-10 | **1.0** | **核心 API 冻结**：`StateGraph`、`Command`、`interrupt`、`create_react_agent`、Middleware 等 |
| 2025-2026 | 1.1 → 1.2.x | 持续加能力不破坏兼容：Functional API、deepagents、delta channels（状态 Schema 演进）、`trace_policy`、stream_events v3 等 |

截至本教程编写时（2026-08），最新版为 **1.2.11**。1.x 的承诺是：本教程讲的这些 API 在未来版本中保持兼容，可放心用于生产。

## 1.6 我该用 LangGraph 吗？（选型判断）

**适合 LangGraph 的场景：**
- 需要循环/分支/多步推理的 Agent（工具调用、ReAct、plan-and-execute）
- 需要人机协同（审批、确认、纠偏）
- 多 Agent 协作系统
- 长任务（分钟级以上）需要断点续跑
- 需要精细控制流程，且流程本身是产品的一部分

**可以不用 LangGraph 的场景：**
- 单轮、无状态的"提示词 + 模型"调用（直接调 SDK 即可）
- 纯 RAG 问答、没有循环需求（LangChain LCEL 或向量库自带能力足够）
- 流程完全固定、无 LLM 参与决策（普通工作流引擎如 Airflow/Temporal 更合适）

**经验法则**：如果你的 LLM 应用有"让模型决定下一步做什么"，或者"要在中间停下来等人"，LangGraph 就是目前 Python 生态最成熟的选择。

## 1.7 心智模型：LangGraph 的执行模型

理解一个核心概念就理解了 LangGraph 的一半——**superstep（超步）**：

1. 图的执行按"轮"进行，每一轮（superstep）内，**所有可执行的节点并发运行**
2. 一轮结束后，各节点的状态写入被**合并（reducer）**成新的快照
3. 根据边决定下一轮要执行哪些节点，如此往复，直到到达 `END`

这借鉴了 Google 的 **Pregel/Bulk Synchronous Parallel** 模型。记住两点即可：
- 同一轮内的节点互不依赖时是**并行**的
- 状态更新不是"覆盖"而是"合并"，合并规则由 Reducer 决定（第 3 章详解）

## 本章小结

- LangGraph = 低层级编排框架 + 持久化运行时，可以脱离 LangChain 使用
- 图解决链解决不了的循环与分支；运行时解决生产环境的持久化/流式/HITL
- 1.0 后 API 已冻结，现在是入手生产级使用的稳定期
- 生态：LangChain（接口）/ LangGraph（运行时）/ Platform（部署）/ LangSmith（观测）

> 下一章：10 分钟跑通你的第一个 LangGraph 应用。
