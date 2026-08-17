# LangGraph 完整学习教程

> 基于 **LangGraph 1.2.x**（截至 2026-08 最新稳定版）· Python 主线 · 面向生产与企业应用
>
> 1.0 版本（2025-10 发布）已冻结核心 API，本教程所有代码均基于 1.x 稳定 API 编写。

## 目录

### 第一部分：认识 LangGraph
- [第 1 章：LangGraph 全景](01-overview.md)
- [第 2 章：环境准备与第一个应用](02-quickstart.md)

### 第二部分：核心概念（最重要）
- [第 3 章：状态 State —— LangGraph 的心脏](03-state.md)
- [第 4 章：节点 Node](04-nodes.md)
- [第 5 章：边 Edge 与 Command](05-edges-command.md)
- [第 6 章：图的构建、编译与可视化](06-build-compile.md)

### 第三部分：进阶控制流
- [第 7 章：循环、分支与并行（Send API）](07-control-flow.md)
- [第 8 章：子图 Subgraph 与图组合](08-subgraphs.md)
- [第 9 章：Functional API（@entrypoint / @task）](09-functional-api.md)
- [第 10 章：流式输出 Streaming](10-streaming.md)

### 第四部分：持久化与记忆
- [第 11 章：Checkpointer 与短期记忆](11-checkpointer.md)
- [第 12 章：长期记忆 Store](12-long-term-memory.md)
- [第 13 章：时间旅行 Time Travel](13-time-travel.md)

### 第五部分：Human-in-the-Loop
- [第 14 章：interrupt 机制详解](14-interrupt.md)
- [第 15 章：HITL 工程化](15-hitl-production.md)

### 第六部分：构建单个 Agent
- [第 16 章：工具 Tools 深度解析](16-tools.md)
- [第 17 章：create_react_agent 预构建 Agent](17-create-react-agent.md)
- [第 18 章：Middleware 中间件](18-middleware.md)
- [第 19 章：上下文工程与 MCP 集成](19-context-mcp.md)

### 第七部分：多 Agent 系统
- [第 20 章：多 Agent 架构总览](20-multi-agent-overview.md)
- [第 21 章：经典多 Agent 模式实现](21-multi-agent-patterns.md)
- [第 22 章：Deep Agents](22-deep-agents.md)

### 第八部分：企业级部署与运维
- [第 23 章：本地开发与调试](23-local-dev.md)
- [第 24 章：LangGraph Platform / Server 架构](24-platform.md)
- [第 25 章：API 与 SDK](25-sdk-api.md)
- [第 26 章：部署方案](26-deployment.md)
- [第 27 章：可观测性与质量保障（LangSmith）](27-langsmith.md)
- [第 28 章：生产最佳实践](28-production.md)

### 第九部分：实战项目
- [第 29 章：项目一 —— 智能客服机器人](29-project-customer-service.md)
- [第 30 章：项目二 —— RAG 知识库问答 Agent](30-project-rag.md)
- [第 31 章：项目三 —— 多 Agent 研究助手](31-project-research.md)
- [第 32 章：项目四 —— 企业审批工作流](32-project-approval.md)

### 第十部分：附录
- [附录 A：LangGraph JS/TS 版差异速览](33-appendix-js.md)
- [附录 B：0.x → 1.x 迁移指南](34-appendix-migration.md)
- [附录 C：常见报错与踩坑集锦](35-appendix-pitfalls.md)
- [附录 D：高频面试题与答案要点](36-appendix-interview.md)
- [附录 E：官方资源与延伸阅读](37-appendix-resources.md)

## 推荐学习路线

```
入门（1-2 周）        进阶（2-3 周）          生产（2-3 周）
─────────────       ─────────────          ─────────────
第 1-2 章  概念+跑通   第 7-10 章  控制流+流式   第 23-28 章 部署运维
第 3-6 章  核心四概念   第 11-15 章 持久化+HITL   第 29-32 章 实战项目
第 16-17 章 单 Agent   第 18-22 章 Middleware+多Agent  附录 查阅用
```

## 约定

- 所有示例默认已设置环境变量：`OPENAI_API_KEY`（或 `ANTHROPIC_API_KEY`）
- 模型统一通过 `init_chat_model` 初始化，方便切换供应商
- 每个代码块尽量自包含、可直接运行（`pip install langgraph langchain langchain-openai` 之后）
