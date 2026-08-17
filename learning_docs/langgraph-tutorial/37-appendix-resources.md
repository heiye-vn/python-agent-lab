# 附录 E：官方资源与延伸阅读

> 链接以官方域名为准（docs.langchain.com 是 2025 后的统一文档站，老链接 docs.langchain.com/oss/python/langgraph 下的 langgraph 部分仍可访问）。遇到细节争议，**以官方文档和源码为最终依据**。

## E.1 官方文档（首选）

| 资源 | 地址 | 用途 |
|---|---|---|
| LangGraph 概念总览 | docs.langchain.com → OSS → Python → LangGraph | 概念地图、how-to |
| LangChain Agents 文档 | docs.langchain.com → OSS → Python → LangChain | create_react_agent、middleware、工具 |
| LangGraph JS/TS | 同站 JavaScript 分区 | TS 版对照 |
| LangGraph Platform/部署 | docs.langchain.com → LangGraph Platform | Server、SDK、自托管、auth |
| LangSmith | docs.smith.langchain.com | tracing/evals/prompt hub |
| deepagents | GitHub: langchain-ai/deepagents | 深度 Agent |
| langgraph-swarm / supervisor | GitHub: langchain-ai/* | 多 Agent 官方库 |
| langchain-mcp-adapters | GitHub: langchain-ai/langchain-mcp-adapters | MCP 集成 |

## E.2 源码仓库（进阶必读）

- **langchain-ai/langgraph**：核心库。读源码顺序建议：
  1. `libs/langgraph/langgraph/graph/state.py` —— StateGraph 组装
  2. `.../pregel/` —— superstep 执行引擎（Pregel 模型实现）
  3. `.../channels/` —— LastValue/BinaryOperatorAggregate 等通道与 reducer 语义
  4. `.../types.py` —— Command/Send/interrupt 定义
- **langchain-ai/langgraph-sdk**：REST 客户端，理解 Server 协议
- 示例库：langchain-ai/langgraph-examples（多 Agent、HITL、RAG 等完整例程）

## E.3 学习路径建议（配合本书）

```
第 1 周   通读第 1-6 章，跑通第 2 章两个例子
第 2 周   第 7-10 章：改造你的例子——加并行、加流式
第 3 周   第 11-15 章：上 PostgresSaver、做 HITL demo（吃透 get_state/interrupt）
第 4 周   第 16-19 章：把第 2 周的例子改造成 create_react_agent + middleware
第 5 周   第 20-22 章：跑通第 31 章研究助手
第 6 周   第 23-28 章：langgraph dev → Studio 调试 → docker 自托管 → 接 LangSmith
持续      四个实战项目选 1-2 个完整复刻进自己的仓库，badcase 进数据集
```

## E.4 论文与设计思想（加深理解）

- **Pregel: A System for Large-Scale Graph Processing**（Google, 2010）—— LangGraph 执行模型的源头（BSP/superstep）
- **ReAct: Synergizing Reasoning and Acting**（2022）—— create_react_agent 的理论原型
- ** Reflexion / Self-RAG / Plan-and-Execute** 相关论文 —— 第 19 章高级模式的原型
- Anthropic: **Building Effective Agents**（2024）—— "workflow vs agent" 选型观，与第 20 章架构思想一致
- MCP 规范：modelcontextprotocol.io

## E.5 社区与动态

- GitHub Discussions / Issues：langchain-ai/langgraph（问题检索质量高）
- LangChain Blog & Changelog：版本特性第一手（如 1.2 的 delta channels、stream_events v3）
- LangSmith YouTube 频道：官方 deep dive 视频
- Discord：LangChain 社区（#langgraph 频道）

## E.6 版本再确认清单（阅读旧资料时）

学 LangGraph 最大的坑是**按 0.x 时代的博客学 1.x**。判断资料是否过时的三问：

1. 用 `MemorySaver` 还是 `InMemorySaver`？（后者新）
2. HITL 用 `interrupt_before` 还是节点内 `interrupt()`？（后者是 1.x 推荐）
3. 有没有 Command / middleware？（1.x 核心特性）

命中两条"旧"，就换最新官方文档重新核对写法（对照附录 B 迁移表也能读通旧代码）。

---

**全书完。** 回到 [README](README.md) 查看完整目录与学习路线。
