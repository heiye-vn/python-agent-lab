# 附录 B：0.x → 1.x 迁移指南

大量存量教程/博客是 0.2~0.4 时代写的。本附录给出旧写法 → 1.x 标准写法的对照，帮你：读旧代码不迷路、迁移旧项目有章法。

## B.1 总原则

1.x（2025-10 发布）**冻结核心 API**，绝大多数概念延续，但统一/清理了一批接口。迁移优先级：

```
高：MemorySaver 更名、interrupt 用法、条件边 → Command（推荐非强制）
中：create_react_agent 参数规范化、checkpointer 安装拆包
低：命名风格统一（如 astream_events 版本）
```

## B.2 核心对照表

### 检查点器

| 0.x | 1.x |
|---|---|
| `from langgraph.checkpoint.memory import MemorySaver` | `InMemorySaver`（MemorySaver 旧名仍兼容但新代码别用） |
| langgraph 自带 sqlite/postgres 实现 | 拆分到独立包 `langgraph-checkpoint-sqlite` / `langgraph-checkpoint-postgres`，`pip install` 后 import 路径不变 |
| `SqliteSaver.from_conn_string(...)` 直接用 | 推荐上下文管理器 `with ... as cp:`（自动建表/关闭） |

### HITL

| 0.x | 1.x |
|---|---|
| 只有 `interrupt_before/after` 编译参数 | 首选节点内 `interrupt(payload)` + `Command(resume=...)`；编译参数保留用于断点 |
| `graph.update_state(config, values, as_node="x")` | 同（as_node 仍可用） |
| `NodeInterrupt` 异常 | 保留，语义不变 |

### 路由

| 0.x | 1.x |
|---|---|
| 只能 `add_conditional_edges` | 新增节点内 `Command(goto=, update=)`（官方推荐）；条件边仍支持 |
| 子图回父图靠共享 schema/包装 | 新增 `Command(graph=Command.PARENT)` |

### 预构建 Agent

| 0.x | 1.x |
|---|---|
| `create_react_agent(model, tools, prompt, modifiable ...)` 参数杂 | 规范化：`prompt` / `pre_model_hook` / `post_model_hook` / `state_schema` / `response_format` / `middleware` |
| HITL 要手改 agent 结构 | pre_model_hook / middleware 官方插槽 |
| `create_react_agent(..., checkpointer=cp)` 直接挂 | 同（保留） |

### Send / 并行

| 0.x | 1.x |
|---|---|
| `Send` 从 `langgraph.constants` | 从 `langgraph.types`（constants 路径兼容） |
| `Command` 同 | `langgraph.types` |

### Functional API

| 0.x（0.2+ 引入） | 1.x |
|---|---|
| `@entrypoint` / `@task` 实验性 | 稳定；`previous` 状态、`get_entrypoint_state()` 规范化 |

### 流式

| 0.x | 1.x |
|---|---|
| `stream_mode="messages"` 返回裸 chunk | 返回 `(chunk, metadata)` 元组 |
| `astream_events(version="v2")` | 同（1.2 起新增 v3 实现，v2 写法兼容） |

### Middleware（新增，无对照）

0.x 时代靠各种钩子/hack 的横切逻辑（guardrails、审批、模型路由）→ 1.x 统一到 `middleware` 参数 + `AgentMiddleware`（第 18 章）。

## B.3 迁移操作步骤（存量项目）

```
1. 升级依赖
   pip install -U langgraph langchain langgraph-checkpoint-postgres
   （0.4 → 1.x 跨版本，先看 release notes 的 breaking list）

2. 全局搜索替换（低风险机械项）
   MemorySaver → InMemorySaver
   langgraph.constants.Send → langgraph.types.Send
   （旧 import 兼容的可暂不动，但新代码统一新路径）

3. HITL 改造（收益最大）
   interrupt_before=[...] + update_state 的组合
     → 节点内 interrupt(payload) + Command(resume=)
   恢复入口统一为 invoke(Command(resume=...), config)

4. Agent 升级
   老式手改 prebuilt 内部结构 → pre_model_hook / middleware
   顺手把状态裁剪挂到 pre_model_hook（每轮生效）

5. 条件边（可选）
   高价值路由（带状态更新的、需要动态目标的）→ Command
   纯静态展示型路由保留条件边（图画得清楚）

6. 回归验证（第 27 章）
   LangSmith 数据集跑 before/after 对比 → 行为一致再切流
```

## B.4 常见升级报错速查

| 报错 | 原因与解法 |
|---|---|
| `ImportError: MemorySaver` | 用新名 InMemorySaver；或旧版残留缓存，清理环境 |
| `ModuleNotFoundError: langgraph.checkpoint.sqlite` | 装独立包 `langgraph-checkpoint-sqlite` |
| stream messages 解包错误 | 1.x 返回元组，改 `for chunk, meta in ...` |
| `interrupt() called outside node` | interrupt 只能在节点/entrypoint 内调用；HITL 工具栏逻辑移进 pre_model_hook |
| `InvalidUpdateError` 增多 | 1.x 并发校验更严——给被并发写的字段补 reducer |

## B.5 版本兼容策略（给维护者）

- `langgraph>=1.0,<2.0` 锁主版本；小版本升级看 changelog 的 deprecation 提示
- 提示与配置资产放 Hub/Assistant（第 24/27 章），与库版本解耦，降低升级面
- 长期不升级的旧系统：至少把 checkpointer 拆包变更解决掉（否则 0.x 旧版与新版 Python 生态冲突）
