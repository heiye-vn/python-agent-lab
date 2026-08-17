# 第 24 章：LangGraph Platform / Server 架构

LangGraph Platform 是官方的**部署与运行时层**。核心是一个常驻的 **LangGraph Server**，把你的图变成生产级 API 服务。本章讲透它的对象模型与运行机制——这是企业部署篇的理论地基。

## 24.1 为什么不直接 FastAPI 包一层？

自己包 FastAPI 也能跑图，但很快会撞上这些墙：

| 自建 FastAPI 的墙 | LangGraph Server 的答案 |
|---|---|
| 长任务占 HTTP 连接；进程重起任务就丢 | **Runs 异步化** + 持久化，任务与请求解耦 |
| 断线后流式无法续传 | 流式支持 join/resume 重连 |
| 多实例无法共享对话状态 | 状态外置（Postgres/Redis），**水平扩展** |
| 没有统一的 thread/assistant 管理 | 完整 REST API + 多语言 SDK |
| 定时任务、Webhook、鉴权都要自己写 | 内置 Cron、Webhook、路径/资源级 auth |

一句话：**LangGraph Server 把"运行 Agent 的通用难题"变成了平台配置**。

## 24.2 对象模型：五大核心概念

```
Agent（代码里的图）
   └── Assistant（图的一个可调用配置版本：模型参数/提示的具名组合）
         └── Thread（一段会话/任务的状态容器）
               └── Run（thread 上的一次执行，可多次）
Store（跨 thread 的长期记忆）
Cron（定时触发的 run）
```

### Agent
`langgraph.json` 里 `graphs` 暴露的图。部署后是"能力"本身。

### Assistant（助手的"配置版本"）
同一张图，不同参数 = 不同 assistant：

```python
# SDK 预览
client.assistants.create(
    graph_id="my_agent",
    config={"configurable": {"model": "gpt-4o-mini", "tone": "formal"}},
    name="客服-标准版",
)
```

用途：灰度新提示（新 assistant）、A/B（两 assistant 各跑 50%）、为不同租户出不同配置——**不改代码切换行为**。

### Thread
第 11 章的 thread 在 Server 侧成为一等公民：可列出、检索、暂停/恢复、删除。生产把业务工单 ID 与 thread_id 绑定即可拥有完整会话历史。

### Run：一次执行
Run 是理解 Server 的关键，它有**三种模式**：

| 模式 | 行为 | 场景 |
|---|---|---|
| **后台（background）** | 立即返回 run_id，任务在后台跑，随时查状态/取流 | 长任务、批处理 |
| **流式（stream / duplex）** | HTTP 连接内流式返回（duplex 双向流：可中途发新输入） | 聊天 UI |
| **无状态（stateless）** | 不落 thread、不持久化，直接结果 | 高频低成本调用、函数式用法 |

Run 的状态机：`pending → started（执行中）→ success | error | interrupted | timeout | canceled`。注意 `interrupted`：HITL 暂停的 run 就停在这里，等 `Command(resume=...)` 的新 run 接力。

## 24.3 执行模型：任务如何"悬浮"又"续跑"

一次后台 run 的生命周期：

```
POST /threads/{id}/runs        → Server 入队，立即返回 run_id
     ↓ worker 取出 → 在新 thread 快照上执行你的图
     ├─ 图跑完 → 结果与最终状态落盘 → run=success
     ├─ 图 interrupt → 状态落盘 → run=interrupted（HTTP 已早返回了）
     │    （几小时后）POST /threads/{id}/runs  {"command": {"resume": "approve"}}
     │        → 新 run 从断点继续（同一 thread）
     ├─ 崩溃 → 任务标记失败，可重试（断点续跑，见 11.4）
     └─ 超时/取消 → timeout/canceled
```

**关键理解：暂停的 run 不占任何 worker**。等待人工的那 8 小时里，服务器资源为零。这是 HITL 能进生产的根本原因。

## 24.4 水平扩展原理

```
             ┌─ API pod ─┐
请求 → LB →  │ API pod   │   ← 无状态，随便扩
             └─ API pod ─┘
                    │ 任务入队
             ┌─ worker pod ┐
             │ worker pod  │  ← 从队列取 run 执行，随便扩
             └─ worker pod ┘
                    │
        Postgres（checkpoints/threads/runs）+ Redis（队列/发布订阅）
        对象存储（可选，大产物）
```

- 状态全在数据库 → 任何 worker 都能续跑任何 thread
- 流式事件经 Redis 发布订阅 → 你连着的那个 API pod 能把 worker 的 token 转发给你（**断线换 pod 还能 join 回来**）
- 这是 Control Plane（调度/状态）与 Data Plane（执行你的图）分离的架构

## 24.5 定时任务 Cron

Server 内置调度器，让"无输入"的图定期执行：

```python
client.crons.create(
    thread_id=None,                    # 可无 thread（stateless 模式）
    assistant_id="daily_report",
    schedule="0 8 * * *",              # cron 表达式（UTC）
    input={"messages": [("user", "生成昨日运营日报")]},
)
```

典型用途：每日报告、记忆后台提炼（第 12 章）、HITL 超时扫描（第 15 章）。

## 24.6 部署形态总览（细节在下一章）

| 形态 | 说明 | 适合 |
|---|---|---|
| **LangGraph Cloud** | 官方全托管 | 快速上线、不想运维 |
| **自托管（Self-hosted Data Plane）** | `langgraph build` 出 Docker 镜像，自己跑在 K8s | 数据合规、企业内网 |
| **本地开发** | `langgraph dev`（第 23 章） | 开发调试 |

三种形态 API 完全一致——本地开发与生产部署之间没有迁移成本。

## 24.7 安全与扩展点（速览，第 26 章实操）

- **自定义认证**：路径级（谁能调哪些 API）、资源级（谁能访问哪些 thread/assistant/run）
- **自定义 REST API**：`src/api/` 目录写 FastAPI 路由，与图的 API 同进程部署
- **Webhook**：run 状态变化时回调你的业务系统
- **把 Agent 暴露为 MCP endpoint**：其他 Agent 系统可把你的图当工具用

## 本章小结

- 对象模型：Agent（图）→ Assistant（配置版本）→ Thread（会话）→ Run（执行）
- Run 三模式：后台 / 流式(duplex) / 无状态；interrupted 是 HITL 的稳态
- 暂停的任务不占资源；状态外置数据库 → 水平扩展与断线重连
- Cron 内置；三种部署形态 API 同构

> 下一章：REST API 与 SDK 实操——把第 10 章的流式、第 14 章的 HITL 通过 Server 跑通。
