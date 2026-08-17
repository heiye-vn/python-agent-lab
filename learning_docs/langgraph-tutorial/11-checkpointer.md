# 第 11 章：Checkpointer 与短期记忆

Checkpointer（检查点器）是 LangGraph 的"企业级卖点"：**每个 superstep 自动把状态存档**，由此获得对话记忆、崩溃恢复、HITL、时间旅行四大能力。本章讲透原理与生产用法。

## 11.1 Thread：一切的坐标

- 一个 **thread**（由 `configurable.thread_id` 标识）= 一段持续演进的应用状态（通常是一段会话/一个任务）
- checkpointer 按 thread 存储**一串快照（checkpoint）**，每个 superstep 一个
- 同一 thread 再次 invoke 时，从**最新快照**接着跑

```python
config = {"configurable": {"thread_id": "session-42"}}
graph.invoke({"messages": [("user", "第一句")]}, config)
graph.invoke({"messages": [("user", "第二句")]}, config)   # 接着上一轮状态
```

thread_id 怎么生成？常见做法：每个用户会话一个（UUID），存到你的 session 表里。

## 11.2 内置 Checkpointer 选型

| Checkpointer | 包 | 用途 |
|---|---|---|
| `InMemorySaver` | 内置 | 开发/测试。重启即失 |
| `SqliteSaver` | `langgraph-checkpoint-sqlite` | 单机原型、轻量场景 |
| `PostgresSaver` | `langgraph-checkpoint-postgres` | **生产标配** |
| `RedisSaver` | `langgraph-checkpoint-redis` | 高吞吐、低延迟场景 |

```python
# SQLite（本地开发）
from langgraph.checkpoint.sqlite import SqliteSaver
with SqliteSaver.from_conn_string("checkpoints.db") as cp:
    graph = builder.compile(checkpointer=cp)

# Postgres（生产）
# pip install psycopg pgvector psycopg-pool "langgraph-checkpoint-postgres"
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

pool = ConnectionPool(
    conninfo="postgresql://user:pass@localhost:5432/mydb",
    max_size=20,
    kwargs={"autocommit": True},
)
checkpointer = PostgresSaver(pool)
checkpointer.setup()            # 首次运行建表（生产跑一次即可）
graph = builder.compile(checkpointer=checkpointer)
```

**注意**：checkpointer 表和业务表建议同库，方便运维一致性；`setup()` 重复执行是安全的（幂等）。

## 11.3 一个 Checkpoint 里有什么

```python
snapshot = graph.get_state(config)
snapshot.values      # 该时刻的完整 State（dict）
snapshot.next        # 下一批将执行的节点；非空说明图"停"在这里（interrupt / 未跑完）
snapshot.tasks       # 待执行任务详情（PENDING / 含 interrupt 信息）
snapshot.metadata    # {"step": 12, "source": "loop", "writes": {...}, "parents": ...}
snapshot.config      # 含 checkpoint_id 的完整 config
snapshot.created_at  # 时间戳
```

理解结构后，"记忆/恢复/HITL/时间旅行"都只是对快照链的不同操作：

```
checkpoint_0 → checkpoint_1 → checkpoint_2 → ... → checkpoint_n（最新）
       │            │              │
     输入后        node_a 后      node_b 后
```

## 11.4 崩溃恢复：durable execution 实战

```python
# 进程 A：跑到一半崩了（比如 node_b 执行中 OOM）
graph.invoke(inputs, config)   # 异常退出

# 进程 B：重启后，同 thread 再跑——从最后完整快照恢复，重执行未完成节点
graph.invoke(None, config)     # 注意输入传 None = "继续跑，不要新输入"
```

关键语义：
- **已完成 superstep 的节点不会重跑**（副作用安全）
- **执行到一半的节点会整个重跑**（节点是持久化原子单位）→ **节点内副作用要幂等，或放到确认成功之后**
- `invoke(None, config)` 是"继续"的标准姿势，HITL resume 也用它（配合 `Command(resume=...)`）

异步任务恢复：进程内叫 durability；跨进程/服务器的长任务恢复需要 LangGraph Server 的 runs 机制（第 24 章）。

## 11.5 短期记忆管理：裁剪与摘要

消息无限增长 → token 爆炸。两种标准策略（可叠加）：

### 策略一：滑动窗口裁剪 trim

```python
from langchain_core.messages import trim_messages

def trim_messages_strategy(messages):
    return trim_messages(
        messages,
        max_tokens=4000,                    # 预算
        strategy="last",                    # 保留最近的
        token_counter=llm,                  # 用模型自己的 tokenizer 数
        start_on="human",                   # 保证从 human 消息开始（对话完整性）
        include_system=True,                # system prompt 永远保留
    )

# 在 pre_model_hook（第 17 章）或节点入口调用：
def agent_node(state):
    trimmed = trim_messages_strategy(state["messages"])
    response = llm.invoke(trimmed)
    return {"messages": [response]}
```

### 策略二：滚动摘要 summarize

```python
async def maybe_summarize(state):
    if num_tokens(state["messages"]) < 6000:
        return None                          # 不用摘要

    summary_prompt = (
        f"把以下对话渐进式地压缩为摘要，保留关键事实与用户偏好：\n"
        f"旧摘要：{state.get('summary', '（无）')}\n\n对话：\n{tail(state['messages'], 20)}"
    )
    new_summary = await llm.ainvoke(summary_prompt)

    return {
        "messages": [RemoveMessage(id=m.id) for m in state["messages"][:-2]],  # 清掉旧消息
        "summary": new_summary.content,      # 摘要存进自定义字段
    }
```

**企业标配组合**：system prompt + 长期摘要 + 最近 N 轮原文。

## 11.6 管理检查点（存储卫生）

```python
# 删除一个 thread 的全部历史（GDPR / 用户注销场景）
graph.checkpointer.delete_thread(thread_id)

# 查看历史长度
len(list(graph.get_state_history(config)))
```

生产建议：
- checkpoints 表会持续增长，Postgres 上配置定期归档/清理策略
- 只存必要的 State 字段——大对象（原始文档、图片 base64）放对象存储，State 里只放 URL/引用

## 11.7 自定义 Checkpointer（何时需要）

实现 `BaseCheckpointSaver` 接口（get_tuple / list / put / delete 等）即可接 MySQL、Mongo、云数据库。**只有当公司标准化中间件不在支持列表时才写**；99% 场景 Postgres/Redis 够用。

## 本章小结

- thread_id 定位会话；checkpointer 每 superstep 存一档
- 生产选型：Postgres（标配）/ Redis（低延迟高并发）；`setup()` 建表幂等
- `invoke(None, config)` 恢复中断的图；节点是重试原子单位——副作用要幂等
- 短期记忆两策略：trim 滑窗 + summarize 滚动摘要
- 存储卫生：删 thread、大对象外置

> 下一章：长期记忆 Store——让 Agent 跨会话"认识"用户。
