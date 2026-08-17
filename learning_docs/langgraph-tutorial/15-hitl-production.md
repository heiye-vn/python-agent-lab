# 第 15 章：HITL 工程化

第 14 章讲了机制；本章讲**产品与工程**：暂停的图如何通知人、前端如何展示与收集决定、审批如何留痕、超时怎么办。理解本章可以直接落地企业审批流。

## 15.1 完整交互时序（前后端协作全景）

```
用户端                后端服务               LangGraph              审批人端
  │  发起任务             │                     │                     │
  │────────────────────>│  invoke()           │                     │
  │                     │────────────────────>│  ...跑至 interrupt() │
  │                     │   返回+__interrupt__ │  落盘暂停            │
  │                     │<────────────────────│                     │
  │  返回 task_id        │                     │                     │
  │  ("已提交，等待审批")  │─── 通知（IM/邮件/工单）───────────────────────>│
  │                     │                     │                     │ 审批
  │                     │  resume(Command)    │                     │<─┐
  │                     │────────────────────>│  从断点恢复           │  │
  │  轮询/推送拿结果      │   stream 结果       │  ...跑完             │  │
  │<────────────────────│<────────────────────│                     │  │
```

后端要做的四件事：

1. **发起**：为用户创建 thread、发起 run、把 thread_id/run_id 关联到业务工单
2. **捕获暂停**：从 stream 或 `get_state` 拿到 `__interrupt__` payload（审批材料）
3. **通知**：把 payload 推给审批人（IM webhook / 邮件 / 审批中心页面）
4. **恢复**：审批人提交决定 → `invoke(Command(resume=...), 同 thread config)`

## 15.2 后端服务参考实现（FastAPI）

```python
# 与第 14 章的 approval graph 配套的最小后端
from fastapi import FastAPI
from pydantic import BaseModel
from langgraph.types import Command

app = FastAPI()


@app.post("/tickets")
def create_ticket(q: str, user_id: str):
    """创建工单：跑图直到 interrupt，返回审批材料"""
    config = {"configurable": {"thread_id": f"ticket-{user_id}-{uuid4().hex[:8]}"}}
    result = graph.invoke({"messages": [("user", q)]}, config)

    state = graph.get_state(config)
    payload = state.tasks[0].interrupts[0].value   # {"question":..., "draft":...}
    # TODO: 存 payload 到工单表，推送审批人
    return {"thread_id": config["configurable"]["thread_id"], "review": payload}


class Decision(BaseModel):
    thread_id: str
    action: str          # approve | reject
    comment: str = ""


@app.post("/tickets/decide")
def decide(d: Decision):
    """审批人提交决定：恢复图"""
    config = {"configurable": {"thread_id": d.thread_id}}
    result = graph.invoke(Command(resume={"action": d.action, "comment": d.comment}), config)
    return {"final": result["messages"][-1].content}
```

生产上，这两个接口通常由 LangGraph Server 的 Threads/Runs API + 自定义业务端点共同承担（第 24-25 章），但时序完全一致。

## 15.3 审批 UI 设计要点

`interrupt(payload)` 的 payload 就是给 UI 的**数据契约**，建议标准化：

```python
interrupt({
    "type": "approval",                  # UI 按类型渲染不同组件
    "title": "退款审批",
    "summary": "客户要求退款 ¥3,200，AI 已生成回复",
    "fields": {"订单号": "SO-2026-8891", "风险分": 0.23},
    "draft": "尊敬的客户，您的退款申请已受理……",
    "actions": [                          # 告诉 UI 提供哪些按钮
        {"id": "approve", "label": "批准", "style": "primary"},
        {"id": "reject", "label": "拒绝", "style": "danger"},
        {"id": "edit",    "label": "修改后批准", "style": "default"},
    ],
})
```

resume 值同样标准化：`{"action": "approve"|"reject"|"edit", "content": "..."}`。

## 15.4 留痕与审计（合规刚需）

- **审批前快照**：暂停时把 `checkpoint_id` 记入审计日志——事后可精确回放"当时 AI 看到了什么"
- **决定落库**：谁（approver_id）、何时、决定、备注、对应 thread/checkpoint
- **恢复来源校验**：resume 接口必须校验"当前用户是否有该 thread 的审批权限"（配合第 26 章的 thread 级鉴权）
- 拒绝路径的 resume 值也写进状态，让模型据此调整话术而非硬闯

## 15.5 超时与升级策略

人可能一直不审批。常用三级策略（在图外加一个调度器实现）：

```python
# 1. 定时扫描"暂停超过 N 小时"的 thread
for t in db.query_pending_threads(older_than="2h"):
    # 2. 升级：通知上级（业务系统操作）
    notify_escalation(t)
    # 3. 超时自动策略：可用 resume 传入超时决定，或直接 update_state 后 invoke(None)
    if t.age > "24h":
        graph.invoke(Command(resume={"action": "reject", "comment": "超时自动驳回"}), t.config)
```

也可以用 LangGraph 的 **Cron 任务**（第 24 章）在 Server 侧定时扫描。

## 15.6 并发审批与幂等

- 同一 thread 的恢复调用要做**幂等保护**：后端对 (thread_id, checkpoint) 加锁/唯一约束，防止两个审批人同时 resume
- 恢复后若又遇到下一个 interrupt，循环 15.1-15.3 的流程即可——多级审批 = 多个 interrupt 节点串联

## 15.7 决策表：HITL 放在哪一层

| 需求 | 方案 |
|---|---|
| 危险工具调用前确认 | pre_model_hook / Middleware + interrupt（第 17/18 章） |
| 最终产出发布前审阅 | 独立审批节点 + interrupt |
| 运行中人工纠偏 | `update_state` + `invoke(None)` |
| 每次都停（调试） | `interrupt_before` 编译参数 |
| 长时间等待 + 超时升级 | 本章调度器 / Cron |
| 人工可以直接改 AI 输出后放行 | resume 携带编辑后内容，节点内用其覆盖 |

## 本章小结

- HITL 产品闭环四步：发起 → 捕获暂停 → 通知 → 恢复
- payload / resume 值做**标准化数据契约**，UI 与图解耦
- 审计三件套：checkpoint_id 留痕、决定落库、恢复鉴权
- 超时升级靠外部调度器或 Server Cron；并发恢复要幂等加锁

> 第五部分完成。接下来进入 Agent 构建主战场：工具、create_react_agent、Middleware、MCP。
