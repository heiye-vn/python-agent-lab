# 第 14 章：interrupt 机制详解

Human-in-the-Loop（HITL）= 在图执行的任意点暂停、等人、带着人的决定继续。LangGraph 1.x 的标准答案是节点内 `interrupt()` 函数。本章从机制到模式完整讲透。

## 14.1 为什么 HITL 不简单

朴素做法（后台轮询/外部消息队列）的难点在于：**LLM 应用暂停可能长达几天**（等审批人回来），HTTP 请求早超时了，进程也不能干等着。

LangGraph 的解法：**interrupt 不是"阻塞等待"，而是"安全落盘 + 干净退出"**：

```
invoke → node_a → node_b 执行到 interrupt() 
  → 状态已持久化（含"我停在 node_b"）→ 本次 invoke 正常返回
  ……（几小时后，甚至换了一台服务器）
invoke(Command(resume="批准"), 同thread) → 从断点恢复，resume 值注入 node_b
```

暂停的图不占任何资源；恢复与暂停可以在不同进程、不同机器。

## 14.2 interrupt() 基本用法

```python
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver

class State(TypedDict):
    messages: Annotated[list, add_messages]
    approval: str


def human_approval(state: State) -> dict:
    # interrupt(payload)：payload 是发给人的"问题材料"，会出现在状态快照里
    decision = interrupt({
        "question": "以下是 AI 生成的退款回复，是否发出？",
        "draft": state["messages"][-1].content,
    })
    # ↓ 只有恢复执行时才会走到这里，decision = resume 传入的值
    if decision == "approve":
        return {"approval": "approved"}
    return {"approval": "rejected"}


builder = StateGraph(State)
builder.add_node("draft_reply", draft_reply)
builder.add_node("human_approval", human_approval)
builder.add_node("send_email", send_email)
builder.add_edge(START, "draft_reply")
builder.add_edge("draft_reply", "human_approval")
builder.add_conditional_edges("human_approval",
    lambda s: "send_email" if s["approval"] == "approved" else END)
builder.add_edge("send_email", END)
graph = builder.compile(checkpointer=InMemorySaver())   # HITL 必须有 checkpointer！

config = {"configurable": {"thread_id": "ticket-9527"}}

# 第一跳：跑到 human_approval 处暂停
result = graph.invoke({"messages": [("user", "我要退款")]}, config)
print(graph.get_state(config).next)      # ('human_approval',) ← 停在这
print(graph.get_state(config).tasks[0].interrupts)  # payload 在这里，给前端展示

# 第二跳：人工决定后恢复
graph.invoke(Command(resume="approve"), config)
# decision == "approve"，继续走 send_email
```

三个关键点：
1. **必须配 checkpointer**（暂停状态要落盘）
2. `interrupt(payload)` 的 payload 是给人看的信息，通过 `get_state().tasks` 或 stream 的 `__interrupt__` 事件取出来给前端渲染
3. 恢复 = `invoke(Command(resume=值), 同config)`；resume 值会成为 interrupt() 调用的**返回值**

## 14.3 检测与消费 interrupt（前端对接）

```python
# 方式一：stream 里拿到 __interrupt__（推荐，配合前端推送）
for chunk in graph.stream(inputs, config, stream_mode="updates"):
    if "__interrupt__" in chunk:
        for intr in chunk["__interrupt__"]:
            print("暂停了：", intr.interrupt_id, intr.value)  # value=payload

# 方式二：事后查询状态
state = graph.get_state(config)
if state.next:
    for task in state.tasks:
        if task.interrupts:
            print(task.interrupts[0].value)
```

`interrupt_id` 可用于多 interrupt 场景精确恢复：`Command(resume=..., resume_interrupt_id=intr.interrupt_id)`? 实际生产建议**每个节点一次 interrupt、循环内 interrupt 索引自增**（见 14.5）。

## 14.4 老方式对比：interrupt_before / interrupt_after

```python
graph = builder.compile(
    checkpointer=cp,
    interrupt_before=["send_email"],   # 进入 send_email 之前停
    # interrupt_after=["draft_reply"], # 某 node 跑完后停
)
graph.invoke(inputs, config)          # 停在 send_email 前
graph.update_state(config, {...})     # 可选：人工改状态
graph.invoke(None, config)            # 继续执行 send_email
```

| | `interrupt()`（新） | `interrupt_before/after`（老） |
|---|---|---|
| 粒度 | 节点内任意处、可携带问题、可拿回复 | 节点边界 |
| 数据交互 | payload → 人 → resume 值 | 只能靠 update_state |
| 声明位置 | 业务代码里 | compile 参数 |
| 适用 | 审批、确认、收集信息（**默认选它**） | 通用断点、调试 |

**规则**：写业务用 `interrupt()`；"每次到这里都想停下来看看"用编译参数（更像 debugger 断点）。

## 14.5 高级模式

### 模式一：多轮收集信息

```python
def collect_info(state):
    answers = []
    for question in ["你的订单号？", "退款原因？"]:
        ans = interrupt({"question": question})
        answers.append(ans)
    return {"answers": answers}
```

注意：恢复时**节点从头重跑**，已完成的 interrupt 会直接返回历史 resume 值（LangGraph 自动按调用序号匹配）——所以循环写法是安全的。

### 模式二：动态打断 NodeInterrupt

```python
from langgraph.errors import NodeInterrupt

def check_safety(state):
    if risk_score(state["draft"]) > 0.8:
        raise NodeInterrupt({"reason": "风险分过高，需人工复核", "score": ...})
    return {}
```

与 `interrupt()` 的区别：NodeInterrupt 是**条件性异常式打断**（不需要人回复具体值，处理后 `invoke(None)` 继续）。

### 模式三：修改后再继续

```python
# 暂停后人工编辑草稿，再继续
graph.update_state(config, {"messages": [AIMessage(content="人工改过的草稿", id=draft_id)]})
graph.invoke(None, config)
```

### 模式四：跳过/重定向

```python
# 人工说"这单别发了，转人工客服"
graph.update_state(config, {"handoff": True})
graph.invoke(None, config)   # 条件边根据 handoff 走人工分支
```

## 14.6 Agent 审批工具调用的标准做法（预览）

生产中最常见的 HITL 需求——"危险工具需审批"，用 `pre_model_hook`（第 17 章）或 Middleware（第 18 章）在工具执行前 intercept。完整实现见第 29 章项目一。骨架：

```python
def pre_model_hook(state):
    if has_dangerous_pending_tool_call(state):
        decision = interrupt({"tool_call": ...})
        if decision["action"] == "reject":
            # 把工具调用改写为"用户拒绝"，模型会看到并调整
            return Command(update={"messages": [ToolMessage("用户拒绝了该操作", ...)]})
    return Command(update={})
```

## 本章小结

- HITL 本质 = 持久化暂停 + 断点恢复，可跨进程跨天
- `interrupt(payload)` → 人 → `Command(resume=...)`，payload 给前端、resume 值回代码
- 必须 checkpointer；恢复时节点重跑、历史 interrupt 自动回放
- 老三样补充：`interrupt_before/after`（调试断点）、`NodeInterrupt`（条件异常打断）、`update_state`（人工改状态）

> 下一章：把 HITL 变成真实产品交互的工程化方案。
