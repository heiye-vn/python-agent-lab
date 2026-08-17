# 第 32 章：项目四 —— 企业审批工作流

**技能点**：确定性骨架+智能节点架构（第 17/20 章）+ 多级 interrupt（第 14 章）+ 时间旅行回滚（第 13 章）+ Server 部署与鉴权（第 25/26 章）。这是收官项目：**从画图到上线全流程**。

## 32.1 需求

合同审批流：员工提交合同 → **风控 Agent** 预审（AI 判断风险）→ 按金额分级路由：

```
金额 < 1万      → 一级审批（直属主管，interrupt）
1万 ≤ 金额 <50万 → 二级审批（主管 + 法务，两级 interrupt）
金额 ≥ 50万     → 三级审批（主管 + 法务 + CFO）+ 高管留言
全程可回滚（时间旅行）、留痕、超时自动升级
```

关键设计认知：**这是"流程引擎"任务，主体必须是确定性 StateGraph；LLM 只出现在"风控预审"这一个智能节点**——不是所有环节都该 Agent 化（第 20 章反面清单）。

## 32.2 完整实现

### graph.py

```python
from typing import Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import NodeInterrupt

llm = init_chat_model("openai:gpt-4o-mini")


class ContractState(TypedDict):
    contract_id: str
    title: str
    amount: float
    content: str
    risk_score: float          # 风控 Agent 产出
    risk_notes: str
    approvals: Annotated[list, lambda a, b: (a or []) + (b or [])]  # 审批记录累积
    current_level: int
    final_status: str


class RiskAssessment(BaseModel):
    risk_score: float = Field(ge=0, le=1)
    risk_notes: str


# ── 智能节点：风控预审（全图唯一 LLM 决策点）──
def risk_review(state: ContractState) -> Command:
    assessment = llm.with_structured_output(RiskAssessment).invoke(
        f"评估合同风险（0=安全 1=高危）：\n标题:{state['title']}\n金额:{state['amount']}\n"
        f"内容:{state['content'][:2000]}"
    )
    # 高危硬规则：AI 说了不算，规则说了算（第 28 章安全心法）
    if assessment.risk_score > 0.85:
        raise NodeInterrupt({
            "type": "risk_block",
            "message": "风控预审未通过，需风控部人工介入",
            "risk": assessment.risk_notes,
        })
    level = 3 if state["amount"] >= 500_000 else (2 if state["amount"] >= 10_000 else 1)
    return Command(
        goto="manager_approval",
        update={"risk_score": assessment.risk_score,
                "risk_notes": assessment.risk_notes,
                "current_level": level},
    )


# ── 确定性节点：各级审批（同构 interrupt，级别决定审批人）──
APPROVERS = {
    1: ["manager"],
    2: ["manager", "legal"],
    3: ["manager", "legal", "cfo"],
}

def manager_approval(state) -> Command:
    return _one_step(state, "manager")

def legal_approval(state) -> Command:
    return _one_step(state, "legal")

def cfo_approval(state) -> Command:
    return _one_step(state, "cfo")


def _one_step(state: ContractState, role: str) -> Command:
    chain = APPROVERS[state["current_level"]]
    idx = chain.index(role)

    decision = interrupt({
        "type": "approval",
        "contract_id": state["contract_id"],
        "step": f"{idx + 1}/{len(chain)}",
        "approver_role": role,
        "amount": state["amount"],
        "risk_notes": state["risk_notes"],
    })

    record = {"role": role,
              "action": decision.get("action"),
              "comment": decision.get("comment", "")}

    if decision.get("action") == "reject":
        return Command(goto="finalize",
                       update={"approvals": [record], "final_status": "rejected"})

    if idx + 1 < len(chain):                        # 还有下一级
        next_node = f"{chain[idx + 1]}_approval"
        return Command(goto=next_node, update={"approvals": [record]})
    return Command(goto="finalize",
                   update={"approvals": [record], "final_status": "approved"})


def finalize(state: ContractState):
    if state["final_status"] == "approved":
        # 真实系统：盖章/归档/通知（幂等！）
        pass
    return {"final_status": state["final_status"]}


builder = StateGraph(ContractState)
builder.add_node("risk_review", risk_review)
for r in ("manager", "legal", "cfo"):
    builder.add_node(f"{r}_approval", globals()[f"{r}_approval"])
builder.add_node("finalize", finalize)
builder.add_edge(START, "risk_review")
builder.add_edge("finalize", END)
# 其余跳转全部由 Command 驱动

workflow = builder.compile(checkpointer=InMemorySaver())
```

### main.py —— 走一遍三级审批

```python
from langgraph.types import Command
from graph import workflow

config = {"configurable": {"thread_id": "CT-2026-001"}}

# ── 提交：金额 68 万 → 三级审批 ──
workflow.invoke({
    "contract_id": "CT-2026-001", "title": "年度云服务采购",
    "amount": 680_000, "content": "向 X 云采购 3 年服务……",
    "approvals": [], "current_level": 0, "final_status": "",
}, config)

state = workflow.get_state(config)
print("停在:", state.next, state.tasks[0].interrupts[0].value["approver_role"])

# ── 主管批 ──
workflow.invoke(Command(resume={"action": "approve", "comment": "同意"}), config)
# ── 法务批 ──
workflow.invoke(Command(resume={"action": "approve", "comment": "条款合规"}), config)
# ── CFO 打回（要求降价）──
workflow.invoke(Command(resume={"action": "reject", "comment": "预算超了"}), config)

print(workflow.get_state(config).values["final_status"])   # rejected
```

### 回滚重审（时间旅行实战）

CFO 拒绝后，业务重新谈判降价到 45 万（降级为二级审批），**从法务审批点分叉重来**：

```python
# 1. 找到"法务审批前"的历史快照
history = list(workflow.get_state_history(config))
legal_checkpoint = [s for s in history
                    if "legal_approval" in (s.next or ())
                    and s.values.get("current_level") == 3][-1]

# 2. fork 到新 thread：改金额 → 从法务继续（级别重算为 2？—— 演示直接续）
fork_config = {"configurable": {
    "thread_id": "CT-2026-001-R2",
    "checkpoint_id": legal_checkpoint.config["configurable"]["checkpoint_id"],
}}
workflow.update_state(fork_config, {"amount": 450_000})
workflow.invoke(None, fork_config)      # 从法务审批点继续
```

原审批链 CT-2026-001 完整保留（审计），新谈判在 R2 线上推进——**这就是"图引擎版流程回滚"**。

## 32.3 部署上线（第 24-26 章串联）

```json
// langgraph.json
{ "graphs": { "contract_approval": "./src/graph.py:workflow" } }
```

```python
# 业务系统对接（SDK，第 25 章）
from langgraph_sdk import get_client
client = get_client(url="https://approval.internal.myco.com")

thread = await client.threads.create(metadata={"contract_id": "CT-2026-001"})
# 发起 → 后台 run，interrupt 后 Webhook 回调审批中心
await client.runs.create(thread["thread_id"], "contract_approval", input={...})
# 审批中心按钮 → resume
await client.runs.wait(thread["thread_id"], "contract_approval",
                       command={"resume": {"action": "approve", "comment": "..."}})
```

- **鉴权**：`@auth.on.threads.create` 校验"审批人角色 ↔ approver_role"匹配（第 26 章）
- **留痕**：每次 resume 前 `get_state` 记 checkpoint_id 入审计表（第 15 章）
- **超时升级**：Cron 扫描 `interrupted` 超 24h 的 thread → 通知上级（第 24 章）

## 32.4 测试要点（第 23 章）

```python
def test_high_risk_blocks():
    """风控分>0.85 → NodeInterrupt，不进入审批"""
    graph = build_with_fake_llm(RiskAssessment(risk_score=0.95, risk_notes="阴阳合同"))
    with pytest.raises(NodeInterrupt):
        graph.invoke(base_inputs, config)

def test_amount_routing():
    """金额决定级别：68万→cfo 在审批链；8千→仅 manager"""
    ...

def test_reject_short_circuit():
    """任何一级 reject → finalize，后续级别不再触发"""
    ...
```

## 32.5 为什么这个项目是收官

它把"企业级"三个字拆成了可复用的模式：

- **确定性与智能的边界**：流程骨架零 LLM，风控点 LLM 结构化输出 + 规则硬门禁
- **多级 HITL**：同构审批节点 + Command 串联，N 级审批只写一次
- **时间旅行**不是玩具：分叉重审 + 原链留痕 = 真实业务回滚
- **Server 化**：Webhook 驱动审批中心、Cron 驱动超时、auth 驱动权限——平台能力各就各位

## 本章小结

- 确定性 StateGraph 骨架 + 单点智能（风控）+ 规则硬门禁 = 可审计的企业流程
- 多级审批 = 同构 interrupt 节点链 + Command 路由
- 时间旅行分叉重审保留原审计链
- Server 侧 Webhook/Cron/auth 组合完成企业闭环

> 实战篇完结。附录：JS/TS 差异、迁移指南、踩坑集、面试题、资源。
