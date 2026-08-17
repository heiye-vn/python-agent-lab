# 第 31 章：项目三 —— 多 Agent 研究助手

**技能点**：Supervisor（第 21 章）+ Send 并行研究（第 7 章）+ 上下文隔离（第 20 章）+ deepagents（第 22 章）+ 长任务持久化（第 11 章）。

## 31.1 需求与架构

"输入一个研究课题 → 多角度并行调研 → 汇总成报告"。两种实现：**A. 手写 Supervisor + Send（教学版，看清机制）**；**B. deepagents（生产版，10 行）**。

```
用户课题 → planner（拆角度）
              ├─ Send → researcher[技术] ┐
              ├─ Send → researcher[市场] ├ 并行（各自独立上下文！）
              └─ Send → researcher[风险] ┘
                        ↓ operator.add 汇聚
          reviewer（审稿，不合格打回）→ writer → 报告
```

## 31.2 实现A：手写教学版

### 状态与角色

```python
import operator
from typing import Annotated, Literal
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command, Send
from langgraph.checkpoint.memory import InMemorySaver

llm = init_chat_model("openai:gpt-4o-mini")
cheap_llm = init_chat_model("openai:gpt-4o-mini")   # 演示用同款；生产换小模型


class ResearchState(MessagesState):
    topic: str
    angles: list = []                                        # planner 拆出的角度
    findings: Annotated[list, operator.add] = []             # 各研究员成果汇聚
    draft: str = ""
    review_passed: bool = False


class Plan(BaseModel):
    angles: list[str] = Field(description="3 个互补的研究角度，每个不超过 8 字")
```

### planner：拆角度 → Send 并行分发

```python
def planner(state: ResearchState):
    plan = llm.with_structured_output(Plan).invoke(
        f"研究课题：{state['topic']}。拆解为 3 个互补角度。"
    )
    # Command + Send：为每个角度派发一个独立 researcher 实例
    return Command(
        update={"angles": plan.angles},
        goto=[Send("researcher", {"angle": a, "topic": state["topic"]})
              for a in plan.angles],
    )


def researcher(payload: dict):
    """注意：收到的是 Send 携带的独立输入，不是全图状态 —— 上下文隔离的关键。
    这里的 Agent 可以读 50 个网页，垃圾上下文不会污染主流程。"""
    angle, topic = payload["angle"], payload["topic"]
    # 模拟"搜索+阅读+提炼"（真实系统：web_search 工具的 create_react_agent）
    finding = cheap_llm.invoke(
        f"你是研究助理。从「{angle}」角度分析课题「{topic}」,"
        f"输出 5 条带来源标注的要点，总字数<300。"
    ).content
    return {"findings": [f"## {angle}\n{finding}"]}   # operator.add 自动汇聚
```

### reviewer：质量门禁（可打回重写）

```python
class Review(BaseModel):
    verdict: Literal["pass", "revise"]
    comments: str

def writer(state: ResearchState):
    draft = llm.invoke(
        f"课题：{state['topic']}\n资料：\n" +
        "\n".join(state["findings"]) +
        "\n\n写一份 800 字结构化报告（结论先行/数据支撑/风险提示）。"
    ).content
    return {"draft": draft}

def reviewer(state: ResearchState) -> Command:
    review = llm.with_structured_output(Review).invoke(
        f"审阅报告草稿，检查：事实支撑、结构、是否覆盖全部角度。\n\n{state['draft']}"
    )
    if review.verdict == "pass":
        return Command(goto="finalize", update={"review_passed": True})
    return Command(
        goto="writer",                                     # 打回重写
        update={"messages": [HumanMessage(f"审稿意见：{review.comments}")]},
    )

def finalize(state: ResearchState):
    return {"messages": [HumanMessage(state["draft"], name="report")]}
```

### 组装

```python
builder = StateGraph(ResearchState)
builder.add_node("planner", planner)
builder.add_node("researcher", researcher)
builder.add_node("writer", writer)
builder.add_node("reviewer", reviewer)
builder.add_node("finalize", finalize)

builder.add_edge(START, "planner")            # planner 内部 Command 分发
builder.add_edge("researcher", "writer")      # 三个 researcher 并行完成后汇聚到 writer
builder.add_edge("writer", "reviewer")
builder.add_edge("finalize", END)
# reviewer 的去留由 Command 决定

graph = builder.compile(checkpointer=InMemorySaver())

if __name__ == "__main__":
    result = graph.invoke(
        {"topic": "2026 年企业采用 AI Agent 的趋势", "messages": []},
        config={"configurable": {"thread_id": "research-1"}},
    )
    print(result["draft"])
```

**体验要点**（LangSmith 里看）：
- 三个 researcher 在同一 superstep **并行**，trace 树三分叉
- writer/reviewer 循环直到 pass——打回重写的轨迹完整可回放
- 跑一半 Ctrl+C，同 thread `invoke(None)` 续跑（第 11 章）

## 31.3 实现B：deepagents 生产版

```python
from deepagents import create_deep_agent

research_agent = create_deep_agent(
    tools=[web_search],
    instructions="""你是首席研究分析师。流程：
    1. write_todos 制定研究计划（3 个角度）
    2. 每个角度用 task 派子研究员（独立上下文），结果写入 notes/*.md
    3. 汇总撰写 report.md：结论先行、引用来源
    4. 完成前核对 todos 全部勾选""",
    subagents={
        "researcher": {
            "description": "单角度深度调研，输出带来源要点",
            "prompt": "你是研究员，只调研被指派的角度，输出 5 条带来源要点。",
            "tools": [web_search],
        },
    },
)

result = research_agent.invoke(
    {"messages": [("user", "调研 2026 企业 AI Agent 采用趋势，写 800 字报告")]},
    config={"configurable": {"thread_id": "deep-research-1"}},
)
```

教学版 vs 生产版：

| | 手写版 | deepagents |
|---|---|---|
| 流程掌控 | 完全显式（角度数、打回逻辑） | 模型自主 |
| 中间产物 | 在 State 里 | 文件系统（notes/*.md） |
| 适用 | 流程即产品、要严格门禁 | 开放式研究任务 |

## 31.4 生产化要点

1. **并行限流**：Send 出 3×N 个子任务时，工具层加信号量/供应商并发配额（第 28 章）
2. **长任务**：研究任务分钟级起步，必须 Server 后台 run + Webhook 通知完成（第 24-25 章）
3. **成本护栏**：每研究员 max 轮数 + 全局 token 预算；小模型跑 researcher（第 28 章）
4. **引用审计**：报告里的来源抽查校验（防模型幻觉来源）；数据集回归（第 27 章）
5. **超时升级**：单角度卡死 → 取消该 Send 分支用部分结果成稿（业务取舍要产品确认）

## 本章小结

- planner(拆角度) → Send 并行(独立上下文) → 汇聚 → writer ⇄ reviewer 门禁 —— 完整可跑
- 上下文隔离收益具象化：研究员随便读，垃圾不进主流程
- deepagents 十行等价，适合开放式任务；强流程门禁场景手写图
- 长任务配后台 run + Webhook + 预算护栏

> 最后一个项目：企业审批工作流——确定性流程 + HITL + 时间旅行 + 部署全流程。
