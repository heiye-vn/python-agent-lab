# 第 21 章：经典多 Agent 模式实现

本章动手实现三种生产最常用的模式：**Supervisor**（官方库）、**Swarm handoff**（官方库）、**纯 Command 手写**（理解本质）。代码均可直接运行。

## 21.1 Supervisor：langgraph-supervisor 官方库

```bash
pip install langgraph-supervisor
```

```python
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langgraph.checkpoint.memory import InMemorySaver

llm = init_chat_model("openai:gpt-4o-mini")


# 1. 定义成员 Agent 的工具（各自领域）
@tool
def web_search(query: str) -> str:
    """联网搜索最新信息。"""
    return fake_search(query)

@tool
def write_document(content: str, filename: str) -> str:
    """把内容写入文档文件。"""
    Path(filename).write_text(content, encoding="utf-8")
    return f"已写入 {filename}"

@tool
def review_document(filename: str) -> str:
    """审阅文档，返回修改意见。"""
    return "结构清晰，建议补充数据来源。"


# 2. 定义成员 Agent（每个都是标准 create_react_agent）
researcher = create_react_agent(
    llm, tools=[web_search],
    prompt="你是研究员。负责搜集事实资料，输出带来源编号的要点。只做研究，不写作。",
    name="researcher",              # name 必填：supervisor 按名字路由
)

writer = create_react_agent(
    llm, tools=[write_document],
    prompt="你是撰稿人。基于研究员的要点撰写正式文档，中文，结构清晰。",
    name="writer",
)

reviewer = create_react_agent(
    llm, tools=[review_document],
    prompt="你是审稿人。严格审阅文档，指出事实/结构/文风问题。",
    name="reviewer",
)


# 3. 组装 supervisor
workflow = create_supervisor(
    [researcher, writer, reviewer],
    model=llm,
    prompt=(
        "你是团队主管。根据用户需求把任务分配给合适的成员：\n"
        "- researcher：查资料\n- writer：写文档\n- reviewer：审稿\n"
        "分配时给出具体任务说明。所有环节完成后回复 FINISH 并汇总结果。"
    ),
    output_mode="last_message",      # 最终输出取最后一条消息
)

app = workflow.compile(checkpointer=InMemorySaver())

result = app.invoke(
    {"messages": [("user", "写一份 500 字的《2026 AI Agent 趋势》短文")]},
    config={"configurable": {"thread_id": "team-1"}},
)
print(result["messages"][-1].content)
```

运行时发生的事：supervisor（一个结构化路由 LLM）→ 派活给 researcher → 其结果回到 supervisor → 派给 writer → …… → FINISH。LangSmith 里能看到完整接力链。

关键参数补充：
- `output_mode="last_message" | "full_history"`：返回最后一条还是全部历史
- `add_handoff_back_messages=True`（默认）：成员完成后自动注入"回到主管"的消息，保证 supervisor 知道成员已返回
- 成员的 `name` 必须唯一且与 supervisor 提示中一致

## 21.2 Swarm：langgraph-swarm 官方库

Swarm 无中心主管：**Agent 之间通过交接工具（handoff tool）直接转移控制权**，适合对话式场景（客服转接、分诊）。

```bash
pip install langgraph-swarm
```

```python
from langgraph_swarm import create_swarm, create_handoff_tool

# 交接工具：本质是个"什么也不做、只触发跳转"的特殊工具
transfer_to_billing = create_handoff_tool(
    agent_name="billing_agent",
    description="用户问题涉及账单、发票、支付时转接",
)
transfer_to_tech = create_handoff_tool(
    agent_name="tech_agent",
    description="用户问题涉及产品使用、故障排查时转接",
)

billing_agent = create_react_agent(
    llm,
    tools=[query_bill, refund, transfer_to_tech],     # 自己的工具 + 交接工具
    prompt="你是账务专员。处理账单/退款；技术问题用交接工具转接。",
    name="billing_agent",
)

tech_agent = create_react_agent(
    llm,
    tools=[search_kb, transfer_to_billing],
    prompt="你是技术支持。查知识库解答；涉及费用问题转接账务。",
    name="tech_agent",
)

swarm = create_swarm(
    [billing_agent, tech_agent],
    default_active_agent="tech_agent",     # 入口分诊
)
app = swarm.compile(checkpointer=InMemorySaver())

for msg, meta in app.stream(
    {"messages": [("user", "这个月账单怎么多扣了 30 块？")]},
    {"configurable": {"thread_id": "s-1"}},
    stream_mode="messages",
):
    print(getattr(msg, "content", ""), end="")
# tech_agent 发现是费用问题 → 调 transfer_to_billing → billing_agent 接手
```

**Supervisor vs Swarm 决策表**：

| | Supervisor | Swarm |
|---|---|---|
| 决策 | 中心 LLM 统一派活 | 每个 Agent 自主交接 |
| 可控性 | 高（谁干什么有日志） | 分散 |
| token 成本 | 每轮多一次主管调用 | 少 |
| 适用 | 工作流型任务（研究/写作/审阅） | 对话分诊（客服/前台） |
| 死循环风险 | 低 | 需防 A↔B 乒乓（提示里写清"已转接过就别再转"） |

## 21.3 纯手写 Supervisor（去库化理解 + 完全掌控）

官方库是糖，底层就是第 20 章的路由模式。手写版便于定制（比如主管带记忆、成员并行）：

```python
import operator
from typing import Annotated, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command


class TeamState(MessagesState):
    findings: Annotated[list, operator.add]      # 各成员产出的汇聚黑板


class Route(BaseModel):
    next: Literal["researcher", "writer", "FINISH"]
    task: str = Field(description="指派给成员的具体任务")


router = llm.with_structured_output(Route)

def supervisor(state: TeamState) -> Command:
    decision = router.invoke([
        {"role": "system",
         "content": "你是主管。研究未完成先给 researcher；资料足够交给 writer；"
                    "文档已产出选 FINISH。"},
        *state["messages"],
    ])
    if decision.next == "FINISH":
        return Command(goto=END)
    return Command(
        goto=decision.next,
        update={"messages": [HumanMessage(decision.task, name="supervisor")]},
    )

# 成员：直接复用 create_react_agent，或手写节点
def researcher(state: TeamState) -> Command:
    result = researcher_agent.invoke({"messages": state["messages"][-1:]})
    finding = result["messages"][-1].content
    return Command(
        goto="supervisor",
        update={
            "findings": [finding],
            "messages": [HumanMessage(f"研究结果：{finding}", name="researcher")],
        },
    )

def writer(state: TeamState) -> Command:
    doc = llm.invoke(f"基于以下资料写文章：\n{state['findings']}").content
    return Command(
        goto="supervisor",
        update={"messages": [HumanMessage(f"文档已完成：{doc[:100]}...", name="writer")]},
    )

builder = StateGraph(TeamState)
builder.add_node("supervisor", supervisor)
builder.add_node("researcher", researcher)
builder.add_node("writer", writer)
builder.add_edge(START, "supervisor")            # 永远从主管开始
team = builder.compile(checkpointer=InMemorySaver())
```

控制流完全显式：`supervisor ⇄ members` 循环，`Command` 是唯一驱动力。

## 21.4 层级化（Hierarchical）：团队套团队

把一个子团队整体当作上级 supervisor 的"成员"：

```python
sub_team = create_supervisor([junior_a, junior_b], model=llm, prompt="...").compile()

top = create_supervisor(
    [sub_team_as_agent, specialist_c],   # 子团队编译后即节点
    model=llm, prompt="...",
).compile()
```

适用：组织型大任务（"研发团队"汇报给"项目主管"）。**不要超过两层**——层数越深，trace 越难读、错误越难定位。

## 21.5 并行多 Agent：研究分头、汇总合一

用第 7 章的 Send 或静态 fan-out，让多个成员**同时**干活：

```python
def fan_out_research(state) -> list[Send]:
    angles = ["技术视角", "市场视角", "风险视角"]
    return [Send("researcher", {"messages": [("user", f"从{a}分析：{state['topic']}")]})
            for a in angles]

builder.add_conditional_edges("__start__", fan_out_research, ["researcher"])
# findings 用 operator.add 自动汇聚 → synthesizer 生成总报告
```

对比串行 supervisor：延迟从 ΣT 变成 max(T)，代价是总 token 不变但并发突刺——注意 provider 限流。

## 21.6 多 Agent 工程清单

- **提示互相点名**：每个成员提示写明"你是谁、只做什么、遇到范围外如何上报/交接"
- **防乒乓**：交接提示加"若你刚被转接过来且无法处理，直接告知用户，不要再转接回去"
- **命名即协议**：成员 name、交接工具 description 要精确（路由全靠它们）
- **观测**：LangSmith 里按 trace 的嵌套 span 看接力链；`subgraphs=True` 流式区分发言人
- **成本护栏**：多 Agent 放大 token 消耗，配 recursion_limit + 单任务预算上限
- **先单体后多体**：原型期单 Agent 跑通，再按 20.1 的标准拆

## 本章小结

- Supervisor：`langgraph-supervisor` 三行组装；成员=具名 create_react_agent
- Swarm：`create_handoff_tool` + `create_swarm`，无中心对等交接，适合对话分诊
- 手写版：supervisor=结构化路由节点，Command 是唯一驱动力——官方库的糖衣之下就是它
- 层级≤2 层；并行用 Send；工程清单防乒乓、控成本

> 下一章：Deep Agents——官方"深度任务 Agent"范式，多 Agent + 规划 + 文件系统的一体化封装。
