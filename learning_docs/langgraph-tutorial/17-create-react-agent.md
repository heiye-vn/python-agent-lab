# 第 17 章：create_react_agent 预构建 Agent

`create_react_agent` 是官方预构建的 ReAct Agent——上一章手写的那个循环的工业级封装。1.x 后它是**标准 Agent 构建入口**：默认很好用，且每个环节都可扩展。本章是完整参数手册。

## 17.1 最简用法

```python
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

llm = init_chat_model("openai:gpt-4o-mini")

agent = create_react_agent(
    llm,
    tools=[get_weather, search],
    prompt="你是一个专业的客服助手，用中文简洁回答。",
    checkpointer=InMemorySaver(),          # 有记忆
)

result = agent.invoke(
    {"messages": [("user", "上海天气如何？")]},
    config={"configurable": {"thread_id": "u-1"}},
)
print(result["messages"][-1].content)
```

内置行为：默认系统提示（"你是一个有用的助手，可用工具…"）、工具循环、工具错误回传、结构化消息协议——开箱即用。

## 17.2 参数全景

```python
agent = create_react_agent(
    model,                          # str | BaseChatModel（"openai:gpt-4o-mini" 也行）
    tools,                          # 工具列表
    prompt=...,                     # 系统提示：str | SystemMessage | callable | 消息模板
    state_schema=CustomState,       # 扩展状态（在 MessagesState 上加字段）
    pre_model_hook=fn,              # 模型调用前的钩子节点（HITL/裁剪/注入）
    post_model_hook=fn,             # 模型调用后的钩子节点
    response_format=OutputModel,    # 最终结构化输出（Pydantic）
    checkpointer=...,               # 持久化
    store=...,                      # 长期记忆
    middleware=[...],               # 中间件（第 18 章）
    config_schema=MyConfig,         # 可配置参数声明
)
```

## 17.3 prompt 的四种写法

```python
# 1. 静态字符串（最常用）
prompt="你是财务助手……"

# 2. 动态（每次调用按 state/config 生成）——上下文注入的标准姿势
def make_prompt(state, config):
    user = config["configurable"].get("user_name", "用户")
    today = date.today().isoformat()
    return [
        {"role": "system", "content": f"今天是 {today}。当前用户：{user}。"},
        {"role": "system", "content": "回答保持三句话以内。"},
    ]

agent = create_react_agent(llm, tools, prompt=make_prompt)
```

动态 prompt 追加在 `state["messages"]` **之前**，不污染消息历史。

## 17.4 扩展状态 state_schema

Agent 需要业务字段时，继承 `MessagesState`：

```python
from langgraph.graph import MessagesState

class AgentState(MessagesState):
    user_name: str
    escalation: bool
```

节点/工具里照常读写这些字段（工具用 `InjectedState` 读，见 16.4）。

## 17.5 pre_model_hook / post_model_hook：不重写循环的扩展点

两个钩子是插在循环上的官方"插槽"：

```
START → pre_model_hook → model → post_model_hook ─┬→ tools → (回到 pre_model_hook)
                                                  └→ END（无工具调用时）
```

**用例一：HITL 审批危险工具（最高频）**

```python
from langgraph.types import Command, interrupt

DANGEROUS = {"delete_account", "transfer_money"}

def approval_hook(state: AgentState) -> Command:
    last = state["messages"][-1]
    if last.tool_calls:
        dangerous = [tc for tc in last.tool_calls if tc["name"] in DANGEROUS]
        if dangerous:
            decision = interrupt({
                "type": "approval",
                "tool_calls": dangerous,       # 前端渲染审批卡片
            })
            if decision["action"] != "approve":
                # 伪造工具结果回填，模型看到"被拒绝"后自行调整
                return Command(update={
                    "messages": [ToolMessage(
                        content=f"用户拒绝了该操作：{decision.get('comment','')}",
                        tool_call_id=tc["id"],
                    ) for tc in dangerous],
                    "goto_tool_node": False,    # 跳过真实执行，回到模型
                })
    return Command(update={})                  # 无事发生，正常走

agent = create_react_agent(llm, tools, pre_model_hook=approval_hook, checkpointer=cp)
# 恢复：agent.invoke(Command(resume={"action": "approve"}), config)
```

**用例二：上下文裁剪**（第 11 章的 trim/摘要逻辑挂这里，每轮模型调用前生效）

## 17.6 response_format：最终答案结构化

让 Agent 在循环结束后输出**带 schema 的结构化结果**（与工具调用分离）：

```python
from pydantic import BaseModel

class TripPlan(BaseModel):
    destination: str
    days: int
    budget_estimates: dict[str, float]
    itinerary: list[str]

agent = create_react_agent(
    llm, tools=[search_flights, search_hotels],
    prompt="帮用户规划旅行，先查信息再给方案。",
    response_format=TripPlan,
)

result = agent.invoke({"messages": [("user", "帮我规划三天的东京之行")]})
print(result["structured_response"])
# TripPlan(destination='东京', days=3, budget_estimates={...}, itinerary=[...])
```

状态里多出 `structured_response` 字段；生成时机是模型判断信息足够、无需再调工具之后。适合"Agent 干活 + 业务要结构化结果"的管线（结果直接入库/驱动 UI）。

## 17.7 与记忆组合：一个"认识用户"的 Agent

```python
from langgraph.store.memory import InMemoryStore
from langgraph.config import get_store

@tool
def save_memory(content: str, category: str) -> str:
    """记住用户的长期信息（偏好/事实）。"""
    uid = get_config()["configurable"]["user_id"]
    get_store().put(("users", uid, category), uuid4().hex[:8], {"text": content})
    return "已记住"

def prompt_with_memories(state, config):
    uid = config["configurable"]["user_id"]
    memories = get_store().search(("users", uid, "preferences"))
    mem_text = "\n".join(f"- {m.value['text']}" for m in memories) or "（暂无）"
    return [{"role": "system", "content": f"已知的用户信息：\n{mem_text}"}]

store = InMemoryStore()

agent = create_react_agent(
    llm, tools=[save_memory],
    prompt=prompt_with_memories,
    checkpointer=InMemorySaver(),   # 短期：会话内
    store=store,                    # 长期：跨会话
)
```

短期记忆（checkpointer）+ 长期记忆（store）+ 记忆工具 = 第 12 章理论的落地形态。

## 17.8 什么时候弃用预构建、回到手写

`create_react_agent` 覆盖"单 Agent + 工具循环"的绝大多数需求。以下情况回到 StateGraph 手写（或子图组合）：

- 流程有**固定业务步骤**（先检索→必须过审核→再生成）——确定性流程不该全交给模型决策
- 需要 **Send 动态并行**等图级编排
- 多 Agent 编排（第 21 章）——但注意：多 Agent 中的每个子 Agent 通常仍是 create_react_agent

**架构经验**：企业系统 = **确定性骨架（StateGraph）+ 智能决策点（create_react_agent 子图）**，而不是一个巨型 Agent 全包。

## 17.9 常见坑

| 现象 | 原因 |
|---|---|
| 模型从不调工具 | prompt 没说明工具用途；工具 docstring 太烂；模型不支持 function calling |
| 循环停不下来 | 工具报错信息让模型反复重试；加"失败 N 次后用 fallback 答复"逻辑或 recursion_limit |
| `structured_response` 缺失 | 模型中途 tool_call 未结束；或 response_format 模型不支持 |
| hook 返回 dict 无效 | pre/post_model_hook 必须返回 `Command(update={...})`（可带 goto），不是普通 dict |

## 本章小结

- create_react_agent = 官方 ReAct 循环封装，标准 Agent 入口
- prompt 支持动态函数——上下文/记忆注入的标准位置
- pre/post_model_hook：HITL 审批、上下文裁剪等扩展的官方插槽
- response_format 输出结构化最终结果；checkpointer+store 组合双记忆
- 复杂系统 = 确定性骨架（图）+ 智能决策点（Agent 子图）

> 下一章：Middleware——1.0 时代替代大量 hooks 的横切关注点机制。
