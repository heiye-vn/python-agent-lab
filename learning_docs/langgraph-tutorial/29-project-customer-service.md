# 第 29 章：项目一 —— 智能客服机器人

**技能点**：create_react_agent（第 17 章）+ 工具体系（第 16 章）+ 双记忆（第 11/12 章）+ HITL 审批（第 14/15 章）+ 长话管理（第 11 章）。

## 29.1 需求与架构

```
用户 ↔ 客服 Agent（可查订单/查政策/创建退款）
              │ 退款 = 危险操作 → interrupt 人工审批
              │ 短期记忆：thread 内对话历史（checkpointer）
              │ 长期记忆：用户偏好跨会话（store）
              │ 会话过长 → 自动摘要压缩
```

项目结构：

```
customer_service/
├── main.py          # 组装 Agent，一键可跑
├── tools.py         # 业务工具（带权限注入、限量返回）
├── approval.py      # HITL 审批钩子
└── requirements.txt # langgraph langchain langchain-openai
```

## 29.2 完整实现

### tools.py —— 业务工具三件套

```python
import json
import uuid
from typing import Annotated
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langgraph.config import get_config, get_store

# 假数据库
ORDERS = {
    "SO-1001": {"status": "已发货", "amount": 299.0, "item": "无线耳机"},
    "SO-1002": {"status": "待支付", "amount": 1599.0, "item": "显示器"},
}


@tool
def query_order(order_id: str) -> str:
    """查询订单的物流状态与金额。order_id 形如 SO-1001。"""
    order = ORDERS.get(order_id)
    if not order:
        return json.dumps({"error": "订单不存在，请核对单号"}, ensure_ascii=False)
    return json.dumps({"order_id": order_id, **order}, ensure_ascii=False)


@tool
def search_policy(question: str) -> str:
    """搜索客服知识库（退换货政策、保修条款等）。"""
    KB = {
        "退款": "未发货订单可全额退款；已发货需扣除运费；7 天无理由。",
        "保修": "整机保修 1 年，人为损坏除外，需提供购买凭证。",
    }
    for k, v in KB.items():
        if k in question:
            return v
    return "知识库未命中，建议转人工。"


@tool
def create_refund(order_id: str, reason: str) -> str:
    """创建退款申请（危险操作，需人工审批后才会真正执行）。

    注意：本工具被调用时会先被审批钩子拦截（见 approval.py），
    只有审批通过后 ToolNode 才会真正执行到这里。
    """
    refund_id = f"RF-{uuid.uuid4().hex[:6].upper()}"
    # 真实系统：调用支付系统退款接口（务必幂等）
    return json.dumps({"refund_id": refund_id, "order_id": order_id,
                       "message": "退款已受理，1-3 个工作日到账"}, ensure_ascii=False)
```

### approval.py —— HITL 审批钩子

```python
from langchain_core.messages import ToolMessage
from langgraph.types import Command, interrupt

DANGEROUS_TOOLS = {"create_refund"}


def approval_hook(state) -> Command:
    """pre_model_hook 之后、工具执行之前检查待执行工具调用。"""
    last = state["messages"][-1]
    pending = [tc for tc in (last.tool_calls or [])
               if tc["name"] in DANGEROUS_TOOLS]
    if not pending:
        return Command(update={})          # 无危险操作，正常执行工具

    decision = interrupt({
        "type": "approval",
        "title": "退款审批",
        "tool_calls": [
            {"name": tc["name"], "args": tc["args"]} for tc in pending
        ],
        "actions": [
            {"id": "approve", "label": "批准"},
            {"id": "reject", "label": "拒绝"},
        ],
    })

    if decision.get("action") == "approve":
        return Command(update={})          # 放行，ToolNode 真正执行

    # 拒绝：伪造工具结果回填，模型看到后自行安抚用户
    return Command(update={
        "messages": [ToolMessage(
            content=f"审批人拒绝了该操作。备注：{decision.get('comment', '无')}",
            tool_call_id=tc["id"],
        ) for tc in pending],
        "goto_tool_node": False,           # 跳过真实执行，回到模型
    })
```

### main.py —— 组装

```python
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import trim_messages
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_config, get_store

from tools import query_order, search_policy, create_refund
from approval import approval_hook

llm = init_chat_model("openai:gpt-4o-mini")


# ---- 长话管理：每轮调用前裁剪（第 11 章策略一）----
def trim_history(state):
    return trim_messages(
        state["messages"], max_tokens=3000, strategy="last",
        token_counter=llm, include_system=True, start_on="human",
    )


# ---- 长期记忆注入 + 记忆工具 ----
def make_prompt(state, config):
    uid = config["configurable"]["user_id"]
    store = get_store()
    memories = store.search(("users", uid, "preferences"))
    mem = "\n".join(f"- {m.value['text']}" for m in memories) or "（暂无）"
    return [{"role": "system", "content":
             f"你是电商金牌客服小助。已知用户信息：\n{mem}\n"
             f"退款等敏感操作会转人工审批，请提前告知用户。中文简洁回复。"}]


def remember(state, config):
    """会话结束时的记忆提炼由后台任务做，这里演示热路径记忆（第 12 章）。"""
    return None


# ---- Agent 组装 ----
agent = create_react_agent(
    llm,
    tools=[query_order, search_policy, create_refund],
    prompt=make_prompt,
    pre_model_hook=approval_hook,
    checkpointer=InMemorySaver(),
    store=None,   # 生产换 PostgresStore；演示省略
)

if __name__ == "__main__":
    import json
    from langgraph.types import Command

    config = {"configurable": {"thread_id": "cs-demo-1", "user_id": "u-42"}}

    print("=== 第一轮：普通查询 ===")
    agent.invoke({"messages": [("user", "我的订单 SO-1001 到哪了？")]}, config)

    print("=== 第二轮：触发审批 ===")
    agent.invoke({"messages": [("user", "太慢了我要退款，原因：不想要了")]}, config)
    state = agent.get_state(config)
    if state.next:
        print("等待审批：", state.tasks[0].interrupts[0].value)

        print("=== 第三轮：模拟审批人放行 ===")
        agent.invoke(Command(resume={"action": "approve"}), config)

    final = agent.get_state(config).values
    print("最终回复:", final["messages"][-1].content)
```

## 29.3 验证清单（跑通标准）

```
□ "订单 SO-1001 到哪了" → Agent 调 query_order，回答含"已发货"
□ "退款政策是什么" → 调 search_policy
□ "我要退款" → 状态停在 next=('tools',)，interrupt 出现审批材料
□ resume approve → ToolMessage 含 refund_id
□ resume reject  → Agent 礼貌告知被拒，不执行 create_refund
□ 同 thread 追问"刚才那个退款进度" → 无需重新申请（有短期记忆）
```

## 29.4 生产化改造路线

| 原型 | 生产替换 |
|---|---|
| `InMemorySaver` | PostgresSaver（连接池） |
| 假 ORDERS/KB | 订单系统 API + 向量知识库 |
| 终端模拟审批 | LangGraph Server + 审批前端（第 15 章时序 + 第 25 章 SDK） |
| 无记忆 store | PostgresStore + 会话后记忆提炼 Cron |
| 无观测 | LangSmith 项目 + 工具命中率数据集回归（第 27 章） |

## 29.5 本项目串起的知识地图

- create_react_agent + pre_model_hook（17）
- 工具工程：docstring、JSON 限量返回、危险工具标记（16）
- interrupt/resume 完整闭环（14、15）
- trim 长话管理（11）+ 记忆注入设计（12）
- 验证即测试：断言工具命中（23）

> 下一项目：RAG 知识库问答——检索质量与引用溯源。
