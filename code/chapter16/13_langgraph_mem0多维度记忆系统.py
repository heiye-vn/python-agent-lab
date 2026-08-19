import os
import sys
from pathlib import Path
from typing import Annotated, Any

from dotenv import load_dotenv
from typing_extensions import TypedDict

# 避免 Windows 终端中文编码异常
sys.stdout.reconfigure(encoding="utf-8")
sys.path.append(str(Path(__file__).parent.parent))
# 加载 .env 环境变量
load_dotenv(Path(__file__).parent / ".env")

# 解决本地系统代理导致的 SSL: UNEXPECTED_EOF_WHILE_READING 报错
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from mem0 import MemoryClient

# 校验 API Key
mem0_api_key = os.getenv("MEM0_API_KEY")
if not mem0_api_key:
    raise ValueError("未检测到 MEM0_API_KEY，请检查 code/chapter16/.env 文件配置！")

bailian_api_key = os.getenv("ALI_BAILIAN_API_KEY")
if not bailian_api_key:
    raise ValueError(
        "未检测到 ALI_BAILIAN_API_KEY，请检查 code/chapter16/.env 文件配置！"
    )

# 1. 初始化 Mem0 客户端与大模型
mem0_client = MemoryClient(api_key=mem0_api_key)

llm = init_chat_model(
    model="qwen3.7-max-2026-05-20",
    model_provider="openai",
    base_url=os.getenv(
        "ALI_BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
    api_key=bailian_api_key,
)

# 全局/应用级公共知识库标识
GLOBAL_APP_ID = "app_global_knowledge_base"


# =====================================================================
# 2. 定义 Mem0 统一记忆沉淀工具 (Agent 自主选择维度)
# =====================================================================
@tool
def save_memory_to_mem0(
    content: str,
    target_scope: str,
    config: RunnableConfig,
) -> str:
    """
    当对话中出现需要持久化记忆的信息时调用此工具。
    :param content: 记忆的具体内容描述
    :param target_scope: 存储维度分类，可选值:
        - 'global': 全局公共常识/规则（通过 app_id 隔离，所有用户与 Agent 共享）
        - 'agent': Agent 自身的工作经验/行为规范/SOP (绑定 agent_id)
        - 'user': 用户个人的画像/习惯/偏好/背景 (绑定 user_id)
        - 'run': 仅用于当前单次任务/流程的临时状态 (绑定 run_id)
        - 'hybrid': 用户与 Agent 的特定交叉协作记忆 (多维绑定)
    """
    configurable = config.get("configurable", {})
    user_id = configurable.get("user_id")
    agent_id = configurable.get("agent_id")
    run_id = configurable.get("run_id")

    payload_kwargs: dict[str, Any] = {}

    if target_scope == "global":
        # 方案 1：全局记忆（绑定统一的 app_id）
        payload_kwargs["app_id"] = GLOBAL_APP_ID
    elif target_scope == "agent":
        # 方案 2：Agent 自身规则
        if agent_id:
            payload_kwargs["agent_id"] = agent_id
        else:
            payload_kwargs["agent_id"] = "default_agent"
    elif target_scope == "user":
        # 方案 3：用户专属画像
        if user_id:
            payload_kwargs["user_id"] = user_id
    elif target_scope == "run":
        # 方案 4：单次 Run 临时记忆
        if run_id:
            payload_kwargs["run_id"] = run_id
    elif target_scope == "hybrid":
        # 方案 5：多维交叉绑定
        if user_id:
            payload_kwargs["user_id"] = user_id
        if agent_id:
            payload_kwargs["agent_id"] = agent_id
        if run_id:
            payload_kwargs["run_id"] = run_id

    if not payload_kwargs:
        payload_kwargs["app_id"] = GLOBAL_APP_ID

    # 写入 Mem0 云端
    mem0_client.add([{"role": "user", "content": content}], **payload_kwargs)

    scope_info = f"维度: {target_scope} | 参数: {payload_kwargs}"
    print(f"\n💾 [Mem0 记忆沉淀] {scope_info}")
    print(f"   内容: {content}\n")
    return f"已成功将记忆同步至 Mem0 ({scope_info})"


tools = [save_memory_to_mem0]
model_with_tools = llm.bind_tools(tools)


# =====================================================================
# 3. 定义 LangGraph 状态与分层记忆召回节点
# =====================================================================
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def recall_and_reason_node(state: AgentState, config: RunnableConfig):
    """
    Agent 核心节点：
    1. 根据当前请求的 user_id, agent_id, run_id，动态从 Mem0 召回 4 层记忆；
    2. 将召回结果分层注入 System Prompt；
    3. LLM 综合多层记忆进行推理与工具调用。
    """
    configurable = config.get("configurable", {})
    user_id = configurable.get("user_id")
    agent_id = configurable.get("agent_id")
    run_id = configurable.get("run_id")

    latest_user_message = state["messages"][-1].content if state["messages"] else ""

    # 分层召回容器
    recalled_sections: list[str] = []

    # A. 召回全局公共常识/制度 (filters={"app_id": GLOBAL_APP_ID})
    try:
        global_hits = mem0_client.search(
            latest_user_message, filters={"app_id": GLOBAL_APP_ID}, limit=2
        )
        items = (
            global_hits.get("results", global_hits)
            if isinstance(global_hits, dict)
            else global_hits
        )
        memories = [it.get("memory") for it in items if it.get("memory")]
        if memories:
            recalled_sections.append(
                "【🌐 全局公共业务规则与常识】:\n"
                + "\n".join(f"- {m}" for m in memories)
            )
    except Exception as e:  # noqa
        pass

    # B. 召回当前 Agent 自身的 SOP 与经验准则
    if agent_id:
        try:
            agent_hits = mem0_client.search(
                latest_user_message, filters={"agent_id": agent_id}, limit=3
            )
            items = (
                agent_hits.get("results", agent_hits)
                if isinstance(agent_hits, dict)
                else agent_hits
            )
            memories = [it.get("memory") for it in items if it.get("memory")]
            if memories:
                recalled_sections.append(
                    f"【🤖 Agent 自身工作规范 (Agent: {agent_id})】:\n"
                    + "\n".join(f"- {m}" for m in memories)
                )
        except Exception:  # noqa
            pass

    # C. 召回用户长期偏好画像
    if user_id:
        try:
            user_hits = mem0_client.search(
                latest_user_message, filters={"user_id": user_id}, limit=3
            )
            items = (
                user_hits.get("results", user_hits)
                if isinstance(user_hits, dict)
                else user_hits
            )
            memories = [it.get("memory") for it in items if it.get("memory")]
            if memories:
                recalled_sections.append(
                    f"【👤 用户长期画像与偏好 (User: {user_id})】:\n"
                    + "\n".join(f"- {m}" for m in memories)
                )
        except Exception:  # noqa
            pass

    # D. 召回本次任务的单次 Run 临时状态
    if run_id:
        try:
            run_hits = mem0_client.search(
                latest_user_message, filters={"run_id": run_id}, limit=2
            )
            items = (
                run_hits.get("results", run_hits)
                if isinstance(run_hits, dict)
                else run_hits
            )
            memories = [it.get("memory") for it in items if it.get("memory")]
            if memories:
                recalled_sections.append(
                    f"【⚡ 当前流程临时上下文 (Run: {run_id})】:\n"
                    + "\n".join(f"- {m}" for m in memories)
                )
        except Exception:  # noqa
            pass

    # 组装 System Prompt
    system_text = "你是一个集成 Mem0 多层记忆体系的高级 AI 智能体。\n"
    if recalled_sections:
        system_text += "\n" + "\n\n".join(recalled_sections) + "\n\n"
    system_text += (
        "请严格结合上述各层级记忆进行友好、专业的回答。"
        "若在对话中获得了新的全局规则、Agent规范、用户画像或任务临时信息，请适时调用 save_memory_to_mem0 工具沉淀记忆。"
    )

    prompt = [SystemMessage(content=system_text)] + state["messages"]
    response = model_with_tools.invoke(prompt, config=config)
    return {"messages": [response]}


# =====================================================================
# 4. 构建与编译 LangGraph 工作流
# =====================================================================
builder = StateGraph(AgentState)
builder.add_node("agent", recall_and_reason_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition, ["tools", END])
builder.add_edge("tools", "agent")

checkpointer = InMemorySaver()
agent_graph = builder.compile(checkpointer=checkpointer)


# =====================================================================
# 5. 五大记忆方案全流程实战演示
# =====================================================================
def run_five_schemes_demo():
    print("=" * 70)
    print("【LangGraph + Mem0：五大记忆维度全场景演练】")
    print("=" * 70)

    # -------------------------------------------------------------
    # 方案 1：全局记忆演示（绑定 app_id，全员共享）
    # -------------------------------------------------------------
    print("\n" + "-" * 60)
    print("【场景 1：全局公共知识（所有 Agent 与用户共享，绑定 app_id）】")
    print("-" * 60)

    # 预置一条全局制度
    mem0_client.add(
        [
            {
                "role": "user",
                "content": "平台所有退款申请需在 3 个工作日内完成财务审核并原路退回",
            }
        ],
        app_id=GLOBAL_APP_ID,
    )

    config_global = {"configurable": {"thread_id": "thread_demo_global"}}
    q1 = "请问平台的退款审核时效是几天？退到哪里？"
    print(f"[User]: {q1}")
    res1 = agent_graph.invoke(
        {"messages": [HumanMessage(content=q1)]}, config=config_global
    )
    print(f"[AI]: {res1['messages'][-1].content}\n")

    # -------------------------------------------------------------
    # 方案 2：Agent 自身规范与 SOP 记忆（仅绑定 agent_id）
    # -------------------------------------------------------------
    print("-" * 60)
    print("【场景 2：Agent 自身的工作规范/SOP（仅绑定 agent_id）】")
    print("-" * 60)

    agent_id = "customer_service_agent"
    mem0_client.add(
        [
            {
                "role": "user",
                "content": "客服处理差评投诉时，必须首先表达诚挚歉意，并主动赠送 20 元无门槛优惠券安抚",
            }
        ],
        agent_id=agent_id,
    )

    config_agent = {
        "configurable": {
            "thread_id": "thread_demo_agent",
            "agent_id": agent_id,
        }
    }
    q2 = "我昨天收到的商品外包装严重破损了，体验极差，要求说法！"
    print(f"[User]: {q2}")
    res2 = agent_graph.invoke(
        {"messages": [HumanMessage(content=q2)]}, config=config_agent
    )
    print(f"[AI]: {res2['messages'][-1].content}\n")

    # -------------------------------------------------------------
    # 方案 3：用户专属画像记忆（仅绑定 user_id）
    # -------------------------------------------------------------
    print("-" * 60)
    print("【场景 3：用户跨会话个性化偏好（仅绑定 user_id）】")
    print("-" * 60)

    user_id = "bob_developer"
    mem0_client.add(
        [
            {
                "role": "user",
                "content": "Bob 是一名资深 Python 开发者，极度偏好使用 FastAPI 和异步编程框架",
            }
        ],
        user_id=user_id,
    )

    # 在一个全新 Thread 中提问
    config_user = {
        "configurable": {
            "thread_id": "thread_demo_user_new",
            "user_id": user_id,
        }
    }
    q3 = "我想快速写一个高性能的 Web 后端接口，推荐用什么技术方案并给个简单骨架？"
    print(f"[User (Bob)]: {q3}")
    res3 = agent_graph.invoke(
        {"messages": [HumanMessage(content=q3)]}, config=config_user
    )
    print(f"[AI]: {res3['messages'][-1].content}\n")

    # -------------------------------------------------------------
    # 方案 4：单次任务/流程临时记忆（仅绑定 run_id）
    # -------------------------------------------------------------
    print("-" * 60)
    print("【场景 4：单次 Run 任务临时状态隔离（仅绑定 run_id）】")
    print("-" * 60)

    run_id_current = "task_flight_book_9921"
    mem0_client.add(
        [
            {
                "role": "user",
                "content": "本次机票订购选定出差人员的身份证号为：110101199003072345",
            }
        ],
        run_id=run_id_current,
    )

    config_run = {
        "configurable": {
            "thread_id": "thread_demo_run",
            "run_id": run_id_current,
        }
    }
    q4 = "请核对一下我刚才提供的乘机人证件号码是多少？"
    print(f"[User (Run: {run_id_current})]: {q4}")
    res4 = agent_graph.invoke(
        {"messages": [HumanMessage(content=q4)]}, config=config_run
    )
    print(f"[AI]: {res4['messages'][-1].content}\n")

    # -------------------------------------------------------------
    # 方案 5：多维度交叉绑定记忆（user_id + agent_id + run_id）
    # -------------------------------------------------------------
    print("-" * 60)
    print("【场景 5：多维交叉绑定（user_id + agent_id + run_id）】")
    print("-" * 60)

    multi_user = "charlie_audit"
    multi_agent = "finance_audit_bot"
    multi_run = "q3_audit_process_887"

    mem0_client.add(
        [
            {
                "role": "user",
                "content": "用户 Charlie 在本次 Q3 审计中特别要求发票必须具备专用电子财务章才可报销",
            }
        ],
        user_id=multi_user,
        agent_id=multi_agent,
        run_id=multi_run,
    )

    config_hybrid = {
        "configurable": {
            "thread_id": "thread_demo_hybrid",
            "user_id": multi_user,
            "agent_id": multi_agent,
            "run_id": multi_run,
        }
    }
    q5 = "请问这次 Q3 报销对于发票盖章有什么特殊规定？"
    print(f"[User (Hybrid)]: {q5}")
    res5 = agent_graph.invoke(
        {"messages": [HumanMessage(content=q5)]}, config=config_hybrid
    )
    print(f"[AI]: {res5['messages'][-1].content}\n")

    print("=" * 70)
    print(
        "✅ 五大场景验证完成：Mem0 的全局、Agent、User、Run 与交叉记忆与 LangGraph 完美融合！"
    )
    print("=" * 70)


if __name__ == "__main__":
    run_five_schemes_demo()
