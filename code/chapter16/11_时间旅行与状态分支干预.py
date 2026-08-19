import os
import sys
from pathlib import Path
from typing import Annotated
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

# 避免 Windows 终端中文编码异常
sys.stdout.reconfigure(encoding="utf-8")

# 加载 .env 环境变量
load_dotenv(Path(__file__).parent / ".env")

# 校验 API Key
api_key = os.getenv("ALI_BAILIAN_API_KEY")
if not api_key:
    raise ValueError(
        "未检测到 ALI_BAILIAN_API_KEY，请检查 code/chapter16/.env 文件配置！"
    )

# 初始化统一的大模型客户端
llm = init_chat_model(
    model="qwen3.7-max-2026-05-20",
    model_provider="openai",
    base_url=os.getenv(
        "ALI_BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
    api_key=api_key,
)


# =====================================================================
# 1. 构建带 Checkpointer 的基础对话状态图
# =====================================================================
class TravelState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def assistant_node(state: TravelState):
    system_prompt = SystemMessage(
        content="你是一个专业的私人旅游规划定制师，请根据用户的需求和预算提供精准、富有吸引力的旅游建议。"
    )
    prompt = [system_prompt] + state["messages"]
    response = llm.invoke(prompt)
    return {"messages": [response]}


builder = StateGraph(TravelState)
builder.add_node("assistant", assistant_node)
builder.add_edge(START, "assistant")
builder.add_edge("assistant", END)

checkpointer = InMemorySaver()
travel_graph = builder.compile(checkpointer=checkpointer)


# =====================================================================
# 2. 演示流程：正常多轮 -> 查看历史快照 -> Replay 重放 -> Fork 平行分叉
# =====================================================================
def run_time_travel_demo():
    print("=" * 65)
    print("【实战：LangGraph 时间旅行 (Time Travel) 与状态分支干预】")
    print("=" * 65)

    thread_id = "trip_planning_main"
    main_config = {"configurable": {"thread_id": thread_id}}

    # -------------------------------------------------------------
    # 步骤一：正常推进两轮对话，形成检查点链
    # -------------------------------------------------------------
    print("\n" + "-" * 55)
    print("【第 1 阶段：主时间线正常推进对话】")
    print("-" * 55)

    msg1 = "我想在五一假期去海边旅游，总预算大概 3000 元人民币。"
    print(f"[User]: {msg1}")
    res1 = travel_graph.invoke(
        {"messages": [HumanMessage(content=msg1)]}, config=main_config
    )
    print(f"[AI]: {res1['messages'][-1].content}\n")

    msg2 = "请根据这个预算，为我列出 3 天的具体穷游行程。"
    print(f"[User]: {msg2}")
    res2 = travel_graph.invoke(
        {"messages": [HumanMessage(content=msg2)]}, config=main_config
    )
    print(f"[AI]: {res2['messages'][-1].content}\n")

    # -------------------------------------------------------------
    # 步骤二：遍历历史检查点 (get_state_history)
    # -------------------------------------------------------------
    print("\n" + "-" * 55)
    print("【第 2 阶段：浏览检查点历史链 (get_state_history)】")
    print("-" * 55)

    history_snapshots = list(travel_graph.get_state_history(main_config))
    print(f"主线程当前共生成了 {len(history_snapshots)} 个状态快照（按时间从新到旧）：\n")

    for idx, snap in enumerate(history_snapshots, start=1):
        ckpt_id = snap.config["configurable"]["checkpoint_id"]
        step = snap.metadata.get("step", 0)
        writes = snap.metadata.get("writes", {})
        msg_count = len(snap.values.get("messages", []))
        print(f"  快照 #{idx} | Step: {step} | Checkpoint ID: {ckpt_id}")
        print(f"    节点输出: {list(writes.keys())} | 当前消息总数: {msg_count}")

    # -------------------------------------------------------------
    # 步骤三：时光倒流 (Replay) —— 修改历史状态并重新生成
    # -------------------------------------------------------------
    print("\n" + "-" * 55)
    print("【第 3 阶段：Replay 重放 —— 回溯到第 1 轮之后并修改状态】")
    print("（场景：用户突然发了一笔奖金，要把预算从 3000 改为 30000 豪华定制游）")
    print("-" * 55)

    # 找到第 1 轮对话刚结束时的快照（对应消息数较少且包含第 1 轮回复的快照）
    target_snapshot = None
    for snap in reversed(history_snapshots):
        # 寻找第一次 assistant 执行完成的检查点
        if snap.metadata.get("step") == 1:
            target_snapshot = snap
            break

    if not target_snapshot:
        target_snapshot = history_snapshots[-2]

    target_ckpt_id = target_snapshot.config["configurable"]["checkpoint_id"]
    checkpoint_ns = target_snapshot.config["configurable"].get("checkpoint_ns", "")
    print(f"🎯 选定回溯目标快照 Checkpoint ID: {target_ckpt_id}")

    # 直接使用历史快照自身的完整 config 调用 update_state
    # update_state 会返回一个指向新写入状态的 updated_config
    updated_config = travel_graph.update_state(
        target_snapshot.config,
        {
            "messages": [
                HumanMessage(
                    content="【纠正/调整】：预算发生重大变更，总预算提升至 30000 元，要求全程五星级奢华度假酒店与海鲜大餐！"
                )
            ]
        },
    )

    # 调用 invoke(None, updated_config) 从该历史分支继续运行
    print("\n🚀 触发 Replay 重新执行...")
    replay_res = travel_graph.invoke(None, config=updated_config)
    print(f"[AI (Replay 重放结果)]:\n{replay_res['messages'][-1].content}\n")

    # -------------------------------------------------------------
    # 步骤四：平行宇宙分叉 (Fork) —— 从历史克隆新 Thread
    # -------------------------------------------------------------
    print("\n" + "-" * 55)
    print("【第 4 阶段：Fork 平行分叉 —— 开启全新独立时间线】")
    print("（场景：保留原 Thread 不变，从最初输入分叉出一个'去日本京都看樱花'的独立世界）")
    print("-" * 55)

    forked_thread_id = "trip_planning_fork_japan"
    fork_config = {
        "configurable": {
            "thread_id": forked_thread_id,  # 新的 thread_id！
            "checkpoint_id": target_ckpt_id,  # 基于历史的同一快照分叉
            "checkpoint_ns": checkpoint_ns,
        }
    }

    fork_msg = "如果我想改成去日本京都体验赏樱与温泉文化，这笔预算够吗？怎么规划？"
    print(f"[User (Fork 分支)]: {fork_msg}")

    fork_res = travel_graph.invoke(
        {"messages": [HumanMessage(content=fork_msg)]},
        config=fork_config,
    )
    print(f"[AI (Fork 分支结果)]:\n{fork_res['messages'][-1].content}\n")

    print("=" * 65)
    print("✅ 时间旅行验证完毕：Replay 修改了原线程后续分支，Fork 创造了平行的独立分支。")
    print("=" * 65)


if __name__ == "__main__":
    run_time_travel_demo()
