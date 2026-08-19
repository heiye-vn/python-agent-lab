import os
import sys
import uuid
from pathlib import Path
from typing import Annotated
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START
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
        "未检测到 ALI_BAILIAN_API_KEY，请检查 code/chapter15/.env 文件配置！"
    )


# 1. 定义状态 (State) 与 Reducer
class State(TypedDict):
    messages: Annotated[list, add_messages]


# 2. 初始化大语言模型
llm = init_chat_model(
    model="qwen3.7-max-2026-05-20",
    model_provider="openai",
    base_url=os.getenv(
        "ALI_BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
    api_key=api_key,
)


# 3. 定义节点逻辑
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# 4. 构建图并绑定 Checkpointer 实现会话持久化
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")

# 内存型检查点保存器 (Checkpointer)，负责根据 thread_id 维护状态快照
memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)


def run_demo():
    """自动化多轮对话与会话隔离演示"""
    print("=" * 60)
    print("【演示1：同一会话 thread_id='user_session_1' 的多轮记忆测试】")
    config_1 = {"configurable": {"thread_id": "user_session_1"}}

    # 第 1 轮对话
    msg1 = "你好，我是工程师小张，我正在学习 LangGraph。"
    print(f"\n[User (Session 1)]: {msg1}")
    res1 = graph.invoke({"messages": [HumanMessage(content=msg1)]}, config=config_1)
    print(f"[AI (Session 1)]: {res1['messages'][-1].content}")

    # 第 2 轮对话：测试是否能记住上轮信息
    msg2 = "请问我是谁？我正在学习什么技术？"
    print(f"\n[User (Session 1)]: {msg2}")
    res2 = graph.invoke({"messages": [HumanMessage(content=msg2)]}, config=config_1)
    print(f"[AI (Session 1)]: {res2['messages'][-1].content}")

    print("\n" + "=" * 60)
    print("【演示2：不同会话隔离性测试 thread_id='user_session_2'】")
    config_2 = {"configurable": {"thread_id": "user_session_2"}}

    # 新会话提问，不应该带有 session_1 的上下文
    msg3 = "请问你知道我叫什么名字吗？"
    print(f"\n[User (Session 2)]: {msg3}")
    res3 = graph.invoke({"messages": [HumanMessage(content=msg3)]}, config=config_2)
    print(f"[AI (Session 2)]: {res3['messages'][-1].content}")


def run_interactive_cli():
    """终端交互式多轮对话循环"""
    thread_id = f"session_{uuid.uuid4().hex[:6]}"
    config = {"configurable": {"thread_id": thread_id}}

    print("\n" + "=" * 60)
    print(f"进入实时交互式对话 (当前会话 ID: {thread_id})")
    print("输入 'exit' 或 'quit' 退出，输入 '/new' 可开启新会话")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nUser > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("已退出对话。")
                break
            if user_input.lower() == "/new":
                thread_id = f"session_{uuid.uuid4().hex[:6]}"
                config = {"configurable": {"thread_id": thread_id}}
                print(f"\n[系统] 已切换至全新会话 (ID: {thread_id})")
                continue

            # 仅需传入当次输入的新消息，Checkpointer 会自动拉取历史
            response = graph.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
            )
            print(f"AI > {response['messages'][-1].content}")
        except KeyboardInterrupt:
            print("\n已退出对话。")
            break
        except Exception as e:
            print(f"\n[错误] 对话执行异常: {e}")


if __name__ == "__main__":
    # 1. 运行自动化多轮验证
    # run_demo()

    # 2. 如果需要控制台实时交互，可取消下一行注释：
    run_interactive_cli()
