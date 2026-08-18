import sys
import os
from pathlib import Path
from dotenv import load_dotenv

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

sys.stdout.reconfigure(encoding="utf-8")

# 加载 .env 文件中的环境变量
load_dotenv(Path(__file__).parent / ".env")


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

from langchain.chat_models import init_chat_model
from langgraph.constants import START

llm = init_chat_model(
    model="qwen3.7-max-2026-05-20",
    model_provider="openai",
    base_url=os.getenv(
        "ALI_BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
    api_key=os.getenv("ALI_BAILIAN_API_KEY"),
)


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# 简单节点/线形图中，可省略 END 节点，复杂图需要显式添加
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")

graph = graph_builder.compile()

mermaid_code = graph.get_graph().draw_mermaid()
# print(mermaid_code)

from langchain_core.messages import HumanMessage, AIMessage

messages_list = [
    HumanMessage(content="你好，我叫大模型真好玩，好久不见。"),
    AIMessage(content="你好呀！我是刘亦菲，一名女演员。很高兴认识你！"),
    HumanMessage(content="请问，你还记得我叫什么名字么？"),
]
final_state = graph.invoke({"messages": messages_list})
# print(final_state["messages"])
print(final_state["messages"][-1].content)
