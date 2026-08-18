import os
import sys
import requests
from pathlib import Path
from typing import Annotated, TypedDict
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode

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


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(AgentState)


# 定义工具
class WeatherQuery(BaseModel):
    loc: str = Field(description="城市名称")


@tool(args_schema=WeatherQuery)
def get_weather(loc):
    """
        查询即时天气函数
        :param loc: 必要参数，字符串类型，用于表示查询天气的具体城市名称，\
        :return：心知天气 API查询即时天气的结果，具体URL请求地址为："https://api.seniverse.com/v3/weather/now.json"
        返回结果对象类型为解析之后的JSON格式对象，并用字符串形式进行表示，其中包含了全部重要的天气信息
    """
    url = "https://api.seniverse.com/v3/weather/now.json"
    params = {
        "key": os.getenv("XINZHI_WEATHER_API_KEY"),
        "location": loc,
        "language": "zh-Hans",
        "unit": "c",
    }
    response = requests.get(url, params=params)
    temperature = response.json()
    return temperature["results"][0]["now"]


# 定义模型
llm = init_chat_model(
    model="qwen3.7-max-2026-05-20",
    model_provider="openai",
    base_url=os.getenv(
        "ALI_BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
    api_key=api_key,
)

tools = [get_weather]
model = llm.bind_tools(tools)


def call_model(state: AgentState):
    system_prompt = SystemMessage(
        "你是一个AI助手，可以依据用户提问产生回答，你还具备调用天气函数的能力"
    )
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}


tool_node = ToolNode(tools)


def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


graph_builder.add_node("agent", call_model)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "agent")
graph_builder.add_edge("tools", "agent")
graph_builder.add_conditional_edges(
    "agent", should_continue, {"continue": "tools", "end": END}
)

graph = graph_builder.compile()

mermaid_code = graph.get_graph().draw_mermaid()
# print(mermaid_code)


# final_state = graph.invoke({"messages": ["请问今天成都天气如何?"]})
final_state = graph.invoke({"messages": ["请问唐宋八大家分别是谁?"]})
print(final_state["messages"][-1].content)
