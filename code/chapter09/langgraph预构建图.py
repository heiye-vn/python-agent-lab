import json
import os
import requests
import sys
from pathlib import Path

from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_core.tools import tool


sys.stdout.reconfigure(encoding="utf-8")

load_dotenv(Path(__file__).parent / ".env")


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


print(
    f"""
name: {get_weather.name}
description: {get_weather.description}
arguments： {get_weather.args}
"""
)

from langchain.chat_models import init_chat_model

llm = init_chat_model(
    model="qwen3.7-max-2026-05-20",
    model_provider="openai",
    base_url=os.getenv(
        "ALI_BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
    api_key=os.getenv("ALI_BAILIAN_API_KEY"),
)

tools = [get_weather]

from langgraph.prebuilt import create_react_agent

agent = create_react_agent(model=llm, tools=tools)

# 获取 Mermaid 字符串
mermaid_code = agent.get_graph().draw_mermaid()
print("\n--- Mermaid 语法 ---")
print(mermaid_code)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "北京现在的天气如何？"}]}
)

print(response["messages"][-1].content)


"""
预构建图（Prebuilt Graphs）

位于 langgraph.prebuilt 模块中。是 LangGraph 针对最常见的 ReAct 架构提供的开箱即用脚手架与与设计点。

无需手动写 StateGraph、add_node、add_edge 等繁琐代码，直接组合即可。
"""
