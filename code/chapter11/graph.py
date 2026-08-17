import os
import sys
from dotenv import load_dotenv
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import requests

sys.stdout.reconfigure(encoding="utf-8")

# 加载 .env 文件中的环境变量
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


tools = [get_weather]

# 创建模型
llm = init_chat_model(
    model="qwen3.7-max-2026-05-20",
    model_provider="openai",
    base_url=os.getenv(
        "ALI_BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
    api_key=os.getenv("ALI_BAILIAN_API_KEY"),
)

# 创建图
graph = create_agent(model=llm, tools=tools)

if __name__ == "__main__":
    response = graph.invoke(
        {"messages": [{"role": "user", "content": "帮我查询成都当前天气情况。"}]}
    )
    print(response["messages"][-1].content)
