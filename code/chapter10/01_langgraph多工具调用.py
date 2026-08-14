import os
import requests
import sys
from pathlib import Path
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent  # 该 api 已弃用，使用 create_agent
from langchain.agents import create_agent

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv(Path(__file__).parent / ".env")


class WeatherQuery(BaseModel):
    loc: str = Field(description="城市名称")


class WriteQuery(BaseModel):
    content: str = Field(description="需要写入文档的具体内容")


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


@tool(args_schema=WriteQuery)
def write_file(content):
    """
    将指定内容写入本地文件。
    :param content: 必要参数，字符串类型，用于表示需要写入文档的具体内容。
    :return：是否成功写入
    """
    # 获取项目根目录 (python-agent-lab) 下的 py_output 目录，.parents[2] 表示向上取第 3层父目录（索引从 0 开始）
    output_dir = Path(__file__).resolve().parents[2] / "py_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / "query_result.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"成功写入：{file_path}"


llm = init_chat_model(
    model="qwen3.7-max-2026-05-20",
    model_provider="openai",
    base_url=os.getenv(
        "ALI_BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
    api_key=os.getenv("ALI_BAILIAN_API_KEY"),
)

tools = [get_weather, write_file]

agent = create_agent(model=llm, tools=tools)

# 获取 Mermaid 字符串
mermaid_code = agent.get_graph().draw_mermaid()
# print("\n--- Mermaid 语法 ---")
# print(mermaid_code)

try:
    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "查询今天成都和玉溪的天气，并将天气结果写入文件",
                }
            ],
        },
        # {"recursion_limit": 4},  # 递归调用次数限制，默认为 25 次
    )
    print(response["messages"])
except GraphRecursionError:
    print("智能体由于超过最多调用次数而停止")
