import os
import sys
import requests
import asyncio
from pathlib import Path

# 解决 Windows 终端 GBK 编码导致的 UnicodeEncodeError
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

load_dotenv(Path(__file__).parent / ".env")

llm = init_chat_model(
    model="qwen3.7-max-2026-05-20",
    model_provider="openai",
    base_url=os.getenv(
        "ALI_BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
    api_key=os.getenv("ALI_BAILIAN_API_KEY"),
)


@tool
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


@tool
def write_file(content):
    """
    将指定内容写入本地文件。
    :param content: 必要参数，字符串类型，用于表示需要写入文档的具体内容。
    :return：是否成功写入
    """
    output_dir = Path("py_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / "res.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"已成功写入本地文件：{file_path}。"


# 定义工具
# tools = [get_weather, write_file]
tools = [get_weather]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是天气助手，请根据用户的问题，给出相应的天气信息,并具备将结果写入文件的能力",
        ),
        ("human", "{input}"),
        (
            "placeholder",
            "{agent_scratchpad}",
        ),
    ]
)

agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True
)  # vervose 表示是否打印细节信息

# response = agent_executor.invoke(
#     {"input": "查一下北京和杭州现在的温度，并将结果写入本地的文件中。"}
# )
# print(response)


def main():
    """打印/获取 Agent 节点执行信息（推荐普通 Stream 场景）"""


for chunk in agent_executor.stream(
    {"input": "请问今天北京和杭州的天气怎么样，哪个城市更热？？"}
):
    # 1. 如果触发了工具调用
    if "actions" in chunk:
        for action in chunk["actions"]:
            print(f"🔧 [正在调用工具]: {action.tool} | 参数: {action.tool_input}")

    # 2. 如果工具返回了观察结果
    elif "steps" in chunk:
        for step in chunk["steps"]:
            print(f"📊 [工具返回结果]: {step.observation}")

    # 3. 如果输出了最终回答
    elif "output" in chunk:
        print(f"🤖 [最终回答]:\n{chunk['output']}")


async def main_():
    """逐字/打字机效果流式输出（使用 stream_events）"""
    result = ""
    # 使用 astream_events 异步迭代事件流
    async for event in agent_executor.astream_events(
        {"input": "请问今天北京和杭州的天气怎么样，哪个城市更热？？"}, version="v2"
    ):
        kind = event["event"]
        # 监听 LLM 的实时 Token 输出事件
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                print(content, end="", flush=True)  # 逐字打印到终端
                result += content
    print("\n\n完整结果：\n", result)


if __name__ == "__main__":
    main()

    # asyncio.run(main_())
