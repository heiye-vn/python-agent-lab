import os
import sys
import requests
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

# 避免 Windows 终端中文编码异常
sys.stdout.reconfigure(encoding="utf-8")
# 将 code 目录加入 sys.path，支持在各个子章节项目中直接引入 utils 通用模块
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.graph_visualizer import draw_colorized_mermaid

load_dotenv(Path(__file__).parent / ".env")

bailian_api_key = os.getenv("ALI_BAILIAN_API_KEY")
if not bailian_api_key:
    raise ValueError("未检测到 ALI_BAILIAN_API_KEY，请检查 .env 文件配置！")

llm = init_chat_model(
    model="qwen3.7-plus",
    model_provider="openai",
    base_url=os.getenv(
        "ALI_BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
    api_key=bailian_api_key,
)


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


SYSTEM_PROMPT = "你是一个天气助手，具备调用get_weather天气函数获取指定地点天气的能力"

config = {"configurable": {"thread_id": "1"}}

agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt=SYSTEM_PROMPT,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"get_weather": {"allowed_decisions": ["approve", "reject"]}}
        )
    ],
    checkpointer=InMemorySaver(),
)

# 使用公共工具渲染带节点功能语义色彩的 Mermaid 图
draw_colorized_mermaid(agent)

# 阻止后续代码执行，直接退出脚本
sys.exit(0)

result = agent.invoke({"messages": "今天成都天气如何？"}, config=config)

# 检查是否触发了人工审批中断
if "__interrupt__" in result:
    interrupt_info = result["__interrupt__"][0].value
    print("\n" + "=" * 30 + " 人工审核拦截 " + "=" * 30)
    print(f"待执行工具调用信息: {interrupt_info}")

    # 获取用户在控制台的输入
    user_choice = input("\n是否批准调用该工具？(y: 批准 / n: 拒绝): ").strip().lower()

    decision_type = "approve" if user_choice in ["y", "yes", "approve"] else "reject"

    # 根据用户的输入动态恢复执行
    result = agent.invoke(
        Command(
            resume={
                "decisions": [
                    {
                        "type": decision_type,
                        # 当选择 reject 时可附带拒绝理由
                        **(
                            {"messages": "用户拒绝了本次工具调用"}
                            if decision_type == "reject"
                            else {}
                        ),
                    }
                ]
            }
        ),
        config=config,
    )

for msg in result["messages"]:
    msg.pretty_print()
