import os
import sys
import requests
import asyncio
from pathlib import Path
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain.agents.structured_output import AutoStrategy

sys.stdout.reconfigure(encoding="utf-8")
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
    extra_body={"enable_thinking": False},
)


class Result(BaseModel):
    loc1: str = Field(description="起始地点名称")
    loc2: str = Field(description="目的地点名称")
    distance: float = Field(description="两地之间的距离（公里或米）")


mcp_client = MultiServerMCPClient(
    {
        "amap-maps": {
            "command": "cmd",
            "args": ["/c", "npx", "-y", "@amap/amap-maps-mcp-server"],
            "env": {"AMAP_MAPS_API_KEY": os.getenv("AMAP_MAPS_API_KEY")},
            "transport": "stdio",
        }
    }
)


async def get_server_tools():
    mcp_tools = await mcp_client.get_tools()
    print(f"加载了{len(mcp_tools)}: {[t.name for t in mcp_tools]}")

    agent_with_mcp = create_agent(
        model=llm,
        tools=mcp_tools,
        system_prompt="你是一个高德地图规划助手，需要规划行程或者获取地图基本信息.",
        response_format=AutoStrategy(Result),
    )

    result = await agent_with_mcp.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "请告诉我北京圆明园到北京八达岭长城的距离。",
                }
            ]
        }
    )

    for msg in result["messages"]:
        msg.pretty_print()

    if "structured_response" in result:
        print("\n" + "=" * 20 + " 结构化输出结果 " + "=" * 20)
        print(result["structured_response"])


asyncio.run(get_server_tools())
