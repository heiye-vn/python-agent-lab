import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent  # 较新版本
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv(Path(__file__).parent / ".env")

search_tool = TavilySearch(
    max_results=5,
    topic="general",
    tavily_api_key=os.environ["TAVILY_API_KEY"],
)

llm = init_chat_model(
    model="qwen3.7-max-2026-05-20",
    model_provider="openai",
    base_url=os.getenv(
        "ALI_BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
    api_key=os.getenv("ALI_BAILIAN_API_KEY"),
)

search_agent = create_agent(model=llm, tools=[search_tool])

response = search_agent.invoke(
    {"messages": [{"role": "user", "content": "请帮我搜索最近 DeepSeek 相关的新闻。"}]}
)

print(response["messages"][-1].content)
