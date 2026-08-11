import os
from pathlib import Path

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

search = TavilySearch(max_results=2)

tools = [search]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一名助人为乐的助手，并且可以调用工具进行网络搜索，获取实时信息。",
        ),
        ("human", "{input}"),
        (
            "placeholder",
            "{agent_scratchpad}",
        ),  # 固定写法，为 Agent 的中间推理过程预留一个占位符
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

response = agent_executor.invoke({"input": "请问苹果2025WWDC发布会召开的时间是？"})

print(response)
