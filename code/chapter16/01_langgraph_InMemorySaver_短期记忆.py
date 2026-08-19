import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent

# 避免 Windows 终端中文编码异常
sys.stdout.reconfigure(encoding="utf-8")

# 加载 .env 环境变量
load_dotenv(Path(__file__).parent / ".env")

bailian_api_key = os.getenv("ALI_BAILIAN_API_KEY")
if not bailian_api_key:
    raise ValueError(
        "未检测到 ALI_BAILIAN_API_KEY，请检查 code/chapter16/.env 文件配置！"
    )

checkpointer = InMemorySaver()  # 设置检查点

llm = init_chat_model(
    model="qwen3.7-plus-2026-05-26",
    model_provider="openai",
    base_url=os.getenv(
        "ALI_BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
    api_key=bailian_api_key,
)

agent = create_agent(model=llm, checkpointer=checkpointer)


"""
短期记忆是与会话线程（Session Thread）绑定的，即同一个会话线程（Session）下的所有对话都共享同一份短期记忆。

短期记忆强绑定于 thread_id，只要 thread_id 相同，即使跨进程，跨请求调用，记忆也能延续
"""

config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "你好，我叫刘浩存 ，好久不见！"}]},
    config=config,
)
print(response["messages"][-1].content)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "你还记得我叫什么名字吗？"}]},
    config=config,
)
print("----------线程1----------")
print(response["messages"][-1].content)

new_config = {"configurable": {"thread_id": "2"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "你还记得我叫什么名字吗？"}]},
    config=new_config,
)
print("----------线程2----------")
print(response["messages"][-1].content)
