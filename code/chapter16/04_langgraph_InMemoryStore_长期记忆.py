import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_store
from langgraph.store.memory import InMemoryStore

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv(Path(__file__).parent / ".env")

llm = init_chat_model(
    model="qwen3.7-plus-2026-05-26",
    model_provider="openai",
    base_url=os.getenv("ALI_BAILIAN_BASE_URL"),
    api_key=os.getenv("ALI_BAILIAN_API_KEY"),
)

store = InMemoryStore()  # 创建一个InMemoryStore对象

store.put(("users",), "user_123", {"name": "刘亦菲", "job": "明星，演员"})


def get_user_info(config: RunnableConfig) -> str:
    """查找用户信息，可以查看长期记忆中存储的用户信息"""
    store = get_store()
    user_id = config["configurable"].get("user_id")
    user_info = store.get(("users",), user_id)
    return str(user_info.value) if user_info else "未找到该用户信息"


agent = create_agent(model=llm, tools=[get_user_info], store=store)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "帮我查找长期记忆中存储的用户信息"}]},
    config={"configurable": {"user_id": "user_123"}},
)
print(response["messages"][-1].content)
