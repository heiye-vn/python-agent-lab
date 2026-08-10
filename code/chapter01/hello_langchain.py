import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv(Path(__file__).parent / ".env")

llm = init_chat_model(
    model="qwen3.7-max-2026-05-20",
    model_provider="openai",  # 模型提供商
    base_url=os.getenv(
        "ALI_BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
    api_key=os.getenv("ALI_BAILIAN_API_KEY"),
)

# 复杂 Agent/需要记录消息 ID 与元数据是，推荐用消息类
messages = [
    SystemMessage(content="你是一位很有帮助的编程助手。"),
    HumanMessage(content="请问我如何用SQL命令创建一个数据库表"),
]

# 简单对话/Prompt模板组装，推荐用元组简写
messages_ = [
    ("system", "你是一位很有帮助的编程助手。"),
    ("user", "请问我如何用SQL命令创建一个数据库表"),
]

response = llm.invoke(messages)
print(response.content)
