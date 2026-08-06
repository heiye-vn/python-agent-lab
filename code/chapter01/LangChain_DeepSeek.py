import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI

load_dotenv(Path(__file__).parent / ".env")

model = init_chat_model(
    model="deepseek-v4-pro",
    model_provider="deepseek",  # 模型提供商
    base_url=os.getenv("DEEP_SEEK_BASE_URL", "https://api.deepseek.com"),
    api_key=os.getenv("DEEP_SEEK_API_KEY"),
)

client = ChatOpenAI(
    model="deepseek-v4-pro",
    api_key=os.getenv("DEEP_SEEK_API_KEY"),
    base_url=os.getenv("DEEP_SEEK_BASE_URL", "https://api.deepseek.com"),
)

print("=" * 50)
# result = model.invoke("你好，请问你是？")
result = client.invoke("你好，请问你是？")
print(result)
print(type(result))


"""
模型提供商 <==> 对应依赖包查询：
https://python.langchain.com/docs/integrations/chat/
"""
