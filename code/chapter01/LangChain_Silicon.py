import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from openai import OpenAI

load_dotenv(Path(__file__).parent / ".env")

model = init_chat_model(
    model="Qwen/Qwen3.6-35B-A3B",
    model_provider="openai",  # 模型提供商
    base_url=os.getenv(
        "SILICON_BASE_URL", "https://api.siliconflow.cn/v1"
    ),  # 硅基流动请求地址
    api_key=os.getenv("SILICON_API_KEY"),
)

client = OpenAI(
    api_key=os.getenv("SILICON_API_KEY"),
    base_url=os.getenv("SILICON_BASE_URL", "https://api.siliconflow.cn/v1"),
)

print("=" * 50)
result = model.invoke("你好，请问你是？")
print(result)
print(type(result))

print("-" * 50)
response = client.chat.completions.create(
    model="Qwen/Qwen3.6-35B-A3B",
    messages=[{"role": "user", "content": "你好，请问你是？"}],
)
print(response)
print(type(response))
